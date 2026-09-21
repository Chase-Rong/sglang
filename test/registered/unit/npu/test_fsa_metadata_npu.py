"""Exact integer metadata equivalence, including dynamic graph replay."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires an idle qualified NPU",
)


@pytest.fixture
def device():
    import torch_npu  # noqa: F401 -- registers the NPU device backend

    device = torch.device("npu", int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"]))
    torch.npu.set_device(device)
    return device


def test_metadata_preserves_causal_pages_padding_and_generation(device):
    from sglang.srt.hardware_backend.npu import fused_sparse_attention as fsa

    build = getattr(fsa, "build_external_metadata_plan", None)
    assert callable(build), "fused external metadata producer is missing"
    inputs = [
        torch.tensor(
            [[0, 127, 128, 511], [1, 255, 256, 900], [0, 1, 2, 3]], dtype=torch.int32
        ),
        torch.tensor(
            [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, -1, 23, 24, 25, 26, 27]],
            dtype=torch.int32,
        ),
        torch.tensor([512, 901], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([0, (1 << 32) + 7, 9], dtype=torch.int64),
    ]
    inputs = [x.to(device) for x in inputs]
    positions = torch.tensor([510, 900, -1], dtype=torch.int64, device=device)
    logical, physical, owners, lengths, stable = build(
        *inputs,
        positions=positions,
        page_size=128,
        mutable_tail_tokens=6,
        full_token_count=28 * 128,
    )
    assert logical[:, 0].cpu().tolist() == [
        [0, 127, 128, -1],
        [1, 255, -1, 900],
        [-1] * 4,
    ]
    assert physical.cpu().tolist() == [
        [1280, 1407, 1408, -1],
        [2561, 2815, -1, 3460],
        [-1] * 4,
    ]
    assert owners.cpu().tolist() == [((1 << 32) + 7) * 3 + 1, 29, -1]
    assert lengths.cpu().tolist() == [512, 901, 901]
    assert stable.cpu().tolist() == [
        [True, True, True, False],
        [True, True, False, False],
        [False] * 4,
    ]


def oracle(
    topk,
    table,
    lengths,
    cumulative,
    slots,
    generations,
    positions,
    page_size,
    total,
    tail,
):
    """Existing unfused CPU path; separate from the new device implementation."""
    from sglang.srt.hardware_backend.npu.fused_sparse_attention import (
        build_reference_sparse_plan,
        mask_causal_topk,
    )

    rows = topk.shape[0]
    requests = torch.bucketize(
        torch.arange(rows, dtype=torch.int32), cumulative, right=True
    )
    valid = requests < cumulative.numel()
    requests = requests.clamp(max=cumulative.numel() - 1)
    s = slots[requests]
    valid &= (s > 0) & (s < generations.numel()) & (lengths[requests] > 0)
    safe = s.clamp(0, generations.numel() - 1)
    owners = torch.where(valid, generations[safe] * generations.numel() + safe, -1)
    logical = mask_causal_topk(
        topk[:, None].to(torch.int32).masked_fill(~valid[:, None, None], -1),
        requests,
        cumulative,
        lengths,
        positions=positions,
    )
    ref = build_reference_sparse_plan(
        logical, table, lengths, cumulative, page_size=page_size
    )
    physical = ref.physical_token_ids[:, 0].masked_fill(
        ref.physical_token_ids[:, 0] >= total, -1
    )
    logical = logical.masked_fill(physical[:, None] < 0, -1)
    stable = (
        None
        if not tail
        else (logical[:, 0] >= 0)
        & (logical[:, 0] < (lengths[requests] - tail).clamp(min=0)[:, None])
    )
    return logical, physical, owners, lengths[requests], stable


@pytest.mark.parametrize(
    "rows,width,page_size", [(1, 1, 4), (6, 2051, 128), (19, 17, 7), (120, 2051, 128)]
)
@pytest.mark.parametrize("tail,with_positions", [(0, False), (6, True)])
@pytest.mark.parametrize("inference", [False, True])
def test_metadata_graph_reads_dynamic_inputs(
    device, rows, width, page_size, tail, with_positions, inference
):
    from sglang.srt.hardware_backend.npu import fused_sparse_attention as fsa

    build = getattr(fsa, "build_external_metadata_plan", None)
    assert callable(build), "fused external metadata producer is missing"
    torch.manual_seed(9821)
    batch = 20 if rows == 120 else (3 if rows > 1 else 1)
    cpu = [
        torch.randint(-4, 12 * page_size, (rows, width), dtype=torch.int32),
        torch.randint(-1, 20, (batch, 8), dtype=torch.int32),
        torch.full((batch,), 10 * page_size, dtype=torch.int32),
        torch.arange(1, 21, dtype=torch.int32) * 6
        if batch == 20
        else torch.tensor(
            [rows] if batch == 1 else [rows // 3, rows - 2, rows - 1], dtype=torch.int32
        ),
        torch.arange(1, batch + 1, dtype=torch.int64),
        torch.arange(batch + 2, dtype=torch.int64) + (1 << 32),
    ]
    if rows == 19:
        cpu[0] = cpu[0].long()
        cpu[1] = cpu[1].long()
        # Request 1 has a valid causal row selecting this page. Multiplication
        # by page_size=7 wraps int64 negative, independently of owner validity.
        cpu[1][1, 0] = 1 << 62
        cpu[0][rows // 3, 0] = 0
    # The captured page table may reserve more columns than the active view.
    table_storage = torch.full((batch, 11), -1, dtype=cpu[1].dtype)
    table_storage[:, :8].copy_(cpu[1])
    cpu[1] = table_storage[:, :8]
    positions = (
        torch.randint(-1, 10 * page_size, (rows,), dtype=torch.int64)
        if with_positions
        else None
    )
    if positions is not None and rows == 19:
        positions = positions.int()
        positions[0] = torch.iinfo(torch.int32).max
        positions[rows // 3] = 1
    graph = torch.npu.NPUGraph()
    with torch.inference_mode(inference):
        gpu = [x.to(device) for x in cpu]
        device_table = torch.full((batch, 11), -1, dtype=cpu[1].dtype, device=device)
        device_table[:, :8].copy_(cpu[1])
        gpu[1] = device_table[:, :8]
        pos = None if positions is None else positions.to(device)
        kwargs = dict(
            positions=pos,
            page_size=page_size,
            full_token_count=16 * page_size,
            mutable_tail_tokens=tail,
        )
        build(*gpu, **kwargs)
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            outputs = build(*gpu, **kwargs)
        addresses = [x.data_ptr() for x in outputs if x is not None]
        try:
            for step in range(5):
                if step:
                    cpu[0].copy_(
                        torch.randint(
                            -4, 12 * page_size, (rows, width), dtype=torch.int32
                        )
                    )
                    cpu[1][:, step % 8] = -1 if step == 1 else 18
                    cpu[2].sub_(1)
                    if step == 2:
                        cpu[2][0] = 0
                        if batch > 1:
                            # Request 1 keeps a valid slot and active query rows.
                            cpu[2][1] = 0
                    cpu[3].clamp_(max=max(rows - step, 0))
                    cpu[4][0] = [0, -1, 99, 1][step - 1]
                    cpu[5].add_(step)
                    if positions is not None:
                        positions.sub_(1)
                    for dst, src in zip(gpu, cpu):
                        dst.copy_(src)
                    if pos is not None:
                        pos.copy_(positions)
                expected = oracle(*cpu, positions, page_size, 16 * page_size, tail)
                graph.replay()
                for actual, want in zip(outputs, expected):
                    if want is None:
                        assert actual is None
                    else:
                        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
                assert addresses == [x.data_ptr() for x in outputs if x is not None]
                for actual, want in zip(gpu, cpu):
                    torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
                if pos is not None:
                    torch.testing.assert_close(pos.cpu(), positions, rtol=0, atol=0)
        finally:
            torch.npu.synchronize()
            graph.reset()
