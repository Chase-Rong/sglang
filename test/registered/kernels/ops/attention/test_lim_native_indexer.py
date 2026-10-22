"""Qualify the A5 native Indexer and exported scores before management fusion."""

import pytest
import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.npu_offload_device import offload_test_device
from sglang.test.npu_test_paths import run_pytest

register_npu_ci(
    est_time=20,
    suite="base-a-test-npu",
    disabled="Opt-in matched A5 LIM library and explicitly selected device required",
)


@pytest.fixture
def device():
    return offload_test_device(extensions=("lim",))


@pytest.mark.parametrize("rows,heads", [(1, 32), (6, 64)])
def test_native_indexer_matches_installed_op_and_scores(device, rows, heads):
    import torch_npu

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_indexer import (
        native_indexer_ops,
    )

    native_indexer_ops()
    generator = torch.Generator().manual_seed(771)
    query_cpu = (torch.randn(rows, heads, 128, generator=generator) * 0.1).bfloat16()
    key_cpu = (torch.randn(33, 128, 1, 128, generator=generator) * 0.1).bfloat16()
    weights_cpu = (torch.rand(rows, heads, generator=generator) * 0.2 - 0.05).bfloat16()
    query, key, weights = [
        value.to(device) for value in (query_cpu, key_cpu, weights_cpu)
    ]
    table_cpu = torch.arange(1, 33, dtype=torch.int32).flip(0).repeat(rows, 1)
    table = table_cpu.to(device)
    lengths_cpu = torch.tensor(
        [4096 - row * 137 for row in range(rows)], dtype=torch.int32
    )
    lengths = lengths_cpu.to(device)
    query_ends = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    indices = torch.empty((rows, 512), dtype=torch.int32, device=device)
    scores = torch.empty((rows, 4096), dtype=torch.int32, device=device)
    cores = 8
    workspace = torch.empty(cores * 4 * 4096 * 4, dtype=torch.uint8, device=device)

    def invoke():
        torch.ops.sgl_lim_native.indexer_out(
            query,
            key,
            weights,
            query_ends,
            lengths,
            table,
            indices,
            scores,
            workspace,
            cores,
        )

    def check():
        reference, _ = torch_npu.npu_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_key=lengths,
            block_table=table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=3,
        )
        torch.npu.synchronize()
        actual = indices.cpu().long()
        codes = scores.cpu().long() & 0xFFFFFFFF
        bits = torch.where(codes & 0x80000000 != 0, codes ^ 0x80000000, ~codes)
        decoded = bits.to(torch.int32).view(torch.float32)
        visible_cpu = lengths.cpu()
        current_query = query.cpu().float()
        current_weights = weights.cpu().float()
        for row in range(rows):
            visible = int(visible_cpu[row])
            keys = key_cpu[table_cpu[row].long()].reshape(-1, 128).float()[:visible]
            golden = (
                (current_query[row] @ keys.T).relu() * current_weights[row, :, None]
            ).sum(0)
            torch.testing.assert_close(
                decoded[row, :visible], golden, atol=0.0002, rtol=0.003
            )
            selected = actual[row][actual[row] >= 0]
            assert selected.numel() == min(512, visible)
            assert torch.unique(selected).numel() == selected.numel()
            assert torch.all(selected < visible)
            native_set = selected.sort().values
            baseline_set = reference[row].cpu().flatten().long()
            baseline_set = (
                baseline_set[(baseline_set >= 0) & (baseline_set < visible)]
                .sort()
                .values
            )
            torch.testing.assert_close(native_set, baseline_set, rtol=0, atol=0)

    invoke()
    check()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            invoke()
        for visible in (511, 0, 3077):
            lengths.fill_(visible)
            query.mul_(0.875)
            graph.replay()
            check()
    finally:
        torch.npu.synchronize()
        graph.reset()


def test_native_pooled_adapter_causal_expansion_and_replay(device):
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import (
        LimCopySfaSpec,
    )
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_indexer import (
        NativePooledIndexer,
    )

    row_count = 7
    spec = LimCopySfaSpec(16384, 6, 12416)
    indexer = NativePooledIndexer(spec, row_count, device)
    generator = torch.Generator().manual_seed(99)
    query = torch.randn(row_count, 32, 128, generator=generator).bfloat16().to(device)
    keys = torch.randn(33, 128, 1, 128, generator=generator).bfloat16().to(device)
    weights = torch.rand(row_count, 32, generator=generator).bfloat16().to(device)
    table = torch.arange(1, 33, dtype=torch.int32, device=device).repeat(row_count, 1)
    initial = torch.tensor([0, 1, 3, 2048, 2051, 4099, 16384], dtype=torch.int32)
    lengths = initial.to(device)

    def check(topk, scores):
        torch.npu.synchronize()
        lengths_cpu = lengths.cpu()
        selected = topk.cpu()
        pools = indexer.indices.cpu()
        score_cpu = scores.cpu()
        for row, visible in enumerate(lengths_cpu.tolist()):
            if visible <= 2048:
                expected = torch.arange(visible, dtype=torch.int32)
            else:
                expected = torch.cat(
                    (
                        (pools[row, :, None] * 4 + torch.arange(4)).flatten(),
                        torch.arange(visible // 4 * 4, visible),
                    )
                ).to(torch.int32)
            valid = selected[row][selected[row] >= 0]
            torch.testing.assert_close(valid, expected, rtol=0, atol=0)
            assert torch.isfinite(score_cpu[row, : visible // 4]).all()
            assert torch.isneginf(score_cpu[row, visible // 4 :]).all()

    check(*indexer.select(query, keys, weights, lengths, table))
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            topk, scores = indexer.select(query, keys, weights, lengths, table)
        for shift in (1, 2, 3):
            lengths.copy_(torch.roll(initial, shifts=shift))
            graph.replay()
            check(topk, scores)
    finally:
        torch.npu.synchronize()
        graph.reset()


if __name__ == "__main__":
    run_pytest(__file__)
