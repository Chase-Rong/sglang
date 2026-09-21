"""Opt-in standalone-extension checks on one explicitly selected idle NPU."""

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_NPU_FSA_EXTENSION"),
    reason="requires the standalone FSA extension and an explicitly selected idle device",
)


@pytest.fixture
def device():
    import torch_npu

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_registered_host_copy_roundtrip_and_graph(device, dtype):
    from sglang.srt.hardware_backend.npu import fsa_native

    source = torch.randn(16, 512, dtype=dtype, device=device)
    destination = torch.zeros_like(source)
    region = fsa_native.RegisteredHostRegion.allocate(
        (16, 512), dtype, device_id=device.index
    )
    source_ids = torch.tensor([5, 1, 9], dtype=torch.int64, device=device)
    host_ids = torch.tensor([11, 4, 0], dtype=torch.int64, device=device)
    valid = torch.tensor([True, True, False], device=device)

    def copy():
        fsa_native.copy_rows_to_registered_host(source, region, source_ids, host_ids, valid)
        fsa_native.copy_rows_from_registered_host(region, destination, host_ids, source_ids, valid)

    graph = None
    try:
        copy()
        torch.npu.synchronize()
        torch.testing.assert_close(destination[[5, 1]], source[[5, 1]], rtol=0, atol=0)
        assert torch.count_nonzero(destination[9]).item() == 0
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            copy()
        source.add_(1)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(region.tensor[[11, 4]], source[[5, 1]].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(destination[[5, 1]], source[[5, 1]], rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        region.close(drained=True)


def test_native_planner_expanded_topk_and_graph_replay(device):
    from sglang.srt.hardware_backend.npu.external_sparse_planner import NativeSparsePlanner

    planner = NativeSparsePlanner(device, rows=1, topk=2051, capacity=4096)
    keys = torch.arange(2051, dtype=torch.int64, device=device).view(1, -1) + 128
    owners = torch.tensor([17], dtype=torch.int64, device=device)
    lengths = torch.tensor([8192], dtype=torch.int64, device=device)
    graph = None
    try:
        slots = planner.enqueue(keys, owners, lengths)
        torch.npu.synchronize()
        initial = slots.cpu().clone()
        torch.testing.assert_close(planner.resident.gather(1, initial), keys.cpu())
        keys.copy_(keys.flip(-1))
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            slots = planner.enqueue(keys, owners, lengths)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(slots.cpu(), initial.flip(-1))
        owners.add_(1)
        keys.add_(10000)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(planner.resident.gather(1, slots.cpu()), keys.cpu())
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        planner.close(drained=True)


def test_clock_state_shared_by_two_graphs_and_eager(device, monkeypatch):
    from sglang.srt.hardware_backend.npu.external_sparse_planner import NativeSparsePlanner

    monkeypatch.setenv("SGLANG_NPU_FSA_PLANNER_POLICY", "clock")
    planner = NativeSparsePlanner(device, rows=6, topk=7, capacity=8)
    keys = torch.arange(42, dtype=torch.int64, device=device).view(6, 7)
    owners = torch.arange(6, dtype=torch.int64, device=device)
    lengths = torch.full((6,), 100, dtype=torch.int64, device=device)
    graphs = []

    def invoke():
        return planner.enqueue(keys, owners, lengths)

    def check():
        torch.npu.synchronize()
        slots, expected = planner.slots.cpu(), keys.cpu()
        assert ((slots >= 0) & (slots < 8)).all()
        for row in slots:
            assert row.unique().numel() == 7
        valid = expected >= 0
        assert torch.equal(planner.resident.gather(1, slots)[valid], expected[valid])

    try:
        invoke()
        check()
        for _ in range(2):
            graph = torch.npu.NPUGraph()
            graphs.append(graph)
            with torch.npu.graph(graph):
                invoke()
        for step in range(36):
            # Frequent eviction and collisions, including duplicate keys and
            # invalid positions. Alternate payloads against the same resident.
            values = (torch.arange(42).view(6, 7) + step * 3) * 8192
            values[:, 1] = values[:, 0]
            values[:, -1] = -1
            keys.copy_(values.to(device))
            if step == 11:
                owners.add_(1)
            if step == 23:
                lengths.sub_(1)
            if step % 3 == 2:
                invoke()
            else:
                graphs[step % 3].replay()
            check()
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
        planner.close(drained=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backing", ["registered_host", "npu"])
def test_copy_across_four_gib_and_graph(device, dtype, backing):
    from sglang.srt.hardware_backend.npu import fsa_native

    boundary = (1 << 32) // 1024
    rows = boundary + 9
    region = None
    graph = None
    if backing == "registered_host":
        region = fsa_native.RegisteredHostRegion.allocate(
            (rows, 512), dtype, device_id=device.index
        )
        storage = region.tensor
        pointer = region.device_ptr
    else:
        storage = torch.empty((rows, 512), dtype=dtype, device=device)
        pointer = None
    source = torch.randn(8, 512, dtype=dtype, device=device)
    destination = torch.zeros_like(source)
    small_ids = torch.arange(8, dtype=torch.int64, device=device)
    large_ids = torch.tensor(
        [0, boundary - 1, boundary, boundary + 3, rows - 1, -1, rows, boundary + 1],
        dtype=torch.int64,
        device=device,
    )
    valid = torch.tensor([1, 2, 255, 1, 1, 1, 1, 0], dtype=torch.uint8, device=device)

    def copy():
        fsa_native.unidex_copy_inplace(
            source, storage, small_ids, large_ids, valid, 1, 1, dst_ptr=pointer
        )
        fsa_native.unidex_copy_inplace(
            storage, destination, large_ids, small_ids, valid, 1, 1, src_ptr=pointer
        )

    try:
        for captured in (False, True):
            if captured:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    copy()
                source.add_(1)
                large_ids[:5].copy_(large_ids[:5].flip(0))
                graph.replay()
            else:
                copy()
            torch.npu.synchronize()
            torch.testing.assert_close(destination[:5], source[:5], rtol=0, atol=0)
            assert torch.count_nonzero(destination[5:]).item() == 0
            selected_ids = large_ids[:5].cpu() if region is not None else large_ids[:5]
            torch.testing.assert_close(
                storage[selected_ids].cpu(), source[:5].cpu(), rtol=0, atol=0
            )
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        if region is not None:
            region.close(drained=True)
