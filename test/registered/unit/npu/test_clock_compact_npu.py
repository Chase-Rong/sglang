"""Compact CLOCK preserves placement and full state across eager and graphs."""

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_NPU_FSA_EXTENSION"),
    reason="requires the standalone FSA extension and an idle NPU",
)


@pytest.fixture
def device():
    import torch_npu

    index = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch.npu.set_device(index)
    return torch.device(f"npu:{index}")


@pytest.mark.parametrize(
    "rows,width,capacity",
    [(1, 1, 1), (6, 7, 8), (1, 2051, 4096), (6, 2051, 4096), (1, 2051, 32767)],
    ids=["capacity1", "small", "single", "verify", "capacity32767"],
)
def test_compact_matches_legacy_across_graphs(device, monkeypatch, rows, width, capacity):
    from sglang.srt.hardware_backend.npu.external_sparse_planner import NativeSparsePlanner

    monkeypatch.setenv("SGLANG_NPU_FSA_PLANNER_POLICY", "clock")
    planners, graphs = [], []
    try:
        for _ in range(2):
            planners.append(NativeSparsePlanner(device, rows, width, capacity))
        compact, legacy = planners
        assert compact.clock_workspace.dtype == torch.int16, "CLOCK_COMPACT_NOT_SELECTED"
        legacy.clock_workspace = legacy.clock_workspace.long()
        keys = torch.arange(rows * width, dtype=torch.int64, device=device).view(rows, width)
        owners = torch.arange(rows, dtype=torch.int64, device=device)
        lengths = torch.full((rows,), 262144, dtype=torch.int64, device=device)

        def invoke(active):
            for planner in planners:
                planner.enqueue(keys[:active], owners[:active], lengths[:active])

        def check(active):
            torch.npu.synchronize()
            for name in ("resident", "ages", "row_state", "clock_workspace"):
                assert torch.equal(getattr(compact, name).long(), getattr(legacy, name).long()), name
            actual = compact.slots[:active].cpu()
            assert torch.equal(actual, legacy.slots[:active].cpu())
            assert ((actual >= 0) & (actual < capacity)).all()
            assert all(row.unique().numel() == width for row in actual)
            expected = keys[:active].cpu()
            valid = expected >= 0
            assert torch.equal(compact.resident[:active].gather(1, actual)[valid], expected[valid])

        invoke(rows)
        check(rows)
        for active in (rows, 1):
            graph = torch.npu.NPUGraph()
            graphs.append(graph)
            with torch.npu.graph(graph):
                invoke(active)
        for step in range(36):
            values = (torch.arange(rows * width).reshape(rows, width) + step * width) * 8192
            if width > 1:
                values[:, 1] = values[:, 0]
                values[:, -1] = -1
            if step == 7:
                values[:, 0] = torch.iinfo(torch.int64).max
            keys.copy_(values.to(device))
            if step == 11:
                owners.add_(1)
            if step == 17:
                for planner in planners:
                    planner.row_state[:, 2] = torch.iinfo(torch.int64).max // 2
            if step == 23:
                lengths.sub_(1)
            if step % 3 == 2:
                invoke(rows)
                active = rows
            else:
                active = rows if step % 3 == 0 else 1
                graphs[step % 3].replay()
            check(active)
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
        for planner in planners:
            planner.close(drained=True)


@pytest.mark.parametrize("kind", ["float", "int32", "strided", "shape"])
def test_clock_rejects_invalid_workspace_before_launch(device, monkeypatch, kind):
    from sglang.srt.hardware_backend.npu.external_sparse_planner import NativeSparsePlanner

    monkeypatch.setenv("SGLANG_NPU_FSA_PLANNER_POLICY", "clock")
    planner = NativeSparsePlanner(device, rows=1, topk=7, capacity=8)
    try:
        if kind == "float":
            planner.clock_workspace = planner.clock_workspace.float()
        elif kind == "int32":
            planner.clock_workspace = planner.clock_workspace.int()
        elif kind == "strided":
            planner.clock_workspace = torch.empty((1, 82), dtype=torch.int16)[:, ::2]
        else:
            planner.clock_workspace = torch.empty((1, 40), dtype=torch.int16)
        keys = torch.arange(7, dtype=torch.int64, device=device).view(1, 7)
        owners = torch.ones(1, dtype=torch.int64, device=device)
        lengths = torch.full((1,), 100, dtype=torch.int64, device=device)
        with pytest.raises(RuntimeError, match="workspace"):
            planner.enqueue(keys, owners, lengths)
    finally:
        torch.npu.synchronize()
        planner.close(drained=True)
