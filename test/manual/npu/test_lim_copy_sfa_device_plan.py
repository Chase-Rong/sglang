"""Small-memory NPU correctness tests for the supplied-TopK plan component."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an explicitly authorized NPU component environment",
)


def reference_module():
    spec = importlib.util.spec_from_file_location(
        "lim_plan_reference", Path(__file__).with_name("lim_copy_sfa_reference.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch,routes", [(1, 1), (1, 6), (10, 1), (10, 6)])
def test_device_plan_matches_score_union_reference(batch, routes):
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_device_plan import (
        DeviceUnionPlanner,
    )

    run_plan_reference_case(batch, routes, DeviceUnionPlanner)


def run_plan_reference_case(batch, routes, planner_type):
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import LimCopySfaSpec

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    reference = reference_module()
    capacity = 4096 if routes == 1 else 12416
    spec = LimCopySfaSpec(16384, routes, capacity)
    planner = planner_type(spec, batch + 1, batch, "npu:0")
    states = [reference.ReferenceResident(capacity, 16384,
              getattr(planner_type, "replacement_policy", "score")) for _ in range(batch)]
    rows = batch * routes
    lengths_cpu = torch.tensor(
        [16384 - routes + route + 1 for _ in range(batch) for route in range(routes)],
        dtype=torch.int32,
    )
    table_cpu = torch.arange(1, batch * 128 + 1, dtype=torch.int32).reshape(batch, 128)
    ends = torch.arange(routes, rows + 1, routes, device="npu", dtype=torch.int32)
    owners = torch.arange(1, batch + 1, device="npu", dtype=torch.int64)
    generations = torch.ones(batch + 1, device="npu", dtype=torch.int64)
    ids_cpu = torch.full((rows, 2051), -1, dtype=torch.int32)
    scores_cpu = torch.rand(rows, 4096, generator=torch.Generator().manual_seed(13))
    scores = scores_cpu.to("npu")
    table = table_cpu.to("npu")
    lengths = lengths_cpu.to("npu")
    ids = ids_cpu.to("npu")
    for iteration in range(4):
        for row in range(rows):
            shift = (row % routes) * 256 + (iteration // 2) * 512
            ids_cpu[row, :2048] = torch.arange(2048) + shift
            ids_cpu[row, 2048:2050] = lengths_cpu[row] - 2
            if routes > 1 and row % routes == routes - 1:
                ids_cpu[row, 2047] = lengths_cpu[row] - 16
        if iteration % 2:
            ids_cpu = ids_cpu.flip(1).contiguous()
        ids.copy_(ids_cpu)
        output_ids, output_slots, membership = planner.plan(
            ids,
            scores,
            ends,
            lengths,
            table,
            owners,
            generations,
            host_tokens=(batch * 128 + 1) * 128,
        )
        torch.npu.synchronize()
        plans = []
        for request in range(batch):
            subset = slice(request * routes, (request + 1) * routes)
            plan = states[request].plan(
                ids_cpu[subset],
                table_cpu[request],
                lengths_cpu[subset].tolist(),
                1,
                scores=scores_cpu[subset],
            )
            plans.append(plan)
            torch.testing.assert_close(
                output_ids[subset].cpu().long(), plan.logical_ids
            )
            torch.testing.assert_close(output_slots[subset].cpu().long(), plan.slots)
            encoded = membership[subset, 16384 - 2051 : 16384].cpu()
            torch.testing.assert_close(encoded < 0, plan.misses)
            if iteration > 0 and routes > 1:
                selected = ids_cpu[subset][-1] == lengths_cpu[subset][-1] - 16
                assert torch.all(encoded[-1, selected] > 0)
        planner.commit(owners, ends, lengths)
        torch.npu.synchronize()
        for state, plan in zip(states, plans):
            state.commit(plan)
