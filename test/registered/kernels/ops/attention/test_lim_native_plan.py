"""Native A5 resident manager versus the request-union CPU oracle."""

import pytest
import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.npu_lim_reference import ReferenceResident
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


@pytest.mark.parametrize("valid_tokens", [0, 4])
def test_native_commit_ignores_poisoned_union_suffix(device, valid_tokens):
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import (
        LimCopySfaSpec,
    )
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeUnionPlanner,
    )

    planner = NativeUnionPlanner(LimCopySfaSpec(16384, 1, 4096), 2, 1, device)
    ids = torch.full((1, 2051), -1, dtype=torch.int32, device=device)
    ids[0, :valid_tokens] = torch.arange(valid_tokens, device=device)
    scores = torch.ones(1, 4096, device=device)
    ends = torch.tensor([1], dtype=torch.int32, device=device)
    lengths = torch.tensor([16384], dtype=torch.int32, device=device)
    table = torch.arange(128, dtype=torch.int32, device=device).view(1, 128)
    requests = torch.tensor([1], dtype=torch.int64, device=device)
    generations = torch.tensor([0, 1], dtype=torch.int64, device=device)
    planner.plan(
        ids, scores, ends, lengths, table, requests, generations, host_tokens=16384
    )
    assert int(planner.state[1, 3]) == valid_tokens
    planner.union_slots[0, valid_tokens:] = 1024
    planner.union_sources[0, valid_tokens:] = 777
    planner.union_physical[0, valid_tokens:] = 777
    planner.commit(requests, ends, lengths)
    torch.npu.synchronize()
    assert int(planner.physical[1, 1024]) == -1
    if valid_tokens:
        torch.testing.assert_close(
            planner.physical[1, :valid_tokens].cpu(),
            torch.arange(valid_tokens, dtype=torch.int64),
        )


@pytest.mark.parametrize(
    "source_tokens,score_mode",
    [(16384, "random"), (16384, "tied")],
)
def test_native_plan_eviction_generation_remap_rollback_and_padding(
    device, source_tokens, score_mode
):
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import (
        LimCopySfaSpec,
    )
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeUnionPlanner,
    )

    planner = NativeUnionPlanner(LimCopySfaSpec(source_tokens, 1, 2176), 3, 2, device)
    states = [
        ReferenceResident(2176, source_tokens, "protected_hash") for _ in range(2)
    ]
    source_base = source_tokens - 16384
    page_count = source_tokens // 128
    ids_cpu = torch.full((2, 2051), -1, dtype=torch.int32)
    ids = ids_cpu.to(device)
    scores_cpu = torch.rand(
        (2, source_tokens // 4), generator=torch.Generator().manual_seed(17)
    )
    if score_mode == "tied":
        scores_cpu = (scores_cpu * 3).floor()
    elif score_mode == "extreme":
        scores_cpu *= 1.0e31
    scores = scores_cpu.to(device)
    lengths_cpu = torch.tensor([source_tokens, source_tokens], dtype=torch.int32)
    lengths = lengths_cpu.to(device)
    table_cpu = torch.arange(1, page_count * 2 + 1, dtype=torch.int32).view(
        2, page_count
    )
    table = table_cpu.to(device)
    ends = torch.tensor([1, 2], dtype=torch.int32, device=device)
    owners = torch.tensor([1, 2], dtype=torch.int64, device=device)
    generations_cpu = torch.tensor([0, 1, 1], dtype=torch.int64)
    generations = generations_cpu.to(device)
    graph = torch.npu.NPUGraph()

    def invoke():
        output = planner.plan(
            ids,
            scores,
            ends,
            lengths,
            table,
            owners,
            generations,
            host_tokens=(page_count * 2 + 1) * 128,
        )
        planner.commit(owners, ends, lengths)
        return output

    ids_cpu[:, :2048] = torch.arange(2048) + source_base
    ids.copy_(ids_cpu)
    invoke()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        output_ids, output_slots, output_membership = invoke()
    generations_cpu.add_(1)
    try:
        for step in range(6):
            ids_cpu[:, :2048] = (torch.arange(2048) + source_base + step * 128).flip(0)
            ids_cpu[:, 2048] = ids_cpu[:, 0]
            if step == 2:
                table_cpu = table_cpu.flip(1).contiguous()
            if step == 3:
                lengths_cpu.fill_(source_base + 4096)
            if step == 4:
                generations_cpu.add_(1)
            active = step != 5
            owners[1] = 2 if active else 0
            ids.copy_(ids_cpu)
            table.copy_(table_cpu)
            lengths.copy_(lengths_cpu)
            generations.copy_(generations_cpu)
            graph.replay()
            torch.npu.synchronize()
            for request in range(2):
                plan = states[request].plan(
                    ids_cpu[request : request + 1],
                    table_cpu[request],
                    [int(lengths_cpu[request])],
                    int(generations_cpu[request + 1]),
                    scores=scores_cpu[request : request + 1],
                    active=request == 0 or active,
                )
                torch.testing.assert_close(
                    output_ids[request : request + 1].cpu().long(), plan.logical_ids
                )
                torch.testing.assert_close(
                    output_slots[request : request + 1].cpu().long(), plan.slots
                )
                torch.testing.assert_close(
                    output_membership[request : request + 1, 14333:16384].cpu() < 0,
                    plan.misses,
                )
                states[request].commit(plan)
    finally:
        torch.npu.synchronize()
        graph.reset()


if __name__ == "__main__":
    run_pytest(__file__)
