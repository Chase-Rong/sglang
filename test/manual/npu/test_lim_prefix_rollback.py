"""Verify suffix invalidation independently of attention arithmetic."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an authorized NPU component environment",
)


@pytest.mark.parametrize("preserve", [False, True])
def test_rollback_invalidates_unselected_suffix_and_preserves_generation_reset(
    preserve,
):
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import LimCopySfaSpec
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeUnionPlanner,
    )

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    planner = NativeUnionPlanner(LimCopySfaSpec(16384, 1, 4096), 2, 1, "npu:0")
    ids = torch.full((1, 2051), -1, dtype=torch.int32, device="npu")
    ids[0, :2048] = torch.arange(2048, device="npu")
    ids[0, 2048:] = torch.tensor([7000, 7950, 7970], device="npu")
    scores = torch.ones(1, 4096, device="npu")
    ends = torch.tensor([1], dtype=torch.int32, device="npu")
    lengths = torch.tensor([8000], dtype=torch.int32, device="npu")
    requests = torch.tensor([1], dtype=torch.int64, device="npu")
    generations = torch.tensor([0, 1], dtype=torch.int64, device="npu")
    table = torch.arange(128, dtype=torch.int32, device="npu").view(1, 128)

    def plan(allow=False):
        return planner.plan(
            ids,
            scores,
            ends,
            lengths,
            table,
            requests,
            generations,
            host_tokens=16384,
            mutable_tail=12,
            preserve_prefix=allow,
        )

    plan()
    planner.commit(requests, ends, lengths)
    torch.npu.synchronize()
    old_slots = planner.slots[0, 2048:].cpu().tolist()
    for token, slot in zip((7000, 7950, 7970), old_slots):
        assert planner.physical[1, slot].item() == token
    ids[0, 2048:] = -1
    lengths.fill_(7955)
    plan(preserve)
    torch.npu.synchronize()
    encoded = planner.membership[0, 14333 : 14333 + 2048].cpu()
    if preserve:
        assert torch.all(encoded > 0)
        assert planner.physical[1, old_slots[0]].item() == 7000
        assert planner.source_map[1, 7950].item() == old_slots[1]
        assert planner.source_map[1, 7970].item() == old_slots[2]
    else:
        assert torch.all(encoded < 0)
        assert planner.source_map[1, 7970].item() == -1
    assert planner.physical[1, old_slots[1]].item() == -1
    assert planner.physical[1, old_slots[2]].item() == -1
    planner.commit(requests, ends, lengths)
    generations[1] += 1
    plan(True)
    torch.npu.synchronize()
    assert planner.state[1, 2].item() == 1
    assert torch.all(planner.membership[0, 14333 : 14333 + 2048].cpu() < 0)
