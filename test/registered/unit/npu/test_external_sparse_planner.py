"""External-plan ABI and per-layer validity, without importing the NPU runtime."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


PATH = Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu/external_sparse_planner.py"
SPEC = importlib.util.spec_from_file_location("fsa_external_plan_under_test", PATH)
PLANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLANNER
SPEC.loader.exec_module(PLANNER)


def test_planner_close_requires_drain_and_keeps_failed_registration_for_retry():
    planner = PLANNER.NativeSparsePlanner.__new__(PLANNER.NativeSparsePlanner)
    first, second = Mock(), Mock()
    planner.closed = False
    planner.regions = {"keys": first, "slots": second}
    with pytest.raises(RuntimeError, match="drained"):
        planner.close()
    first.close.assert_not_called()
    second.close.side_effect = RuntimeError("unregister failed")
    with pytest.raises(RuntimeError, match="unregister"):
        planner.close(drained=True)
    assert planner.regions == {"slots": second}
    with pytest.raises(RuntimeError, match="closed"):
        planner.enqueue(None, None, None)
    second.close.side_effect = None
    planner.close(drained=True)
    planner.close()
    assert not planner.regions


def layer_state(rows=2, capacity=8):
    applied = torch.full((rows, capacity + 1), -1, dtype=torch.int64)
    return SimpleNamespace(
        selection_kv_cache=torch.empty(rows * capacity // 2, 2, 4),
        max_tokens=rows, applied_keys=applied[:, :-1], applied_keys_with_padding=applied,
    ), torch.full((rows, 16400), -1, dtype=torch.int16)


def test_external_plan_encodes_misses_then_hits_and_permutations():
    cache, membership = layer_state()
    keys = torch.tensor([[100, 101, 102, 103], [200, 201, 202, 203]])
    slots = torch.tensor([[3, 1, 7, 0], [1, 5, 4, 2]])
    first = PLANNER.materialize_layer_plan(cache, keys, slots, membership)
    assert first.miss_mask.all()
    assert torch.equal(membership[:, 16380:16384], -(slots + 1).short())
    assert torch.all(membership[:, 16385] == 0x5A45)
    PLANNER.commit_layer_plan(cache, first)
    reordered = PLANNER.materialize_layer_plan(cache, keys.flip(1), slots.flip(1), membership)
    assert not reordered.miss_mask.any()
    assert torch.equal(membership[:, 16380:16384], slots.flip(1).short() + 1)


def test_shared_placement_does_not_share_kv_validity_between_layers():
    cache, membership = layer_state()
    other, other_membership = layer_state()
    keys = torch.tensor([[9, 6, 4, 2], [19, 16, 14, 12]])
    slots = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    plan = PLANNER.materialize_layer_plan(cache, keys, slots, membership)
    PLANNER.commit_layer_plan(cache, plan)
    assert not PLANNER.materialize_layer_plan(cache, keys, slots, membership).miss_mask.any()
    assert PLANNER.materialize_layer_plan(other, keys, slots, other_membership).miss_mask.all()


def test_changed_page_mapping_forces_miss_without_invalidating_unrelated_slots():
    cache, membership = layer_state()
    keys = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    slots = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    PLANNER.commit_layer_plan(cache, PLANNER.materialize_layer_plan(cache, keys, slots, membership))
    changed = keys.clone()
    changed[0, 1] = 1000
    plan = PLANNER.materialize_layer_plan(cache, changed, slots, membership)
    assert plan.miss_mask.tolist() == [[False, True, False, False], [False] * 4]


def test_padded_positions_do_not_overwrite_slot_zero():
    cache, membership = layer_state()
    keys = torch.tensor([[8, -1, -1, -1], [18, -1, -1, -1]])
    slots = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    plan = PLANNER.materialize_layer_plan(cache, keys, slots, membership)
    PLANNER.commit_layer_plan(cache, plan)
    assert torch.equal(cache.applied_keys[:, 0], keys[:, 0])
    assert torch.all(cache.applied_keys[:, 1:] == -1)
    assert torch.all(membership[:, 16381:16384] == 0)


def test_plan_requires_complete_control_storage():
    cache, _ = layer_state()
    with pytest.raises(ValueError, match="ABI"):
        PLANNER.materialize_layer_plan(cache, torch.ones((2, 4)).long(),
                                       torch.zeros((2, 4)).long(), torch.empty((2, 16384)).short())


def test_rejected_tail_reusing_the_same_physical_address_is_always_a_miss():
    cache, membership = layer_state()
    keys = torch.tensor([[8, 9, 10, 11], [18, 19, 20, 21]])
    slots = torch.arange(4).expand(2, -1)
    stable = torch.tensor([[True, True, False, False], [True, False, False, False]])
    first = PLANNER.materialize_layer_plan(cache, keys, slots, membership, cacheable_mask=stable)
    PLANNER.commit_layer_plan(cache, first)
    retry = PLANNER.materialize_layer_plan(cache, keys, slots, membership, cacheable_mask=stable)
    assert torch.equal(retry.miss_mask, ~stable)
    assert torch.all(cache.applied_keys[:, :4][~stable] == -1)
    promoted = stable.clone()
    promoted[:, 2] = True
    accepted = PLANNER.materialize_layer_plan(cache, keys, slots, membership, cacheable_mask=promoted)
    assert accepted.miss_mask[:, 2].all()
    PLANNER.commit_layer_plan(cache, accepted)
    reused = PLANNER.materialize_layer_plan(cache, keys, slots, membership, cacheable_mask=promoted)
    assert not reused.miss_mask[:, 2].any()


def test_mutable_tail_overrides_a_previously_cached_hit():
    cache, membership = layer_state()
    keys = torch.tensor([[8, 9, 10, 11], [18, 19, 20, 21]])
    slots = torch.arange(4).expand(2, -1)
    PLANNER.commit_layer_plan(cache, PLANNER.materialize_layer_plan(cache, keys, slots, membership))
    mutable = torch.zeros_like(keys, dtype=torch.bool)
    assert PLANNER.materialize_layer_plan(cache, keys, slots, membership,
                                         cacheable_mask=mutable).miss_mask.all()
