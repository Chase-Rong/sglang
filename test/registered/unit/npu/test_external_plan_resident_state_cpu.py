"""CPU contract tests for external_plan request/layer resident ownership."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODULE = Path(__file__).resolve().parents[4] / (
    "python/sglang/srt/hardware_backend/npu/hisparse/external_sparse_planner.py"
)
BUDGET_MODULE = Path(__file__).resolve().parents[4] / (
    "python/sglang/srt/hardware_backend/npu/hisparse/decode_memory_budget.py"
)


@pytest.fixture
def planner_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("external_sparse_planner", MODULE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def budget_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("decode_memory_budget", BUDGET_MODULE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_expanded_capacity_is_page_aligned_and_bounded(planner_module):
    assert planner_module.expanded_resident_capacity(4096, 6, 2051, 128) == 12416
    with pytest.raises(ValueError, match="int16 ABI"):
        planner_module.expanded_resident_capacity(32700, 6, 2051, 128)


def test_request_layer_state_separates_allocation_and_payload_validity(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 4, 16, max_routes=2, expanded_topk=8, page_size=8
    )
    request = torch.tensor([1], dtype=torch.int64)
    generation = torch.tensor([7], dtype=torch.int64)
    keys = torch.tensor([[101, 102]], dtype=torch.int64)
    slots = torch.tensor([[2, 3]], dtype=torch.int64)

    state.prepare(0, request, generation, mapping_epoch=4)
    state.record_plan(0, request, keys, slots)
    layer = state.layers[0]
    assert layer["slot_allocated"][1, 2:4].tolist() == [True, True]
    assert not layer["slot_allocated"][1, 4:].any()
    assert not layer["payload_valid"][1, 2:].any()

    lease = state.begin_execution(0, request, physical_keys=keys, slots=slots)
    state.complete(0, request, keys, slots, execution_mask=lease)
    assert layer["payload_valid"][1, 2:4].tolist() == [True, True]
    assert layer["reusable"][1, 2:4].tolist() == [True, True]


def test_generation_change_invalidates_every_allocated_layer(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=4
    )
    request = torch.tensor([1], dtype=torch.int64)
    for layer_id in (0, 1):
        state.prepare(layer_id, request, torch.tensor([3]), mapping_epoch=1)
        lease = state.begin_execution(
            layer_id,
            request,
            physical_keys=torch.tensor([[55]]),
            slots=torch.tensor([[0]]),
        )
        state.complete(
            layer_id,
            request,
            torch.tensor([[55]]),
            torch.tensor([[0]]),
            execution_mask=lease,
        )
    state.prepare(0, request, torch.tensor([4]), mapping_epoch=1)
    for layer_id in (0, 1):
        assert not state.layers[layer_id]["payload_valid"][1].any()
        assert not state.layers[layer_id]["identity"][1].ge(0).any()


def test_mapping_change_invalidates_reuse_for_every_layer(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=4
    )
    request = torch.tensor([1], dtype=torch.int64)
    for layer_id in (0, 1):
        state.prepare(layer_id, request, torch.tensor([3]), mapping_epoch=1)
        lease = state.begin_execution(
            layer_id,
            request,
            physical_keys=torch.tensor([[55]]),
            slots=torch.tensor([[0]]),
        )
        state.complete(
            layer_id,
            request,
            torch.tensor([[55]]),
            torch.tensor([[0]]),
            execution_mask=lease,
        )
    state.prepare(0, request, torch.tensor([3]), mapping_epoch=2)
    for layer_id in (0, 1):
        assert state.layers[layer_id]["payload_valid"][1, 0]
        assert not state.layers[layer_id]["reusable"][1, 0]


def test_mapping_epoch_blocks_reuse_without_global_flush(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=4
    )
    request = torch.tensor([1], dtype=torch.int64)
    generation = torch.tensor([3], dtype=torch.int64)
    keys = torch.tensor([[55]], dtype=torch.int64)
    slots = torch.tensor([[0]], dtype=torch.int64)
    state.prepare(0, request, generation, mapping_epoch=1)
    lease = state.begin_execution(0, request, physical_keys=keys, slots=slots)
    state.complete(0, request, keys, slots, execution_mask=lease)
    layer = state.layers[0]

    state.prepare(0, request, generation, mapping_epoch=2)
    assert layer["payload_valid"][1, 0]
    assert not layer["reusable"][1, 0]
    state.record_plan(0, request, keys, slots)
    assert not layer["payload_valid"][1, 0]


def test_old_mapping_completion_cannot_publish_reusable_payload(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=4
    )
    request = torch.tensor([1], dtype=torch.int64)
    keys = torch.tensor([[55]], dtype=torch.int64)
    slots = torch.tensor([[0]], dtype=torch.int64)
    state.prepare(0, request, torch.tensor([3]), mapping_epoch=1)
    lease = state.begin_execution(0, request, physical_keys=keys, slots=slots)
    state.prepare(0, request, torch.tensor([3]), mapping_epoch=2)
    state.complete(0, request, keys, slots, mapping_epoch=1, execution_mask=lease)
    assert not state.layers[0]["payload_valid"][1, 0]
    assert not state.layers[0]["reusable"][1, 0]


def test_generation_reuse_is_rejected_while_any_layer_is_in_flight(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=4
    )
    request = torch.tensor([1], dtype=torch.int64)
    state.prepare(0, request, torch.tensor([3]), mapping_epoch=1)
    state.ensure_layer(1)
    lease1 = state.begin_execution(
        1, request, physical_keys=torch.tensor([[55]]), slots=torch.tensor([[0]])
    )
    with pytest.raises(RuntimeError, match="in flight"):
        state.prepare(0, request, torch.tensor([4]), mapping_epoch=1)
    state.cancel_execution(1, request, execution_mask=lease1)


def test_generation_reuse_invalidates_old_resident_and_release_waits(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=4
    )
    request = torch.tensor([1], dtype=torch.int64)
    state.prepare(0, request, torch.tensor([3]), mapping_epoch=1)
    state.record_plan(0, request, torch.tensor([[55]]), torch.tensor([[0]]))
    lease0 = state.begin_execution(
        0, request, physical_keys=torch.tensor([[55]]), slots=torch.tensor([[0]])
    )
    state.retire(request)
    assert state.release_state[1].item() == state.RELEASE_PENDING
    state.cancel_execution(0, request, execution_mask=lease0)
    assert state.release_state[1].item() == state.RELEASED

    state.prepare(0, request, torch.tensor([4]), mapping_epoch=1)
    layer = state.layers[0]
    assert not layer["payload_valid"][1].any()
    assert not layer["identity"][1].ge(0).any()


def test_release_waits_for_all_layers(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=4
    )
    request = torch.tensor([1], dtype=torch.int64)
    state.ensure_layer(0)
    state.ensure_layer(1)
    lease0 = state.begin_execution(
        0, request, physical_keys=torch.tensor([[55]]), slots=torch.tensor([[0]])
    )
    lease1 = state.begin_execution(
        1, request, physical_keys=torch.tensor([[55]]), slots=torch.tensor([[0]])
    )
    state.retire(request)
    assert state.release_state[1].item() == state.RELEASE_PENDING
    state.cancel_execution(0, request, execution_mask=lease0)
    assert state.release_state[1].item() == state.RELEASE_PENDING
    state.cancel_execution(1, request, execution_mask=lease1)
    assert state.release_state[1].item() == state.RELEASED


def test_external_budget_charges_request_layer_metadata(budget_module):
    small_state = budget_module._external_resident_state_bytes(2, 4, 16)
    large_state = budget_module._external_resident_state_bytes(2, 4, 32)
    assert large_state > small_state
    values = dict(
        execution_abi="shared_v3",
        layers=4,
        planner_groups=2,
        latent_dim=512,
        rope_dim=0,
        index_dim=8,
        element_bytes=2,
        topk=8,
        resident_capacity=128,
        host_bytes=1024,
        backend="external_plan",
        external_resident_state_bytes=small_state,
    )
    budget = budget_module.DecodeKvMemoryBudget(**values)
    _, with_state = budget.fixed_bytes(2, 128)
    values["external_resident_state_bytes"] = 0
    _, without_state = budget_module.DecodeKvMemoryBudget(**values).fixed_bytes(2, 128)
    assert with_state - without_state == small_state


def test_duplicate_completion_aggregates_cacheability(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=2, expanded_topk=4, page_size=1
    )
    owners = torch.tensor([1, 1])
    keys = torch.tensor([[20], [20]])
    slots = torch.tensor([[0], [0]])
    for stable in (torch.tensor([[False], [True]]), torch.tensor([[True], [False]])):
        lease = state.begin_execution(
            0, owners, physical_keys=keys, slots=slots, cacheable_mask=stable
        )
        state.complete(
            0, owners, keys, slots, cacheable_mask=stable, execution_mask=lease
        )
        assert state.layers[0]["payload_valid"][1, 0]
        assert not state.layers[0]["reusable"][1, 0]


def test_fixed_shape_prepare_rejects_busy_generation_without_state_changes(
    planner_module,
):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 3, 8, max_routes=1, expanded_topk=4, page_size=1
    )
    owners = torch.tensor([1])
    state.prepare(0, owners, torch.tensor([3]), mapping_epoch=1)
    state.begin_execution(
        0, owners, physical_keys=torch.tensor([[20]]), slots=torch.tensor([[0]])
    )
    layer = state.layers[0]
    layer["identity"][1, 0] = 20
    layer["slot_allocated"][1, 0] = True
    layer["payload_valid"][1, 0] = True
    layer["reusable"][1, 0] = True
    before = {name: tensor.clone() for name, tensor in layer.items()}
    state._prepare_device(owners, torch.tensor([4]), mapping_epoch=2)
    assert state.generation[1] == 3
    assert state.mapping_epoch[1] == 1
    assert state.blocked_requests[1]
    assert state.lifecycle_error == 1
    for name, tensor in layer.items():
        assert torch.equal(tensor, before[name])
    with pytest.raises(RuntimeError, match="in flight"):
        state.validate_lifecycle()


def _lease_state(planner_module):
    state = planner_module.ExternalPlanResidentState(
        "cpu", 4, 8, max_routes=2, expanded_topk=4, page_size=1
    )
    owner = torch.tensor([1])
    state.prepare(0, owner, torch.tensor([7]), 1)
    return state, owner


@pytest.mark.parametrize("terminal", ["complete", "cancel"])
def test_execution_terminal_is_one_shot(planner_module, terminal):
    state, owner = _lease_state(planner_module)
    lease = state.begin_execution(
        0, owner, physical_keys=torch.tensor([[100]]), slots=torch.tensor([[0]])
    )
    keys, slots = torch.tensor([[100]]), torch.tensor([[0]])
    if terminal == "complete":
        state.complete(0, owner, keys, slots, execution_mask=lease)
    else:
        state.cancel_execution(0, owner, execution_mask=lease)
    before = {name: tensor.clone() for name, tensor in state.layers[0].items()}
    state.complete(0, owner, keys + 99, slots, execution_mask=lease)
    state.cancel_execution(0, owner, execution_mask=lease)
    assert state.layers[0]["in_flight"][1] == 0
    for name, value in before.items():
        assert torch.equal(state.layers[0][name], value), name


def test_late_completion_after_retire_and_reuse_cannot_publish(planner_module):
    state, owner = _lease_state(planner_module)
    old = state.begin_execution(
        0, owner, physical_keys=torch.tensor([[100]]), slots=torch.tensor([[0]])
    )
    state.cancel_execution(0, owner, execution_mask=old)
    state.retire(owner)
    state.prepare(0, owner, torch.tensor([8]), 1)
    new = state.begin_execution(
        0, owner, physical_keys=torch.tensor([[200]]), slots=torch.tensor([[0]])
    )
    state.complete(
        0, owner, torch.tensor([[200]]), torch.tensor([[0]]), execution_mask=new
    )
    state.complete(
        0, owner, torch.tensor([[100]]), torch.tensor([[0]]), execution_mask=old
    )
    assert state.layers[0]["identity"][1, 0] == 200
    assert state.layers[0]["in_flight"][1] == 0
    assert state.layers[0]["reusable"][1, 0]


def test_old_epoch_completion_has_no_destination_effect(planner_module):
    state, owner = _lease_state(planner_module)
    lease = state.begin_execution(
        0, owner, physical_keys=torch.tensor([[100]]), slots=torch.tensor([[0]])
    )
    state.prepare(0, owner, torch.tensor([7]), 2)
    state.complete(
        0,
        owner,
        torch.tensor([[100]]),
        torch.tensor([[0]]),
        mapping_epoch=1,
        execution_mask=lease,
    )
    assert state.layers[0]["identity"][1, 0] == -1
    assert not state.layers[0]["payload_valid"][1, 0]
    assert state.layers[0]["in_flight"][1] == 0


def test_cancelled_old_lease_cannot_drain_new_execution(planner_module):
    state, owner = _lease_state(planner_module)
    old = state.begin_execution(
        0, owner, physical_keys=torch.tensor([[100]]), slots=torch.tensor([[0]])
    )
    state.cancel_execution(0, owner, execution_mask=old)
    new = state.begin_execution(
        0, owner, physical_keys=torch.tensor([[100]]), slots=torch.tensor([[0]])
    )
    state.cancel_execution(0, owner, execution_mask=old)
    assert state.layers[0]["in_flight"][1] == 1
    with pytest.raises(RuntimeError, match="in flight"):
        state.prepare(0, owner, torch.tensor([8]), 1)
    state.cancel_execution(0, owner, execution_mask=new)
    assert state.layers[0]["in_flight"][1] == 0


def test_lease_requires_execution_inputs_and_its_owner(planner_module):
    state, owner = _lease_state(planner_module)
    with pytest.raises(ValueError, match="required"):
        state.begin_execution(0, owner)
    keys, slots = torch.tensor([[100]]), torch.tensor([[0]])
    lease = state.begin_execution(0, owner, physical_keys=keys, slots=slots)
    other, _ = _lease_state(planner_module)
    with pytest.raises(ValueError, match="lease"):
        other.complete(0, owner, keys, slots, execution_mask=lease)
    with pytest.raises(ValueError, match="lease"):
        state.cancel_execution(1, owner, execution_mask=lease)
    with pytest.raises(ValueError, match="lease"):
        state.complete(0, owner, keys, slots)
    assert state.layers[0]["in_flight"][1] == 1
    state.cancel_execution(0, owner, execution_mask=lease)


def test_live_lease_owns_destinations_and_second_begin_is_empty(planner_module):
    state, owner = _lease_state(planner_module)
    keys, slots = torch.tensor([[100]]), torch.tensor([[0]])
    state.record_plan(0, owner, keys, slots)
    lease = state.begin_execution(0, owner, physical_keys=keys, slots=slots)
    before = {name: tensor.clone() for name, tensor in state.layers[0].items()}
    state.record_plan(0, owner, keys + 100, slots + 1)
    second = state.begin_execution(0, owner, physical_keys=keys + 100, slots=slots + 1)
    assert not second.active.any()
    for name, value in before.items():
        assert torch.equal(state.layers[0][name], value), name
    state.complete(0, owner, keys + 100, slots + 1, execution_mask=second)
    assert state.layers[0]["in_flight"][1] == 1
    state.complete(0, owner, keys, slots, execution_mask=lease)
    assert state.layers[0]["identity"][1, 0] == 100
    assert state.layers[0]["identity"][1, 1] == -1
    assert state.layers[0]["in_flight"][1] == 0


def test_lease_snapshots_plan_workspace_and_cacheability(planner_module):
    state, owner = _lease_state(planner_module)
    keys, slots = torch.tensor([[100]]), torch.tensor([[0]])
    stable = torch.tensor([[False]])
    lease = state.begin_execution(
        0, owner, physical_keys=keys, slots=slots, cacheable_mask=stable
    )
    owner.fill_(2)
    keys.fill_(200)
    slots.fill_(1)
    stable.fill_(True)
    state.complete(0, owner, keys, slots, cacheable_mask=stable, execution_mask=lease)
    assert state.layers[0]["identity"][1, 0] == 100
    assert not state.layers[0]["reusable"][1, 0]
    assert state.layers[0]["identity"][2, 1] == -1
    assert not state.layers[0]["in_flight"].any()


def test_cross_layer_retire_drains_each_lease_and_late_terminal_is_inert(
    planner_module,
):
    state, owner = _lease_state(planner_module)
    keys, slots = torch.tensor([[100]]), torch.tensor([[0]])
    first = state.begin_execution(0, owner, physical_keys=keys, slots=slots)
    second = state.begin_execution(1, owner, physical_keys=keys + 1, slots=slots)
    state.retire(owner)
    state.complete(0, owner, keys, slots, execution_mask=first)
    assert state.release_state[1] == state.RELEASE_PENDING
    assert state.layers[0]["identity"][1, 0] == -1
    state.cancel_execution(1, owner, execution_mask=second)
    assert state.release_state[1] == state.RELEASED
    state.prepare(0, owner, torch.tensor([8]), 2)
    third = state.begin_execution(0, owner, physical_keys=keys + 2, slots=slots)
    state.cancel_execution(0, owner, execution_mask=first)
    state.complete(1, owner, keys + 1, slots, execution_mask=second)
    assert state.layers[0]["in_flight"][1] == 1
    state.complete(0, owner, keys + 2, slots, execution_mask=third)
    assert state.layers[0]["identity"][1, 0] == 102


def test_prepare_duplicate_owners_does_not_adopt_either_generation(planner_module):
    state, owner = _lease_state(planner_module)
    before = [
        state.generation.clone(),
        state.mapping_epoch.clone(),
        state.planner_version.clone(),
    ]
    state.prepare(0, torch.tensor([1, 1]), torch.tensor([8, 9]), 2)
    for actual, expected in zip(
        (state.generation, state.mapping_epoch, state.planner_version), before
    ):
        assert torch.equal(actual, expected)
    assert state.lifecycle_error != 0


def test_epoch_tensor_is_read_each_prepare_and_snapshotted_by_lease(planner_module):
    state, owner = _lease_state(planner_module)
    epoch = torch.tensor(1)
    keys, slots = torch.tensor([[100]]), torch.tensor([[0]])
    state.prepare(0, owner, torch.tensor([7]), epoch)
    first = state.begin_execution(0, owner, physical_keys=keys, slots=slots)
    epoch.fill_(2)
    state.prepare(0, owner, torch.tensor([7]), epoch)
    state.complete(0, owner, keys, slots, mapping_epoch=epoch, execution_mask=first)
    assert state.mapping_epoch[1] == 2
    assert state.layers[0]["identity"][1, 0] == -1
    next_lease = state.begin_execution(0, owner, physical_keys=keys + 100, slots=slots)
    state.complete(
        0, owner, keys + 100, slots, mapping_epoch=epoch, execution_mask=next_lease
    )
    assert state.layers[0]["identity"][1, 0] == 200
    assert state.layers[0]["reusable"][1, 0]
