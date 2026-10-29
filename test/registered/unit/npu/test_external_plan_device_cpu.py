"""CPU contracts for external_plan's independent device request-union planner."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
HISPARSE = ROOT / "python/sglang/srt/hardware_backend/npu/hisparse"


def load(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(name, HISPARSE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    return load(monkeypatch, "external_sparse_planner"), load(
        monkeypatch, "external_plan_device"
    )


def make_state(modules, *, requests=4, routes=2, topk=4, capacity=8):
    resident_module, device_module = modules
    resident = resident_module.ExternalPlanResidentState(
        "cpu", requests, capacity, max_routes=routes, expanded_topk=topk, page_size=1
    )
    planner = device_module.ExternalPlanReservation(
        resident, (requests - 1) * routes, topk, requests - 1
    )
    return resident, planner


























@pytest.mark.parametrize(
    "owners, ends, keys",
    [
        ([1, 1], [1, 2], [[10, 11, 12, 13], [20, 21, 22, 23]]),
        ([1, 2], [-1, 1], [[10, 11, 12, 13]]),
        ([1, 2], [2, 1], [[10, 11, 12, 13], [20, 21, 22, 23]]),
        ([1], [1], [[10, torch.iinfo(torch.int64).max, -1, -1]]),
        ([1], [1], [[10, -2, -1, -1]]),
    ],
)
def test_invalid_input_never_changes_resident_or_acquires_execution(
    modules, owners, ends, keys
):
    resident, planner = make_state(modules)
    state = resident.ensure_layer(0)
    before = {name: value.clone() for name, value in state.items()}
    validated = planner.validate_inputs(torch.tensor(keys), torch.tensor(ends), torch.tensor(owners))
    assert not validated.any()
    assert resident.lifecycle_error != 0
    for name, value in before.items():
        assert torch.equal(state[name], value), name




def test_validation_precedes_generation_adoption(modules):
    resident, planner = make_state(modules)
    resident.prepare(0, torch.tensor([1, 2]), torch.tensor([7, 8]), 1)
    before = resident.generation.clone()
    keys = torch.tensor([[10, -2, -1, -1], [20, -1, -1, -1]])
    owners, ends = torch.tensor([1, 2]), torch.tensor([1, 2])
    validated = planner.validate_inputs(keys, ends, owners)
    resident.prepare(0, validated, torch.tensor([9, 10]), 2)
    assert resident.generation[1] == before[1]
    assert resident.generation[2] == 10
