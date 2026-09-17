"""CPU ownership checks; do not qualify runtime registration compatibility."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def region_type(monkeypatch):
    prefix = "sglang.srt.hardware_backend.npu"
    for name in ("sglang", "sglang.srt", "sglang.srt.hardware_backend", prefix):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/fsa_native.py"
    )
    spec = importlib.util.spec_from_file_location(prefix + ".fsa_native", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location(
        "unified_under_test", Path(__file__).with_name("unified_host_lab.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False
    )
    monkeypatch.setenv("SGLANG_UNIFIED_HOST_FLAGS", "268435458")
    return module.UnifiedHostRegion


@pytest.mark.parametrize("shape", [(), (0,), (-1,), (True,), (2.5,)])
def test_invalid_shape_rejected_without_registering(region_type, shape):
    with pytest.raises(ValueError, match="shape"):
        region_type.allocate(shape, torch.bfloat16)


def test_wrong_device_rejected(region_type):
    with pytest.raises(ValueError, match="target device"):
        region_type.allocate((4,), torch.bfloat16, device_id=1)


def test_registration_error_does_not_construct_tensor(region_type, monkeypatch):
    library = SimpleNamespace(unified_host_create=lambda *args: 507899)
    monkeypatch.setattr(region_type, "_library", staticmethod(lambda: library))
    with pytest.raises(RuntimeError, match="507899"):
        region_type.allocate((4,), torch.bfloat16)


def test_close_requires_drain_and_is_idempotent(region_type):
    calls = []
    region = region_type(torch.empty(4), 123, 456, 0, "test", False)
    region.library = SimpleNamespace(
        unified_host_release=lambda address: calls.append(address) or 0
    )
    with pytest.raises(RuntimeError, match="drain"):
        region.close()
    assert not calls and not region.closed
    region.close(drained=True)
    region.close(drained=True)
    assert (
        calls == [123]
        and region.closed
        and region.host_ptr == 0
        and region.device_ptr == 0
    )


def test_failed_unregister_keeps_owner_live(region_type):
    region = region_type(torch.empty(4), 123, 456, 0, "test", False)
    region.library = SimpleNamespace(unified_host_release=lambda address: 7)
    with pytest.raises(RuntimeError, match="release failed"):
        region.close(drained=True)
    assert not region.closed and region.host_ptr == 123
