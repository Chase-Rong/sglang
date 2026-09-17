"""CPU contracts for the branch-specific allocation adapter, not hardware proof."""

import ctypes
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from test_unified_host_contract import region_type as region_type


@pytest.fixture
def allocation(monkeypatch, request):
    request.getfixturevalue("region_type")
    path = Path(__file__).with_name("memfabric_host_lab.py")
    spec = importlib.util.spec_from_file_location("memfabric_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = (ctypes.c_ubyte * 32)()
    host = ctypes.addressof(buffer)
    calls = []
    offload = SimpleNamespace(
        malloc=lambda size: host,
        get_dva=lambda pointer: calls.append(("map", pointer)) or 456,
        free=lambda pointer: calls.append(("free", pointer)),
        uninitialize=lambda: calls.append(("uninit",)),
    )
    monkeypatch.setitem(
        sys.modules, "memfabric_hybrid", SimpleNamespace(offload=offload)
    )
    module.MemFabricHostRegion.pool_initialized = True
    return SimpleNamespace(
        region_type=module.MemFabricHostRegion,
        buffer=buffer,
        host=host,
        calls=calls,
        offload=offload,
    )


def test_branch_mapping_uses_memfabric_query(allocation):
    region = allocation.region_type.allocate((4,), torch.float32)
    assert region.tensor.data_ptr() == allocation.host
    assert region.device_ptr == 456
    assert allocation.calls == [("map", allocation.host)]
    with pytest.raises(RuntimeError, match="drain"):
        region.close()
    assert allocation.calls == [("map", allocation.host)]
    region.close(drained=True)
    region.close(drained=True)
    assert allocation.calls[-1] == ("free", allocation.host)
    assert len(allocation.calls) == 2


def test_failed_mapping_frees_allocation(allocation):
    allocation.offload.get_dva = lambda pointer: 0
    with pytest.raises(RuntimeError, match="mapping query failed"):
        allocation.region_type.allocate((4,), torch.float32)
    assert allocation.calls == [("free", allocation.host)]


def test_pool_initialization_failure_does_not_mark_ready(allocation):
    allocation.region_type.pool_initialized = False
    allocation.offload.OffloadConfig = SimpleNamespace
    allocation.offload.Scene = SimpleNamespace(LOCAL=0, SHARED=1)
    allocation.offload.initialize = lambda config: -1
    with pytest.raises(RuntimeError, match="initialization failed"):
        allocation.region_type.initialize_pool(0)
    assert not allocation.region_type.pool_initialized


def test_shared_pool_only_writer_contributes_physical_memory(allocation):
    configurations = []
    allocation.region_type.pool_initialized = False
    allocation.offload.OffloadConfig = SimpleNamespace
    allocation.offload.Scene = SimpleNamespace(LOCAL=0, SHARED=1)
    allocation.offload.initialize = lambda config: configurations.append(config) or 0
    for rank in (0, 1):
        allocation.region_type.initialize_pool(rank, rank=rank, world_size=2)
        allocation.region_type.close_pool()
    assert [config.alloc_size for config in configurations] == [1 << 30, 0]
    assert all(config.reserve_size == 1 << 30 for config in configurations)
    assert all(config.scene == 1 for config in configurations)


def test_shared_handles_do_not_fall_back_to_sysv(allocation):
    with pytest.raises(NotImplementedError):
        allocation.region_type.attach({}, device_id=0)
