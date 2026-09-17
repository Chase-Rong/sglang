"""Allocator failure cleanup and ownership tests without allocating device memory."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu"


@pytest.fixture
def allocator(monkeypatch):
    prefix = "sglang.srt.hardware_backend.npu"
    for name in ("sglang", "sglang.srt", "sglang.srt.hardware_backend", prefix):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    loaded = {}
    for name in (
        "fsa_native",
        "registered_host_kv",
        "prefill_dma",
        "memfabric_host_kv",
        "decode_host_kv",
    ):
        spec = importlib.util.spec_from_file_location(
            prefix + "." + name, ROOT / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    events = []
    status = SimpleNamespace(init=0, registration_fails=False, configs=[])

    def initialize(config):
        status.configs.append(config)
        return status.init

    def create_region(address, shape, template, device_id, shared):
        events.append(("register", shared))
        if status.registration_fails:
            raise RuntimeError("unmapped peer")
        return torch.empty(shape, dtype=template.dtype), address + 4096

    operations = SimpleNamespace(
        memfabric_region=create_region,
        prefill_dma_copy=lambda *args: events.append(("dma", args[-1])),
        unregister_memfabric_region=lambda *args: events.append(("unregister",)),
    )
    fabric = ModuleType("memfabric_hybrid")
    fabric.offload = SimpleNamespace(
        OffloadConfig=SimpleNamespace,
        Scene=SimpleNamespace(LOCAL=0, SHARED=1),
        initialize=initialize,
        malloc=lambda size: events.append(("malloc", size)) or 12345,
        free=lambda pointer: events.append(("free", pointer)),
        uninitialize=lambda: events.append(("uninitialize",)),
    )
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", fabric)
    monkeypatch.setattr(loaded["memfabric_host_kv"], "native_ops", lambda: operations)
    return SimpleNamespace(
        module=loaded["memfabric_host_kv"],
        backing=loaded["decode_host_kv"],
        events=events,
        status=status,
    )


def test_integrated_backing_uses_dma_and_requires_drained_close(allocator):
    backing = allocator.backing.DecodeHostKvBacking(
        layers=2,
        pages=1,
        page_size=128,
        latent_dim=512,
        rope_dim=0,
        dtype=torch.bfloat16,
        device_id=0,
        host_budget=1 << 20,
        integrated=True,
    )
    assert isinstance(backing.pool, allocator.module.MemFabricHostKvPool)
    assert allocator.status.configs[0].alloc_size == 1 << 30
    with pytest.raises(RuntimeError, match="drain"):
        backing.pool.close()
    backing.pool.close(drained=True)
    before = list(allocator.events)
    backing.pool.close(drained=True)
    assert allocator.events == before
    assert [event[0] for event in before[-5:]] == [
        "unregister",
        "unregister",
        "free",
        "free",
        "uninitialize",
    ]
    assert allocator.module._active_pool is None


def test_failed_registration_frees_allocation_and_backing_closes_domain(allocator):
    allocator.status.registration_fails = True
    with pytest.raises(RuntimeError, match="unmapped peer"):
        allocator.backing.DecodeHostKvBacking(
            layers=1,
            pages=1,
            page_size=128,
            latent_dim=512,
            rope_dim=0,
            dtype=torch.bfloat16,
            device_id=0,
            host_budget=1 << 20,
            integrated=True,
        )
    assert allocator.events[-2:] == [("free", 12345), ("uninitialize",)]
    assert allocator.module._active_pool is None


def test_failed_initialization_does_not_poison_later_pool(allocator):
    allocator.status.init = 1
    with pytest.raises(RuntimeError, match="initialization failed"):
        allocator.module.MemFabricHostKvPool(0, required_bytes=1024)
    assert allocator.module._active_pool is None
    allocator.status.init = 0
    pool = allocator.module.MemFabricHostKvPool(0, required_bytes=1024)
    try:
        with pytest.raises(RuntimeError, match="one integrated"):
            allocator.module.MemFabricHostKvPool(0, required_bytes=1024)
    finally:
        pool.close(drained=True)


def test_shared_reader_never_allocates_or_frees_owner_storage(allocator):
    shared = SimpleNamespace(
        writer_rank=0,
        is_writer=False,
        validate_identity=lambda value: None,
        group=SimpleNamespace(
            rank_in_group=1,
            world_size=2,
            all_gather_object=lambda value: [value, value],
            broadcast_object=lambda value, src: (12345, None),
        ),
    )
    pool = allocator.module.MemFabricHostKvPool(
        1, required_bytes=1024, shared_group=shared
    )
    try:
        pool.allocate_component("kv", (128, 4), torch.bfloat16, page_size=128)
        assert pool.physical_bytes == 0
    finally:
        pool.close(drained=True)
    assert ("register", True) in allocator.events
    assert not any(event[0] in ("malloc", "free") for event in allocator.events)
