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
        "shared_host_kv",
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

    def malloc(size):
        pointer = 12345 + 4096 * sum(event[0] == "malloc" for event in events)
        events.append(("malloc", size))
        return pointer

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
        malloc=malloc,
        free=lambda pointer: events.append(("free", pointer)),
        uninitialize=lambda: events.append(("uninitialize",)),
    )
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", fabric)
    monkeypatch.setattr(loaded["memfabric_host_kv"], "native_ops", lambda: operations)
    return SimpleNamespace(
        module=loaded["memfabric_host_kv"],
        backing=loaded["decode_host_kv"],
        shared_group_type=loaded["shared_host_kv"].SharedHostGroup,
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
        with pytest.raises(ValueError, match="budget"):
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


def make_backing(allocator, *, layers=1, host_budget=4096, shared_group=None):
    return allocator.backing.DecodeHostKvBacking(
        layers=layers,
        pages=1,
        page_size=128,
        latent_dim=4,
        rope_dim=0,
        dtype=torch.bfloat16,
        device_id=0,
        host_budget=host_budget,
        integrated=True,
        shared_group=shared_group,
    )


@pytest.mark.parametrize("target_first", [True, False])
def test_target_and_draft_share_domain_but_keep_storage_until_last_close(
    allocator, target_first
):
    target = make_backing(allocator, layers=2)
    draft = make_backing(allocator)
    assert len(allocator.status.configs) == 1
    target_region = target.regions[0][0]
    draft_region = draft.regions[0][0]
    assert target_region.host_ptr != draft_region.host_ptr
    assert target_region.name != draft_region.name
    target_region.tensor.fill_(3)
    draft_region.tensor.fill_(7)
    assert torch.all(target_region.tensor == 3)
    first, last = (target, draft) if target_first else (draft, target)
    first.pool.close(drained=True)
    assert ("uninitialize",) not in allocator.events
    assert not last.regions[0][0].closed
    last.pool.close(drained=True)
    assert allocator.events.count(("uninitialize",)) == 1
    assert allocator.module._active_pool is None


def test_combined_target_draft_budget_fails_before_new_allocation(allocator):
    target = make_backing(allocator, layers=2, host_budget=2560)
    before = list(allocator.events)
    with pytest.raises(ValueError, match="budget"):
        make_backing(allocator, host_budget=2560)
    assert allocator.events == before
    assert not target.regions[0][0].closed
    target.pool.close(drained=True)


def test_failed_draft_registration_keeps_target_and_releases_budget(allocator):
    target = make_backing(allocator, host_budget=2048)
    allocator.status.registration_fails = True
    with pytest.raises(RuntimeError, match="unmapped peer"):
        make_backing(allocator, host_budget=2048)
    assert ("uninitialize",) not in allocator.events
    assert not target.regions[0][0].closed
    allocator.status.registration_fails = False
    draft = make_backing(allocator, host_budget=2048)
    draft.pool.close(drained=True)
    target.pool.close(drained=True)
    assert len(allocator.status.configs) == 1


@pytest.mark.parametrize("writer", [True, False])
def test_shared_wrappers_reuse_tp_domain_and_collectively_close_each_pool(
    allocator, writer
):
    def gather(value):
        allocator.events.append(("gather", value))
        return [value, value]

    group = SimpleNamespace(
        rank_in_group=0 if writer else 1,
        world_size=2,
        cpu_group=object(),
        all_gather_object=gather,
        broadcast_object=lambda value, src: value if writer else (12345, None),
    )

    def wrapper():
        return SimpleNamespace(
            writer_rank=0,
            is_writer=writer,
            group=SimpleNamespace(**vars(group)),
            group_id=object(),
            validate_identity=lambda value: gather(value),
        )

    target = make_backing(allocator, shared_group=wrapper())
    draft = make_backing(allocator, shared_group=wrapper())
    assert len(allocator.status.configs) == 1
    start = len(allocator.events)
    target.pool.close(drained=True)
    events = allocator.events[start:]
    kinds = [event[0] for event in events]
    readers_drained = events.index(
        ("gather", "MemFabric readers drained and unregistered")
    )
    owner_freed = events.index(("gather", "MemFabric owner freed"))
    assert kinds.index("unregister") < readers_drained < owner_freed
    if writer:
        assert readers_drained < kinds.index("free") < owner_freed
    else:
        assert "free" not in kinds
    assert ("uninitialize",) not in events
    draft.pool.close(drained=True)
    assert allocator.events.count(("uninitialize",)) == 1
    if not writer:
        assert not any(event[0] in ("malloc", "free") for event in allocator.events)


def test_domain_reserves_full_host_budget_before_draft_exists(allocator):
    target = make_backing(allocator, host_budget=(1 << 30) + 1024)
    assert allocator.status.configs[0].reserve_size == 2 << 30
    target.pool.close(drained=True)


def test_individual_pool_cannot_spend_another_pools_budget(allocator):
    pool = allocator.module.MemFabricHostKvPool(0, required_bytes=1024, reserve_bytes=4096)
    with pytest.raises(ValueError, match="reserved Host KV size"):
        pool.allocate_component("too-large", (128, 8), torch.bfloat16, page_size=128)
    assert not any(event[0] == "malloc" for event in allocator.events)
    pool.close(drained=True)


@pytest.mark.parametrize("device_id,budget", [(1, 4096), (0, 8192)])
def test_incompatible_domain_does_not_reinitialize_or_damage_target(
    allocator, device_id, budget
):
    target = make_backing(allocator)
    with pytest.raises((RuntimeError, ValueError), match="domain"):
        allocator.module.MemFabricHostKvPool(
            device_id, required_bytes=1024, reserve_bytes=budget
        )
    assert len(allocator.status.configs) == 1
    assert not target.regions[0][0].closed
    target.pool.close(drained=True)


@pytest.mark.parametrize("peer_diverges", [False, True])
def test_tp_budget_rejection_is_collective_before_any_rank_raises(
    allocator, peer_diverges
):
    identities = []
    diverge = False

    def gather(value):
        identities.append(value)
        if diverge and isinstance(value, tuple) and value[0] == "memfabric-pool":
            return [value, (*value, "peer has different budget state")]
        return [value, value]

    # Skip the real constructor's NPU fence allocation, retaining its real
    # validate_identity implementation and the simulated TP transport only.
    shared = allocator.shared_group_type.__new__(allocator.shared_group_type)
    shared.writer_rank = 0
    shared.failed = False
    shared.group = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        cpu_group=object(),
        all_gather_object=gather,
        broadcast_object=lambda value, src: value,
    )
    target = make_backing(allocator, host_budget=1024, shared_group=shared)
    identities.clear()
    diverge = peer_diverges
    try:
        with pytest.raises(
            ValueError, match="ranks disagree" if peer_diverges else "budget"
        ):
            make_backing(allocator, host_budget=1024, shared_group=shared)
        assert identities, "Every rank must validate before one rank rejects its budget"
        assert len(allocator.status.configs) == 1
        assert not target.regions[0][0].closed
    finally:
        diverge = False
        shared.failed = False
        target.pool.close(drained=True)


def test_failed_tp_geometry_closes_partial_draft_without_closing_target(allocator):
    fail_geometry = False

    def gather(value):
        if (
            fail_geometry
            and isinstance(value, tuple)
            and value[:2] == ("pool-1", "decode-1-latent")
        ):
            return [value, (*value, "peer geometry differs")]
        return [value, value]

    group = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        cpu_group=object(),
        all_gather_object=gather,
        broadcast_object=lambda value, src: value,
    )

    def wrapper():
        shared = allocator.shared_group_type.__new__(allocator.shared_group_type)
        shared.writer_rank = 0
        shared.failed = False
        shared.group = group
        return shared

    target = make_backing(allocator, shared_group=wrapper())
    fail_geometry = True
    failed_group = wrapper()
    with pytest.raises(ValueError, match="ranks disagree"):
        make_backing(allocator, layers=2, shared_group=failed_group)
    assert failed_group.failed
    assert ("free", 16441) in allocator.events
    assert ("free", 12345) not in allocator.events
    assert ("uninitialize",) not in allocator.events
    assert not target.regions[0][0].closed
    fail_geometry = False
    replacement = make_backing(allocator, layers=2, shared_group=wrapper())
    replacement.pool.close(drained=True)
    target.pool.close(drained=True)
    assert allocator.events.count(("uninitialize",)) == 1
