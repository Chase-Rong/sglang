import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "registered_host_kv.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "registered_host_kv_under_test", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
NpuRegisteredHostKvPool = _MODULE.NpuRegisteredHostKvPool
NpuLayerwiseKvSlotPool = _MODULE.NpuLayerwiseKvSlotPool


class FakeRegion:
    @classmethod
    def allocate(cls, shape, dtype, *, device_id, name):
        region = SimpleNamespace(
            shape=tuple(shape), dtype=dtype, device_id=device_id, name=name
        )
        region.closed = False
        region.close = lambda **kwargs: setattr(region, "closed", True)
        return region


class FakeKernel:
    RegisteredHostRegion = FakeRegion

    def free_shm(self, device_id):
        self.freed_device_id = device_id


def test_registered_host_adapter_keeps_components_independent(monkeypatch):
    pool = NpuRegisteredHostKvPool(device_id=3)
    fake_kernel = FakeKernel()
    pool._kernel_module = fake_kernel

    latent = pool.allocate_component(
        "latent_k", (2, 128, 512), dtype="bf16", page_size=128
    )
    indexer = pool.allocate_component(
        "indexer_k", (2, 128, 64), dtype="bf16", page_size=128
    )

    assert latent.region.shape[-1] == 512
    assert indexer.region.shape[-1] == 64
    assert pool.component("latent_k") is latent
    with pytest.raises(ValueError):
        pool.allocate_component("latent_k", (1,), dtype="bf16", page_size=128)

    with pytest.raises(RuntimeError, match="drained"):
        pool.close()
    pool.close(drained=True)
    assert latent.region.closed and indexer.region.closed
    assert not hasattr(fake_kernel, "freed_device_id")


def test_layerwise_slot_pool_has_bounded_physical_layer_dimension():
    pool = NpuLayerwiseKvSlotPool(physical_slots=2, device="cpu")
    latent = pool.allocate_component("latent_k", (128, 512), dtype=torch.float32)
    indexer = pool.allocate_component("indexer_k", (128, 64), dtype=torch.float32)

    assert tuple(latent.shape) == (2, 128, 512)
    assert tuple(indexer.shape) == (2, 128, 64)
    assert pool.slot_view("latent_k", 1).shape == (128, 512)
    with pytest.raises(IndexError):
        pool.slot_view("latent_k", 2)


def test_host_pool_close_is_local_and_pool_names_do_not_alias():
    first = NpuRegisteredHostKvPool(0)
    second = NpuRegisteredHostKvPool(0)
    first._kernel_module = second._kernel_module = FakeKernel()
    first_region = first.allocate_component(
        "latent", (4, 32), "bf16", page_size=2
    ).region
    second_region = second.allocate_component(
        "latent", (4, 32), "bf16", page_size=2
    ).region
    assert first_region.name != second_region.name
    first.close(drained=True)
    first.close(drained=True)
    assert first_region.closed and not second_region.closed
    with pytest.raises(RuntimeError, match="closed"):
        first.allocate_component("new", (4, 32), "bf16", page_size=2)


def test_host_pool_retains_unreleased_region_if_unregister_fails():
    pool = NpuRegisteredHostKvPool(0)
    pool._kernel_module = FakeKernel()
    first = pool.allocate_component("latent", (4, 32), "bf16", page_size=2).region
    second = pool.allocate_component("rope", (4, 32), "bf16", page_size=2).region

    def fail(**kwargs):
        raise RuntimeError("unregister failed")

    second.close = fail
    with pytest.raises(RuntimeError, match="unregister"):
        pool.close(drained=True)
    assert first.closed
    assert pool.component("rope").region is second
    assert not pool._closed


def test_layerwise_slot_pool_requires_matching_generation_owner():
    pool = NpuLayerwiseKvSlotPool(physical_slots=1, device="cpu")
    pool.allocate_component("latent_k", (2, 4), dtype=torch.float32)
    owner = object()
    pool.reserve_slot(0, owner)
    pool.write_slot("latent_k", 0, torch.ones((2, 4), dtype=torch.float32), owner=owner)
    assert torch.all(pool.slot_view("latent_k", 0) == 1)
    with pytest.raises(RuntimeError, match="already reserved"):
        pool.reserve_slot(0, object())
    with pytest.raises(RuntimeError, match="stale"):
        pool.release_slot(0, object())
    pool.release_slot(0, owner)


def test_layerwise_slot_pool_rejects_component_layout_mismatch():
    pool = NpuLayerwiseKvSlotPool(physical_slots=1, device="cpu")
    pool.allocate_component("latent_k", (2, 4), dtype=torch.float32)
    owner = object()
    pool.reserve_slot(0, owner)
    with pytest.raises(ValueError, match="layout mismatch"):
        pool.write_slot(
            "latent_k", 0, torch.ones((2, 3), dtype=torch.float32), owner=owner
        )


def test_layerwise_slot_rejects_writes_after_release_or_from_stale_owner():
    pool = NpuLayerwiseKvSlotPool(physical_slots=1, device="cpu")
    pool.allocate_component("latent_k", (2, 4), dtype=torch.float32)
    previous = object()
    value = torch.ones((2, 4))
    with pytest.raises(RuntimeError, match="owner"):
        pool.write_slot("latent_k", 0, value, owner=previous)
    current = object()
    pool.reserve_slot(0, current)
    with pytest.raises(RuntimeError, match="owner"):
        pool.write_slot("latent_k", 0, value, owner=previous)
    pool.write_slot("latent_k", 0, value, owner=current)
    pool.release_slot(0, current)
    with pytest.raises(RuntimeError, match="owner"):
        pool.write_slot("latent_k", 0, value, owner=current)
