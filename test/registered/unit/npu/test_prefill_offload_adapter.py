import importlib.util
import sys
import types
from importlib.machinery import PathFinder
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import npu_module_name, npu_source_path

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


ROOT = Path(__file__).parents[4]
PATH = ROOT / "python/sglang/srt/hardware_backend/npu"


def load(name, monkeypatch):
    qualified = npu_module_name(name)
    spec = importlib.util.spec_from_file_location(
        qualified, npu_source_path(PATH / f"{name.rsplit('.', 1)[-1]}.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, qualified, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    for package in (
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.hardware_backend.npu.hisparse",
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT / "python" / package.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, package, module)
    protocol = load(
        "sglang.srt.hardware_backend.npu.hisparse.prefill_offload", monkeypatch
    )
    registered = load("sglang.srt.hardware_backend.npu.registered_host_kv", monkeypatch)
    adapter = load(
        "sglang.srt.hardware_backend.npu.hisparse.prefill_offload_adapter", monkeypatch
    )
    return SimpleNamespace(protocol=protocol, adapter=adapter, slot_pool=registered)


def test_adapter_builds_component_descriptors_without_cpu_pointers(modules):
    allocation = modules.protocol.HostPageAllocation("req", (2, 3), 128, 1)
    calls = []

    class FakeHostPool:
        def allocate_component(self, *args, **kwargs):
            calls.append(("allocate", args, kwargs))

        def copy_to_host(self, *args):
            calls.append(("save", args))

        def copy_from_host(self, *args):
            calls.append(("restore", args))

    adapter = modules.adapter.NpuPrefillOffloadAdapter(FakeHostPool(), page_size=128)
    adapter.register_layer_components(
        2,
        latent_shape=(512, 512),
        rope_shape=(512, 64),
        indexer_shape=(512, 128),
        dtype=torch.float32,
    )
    work = SimpleNamespace(
        token=modules.protocol.LayerSlotToken(1, "req", 2, 7),
        host_allocation=allocation,
        valid_range=modules.protocol.KvPageRange(0, 128),
    )
    adapter.bind_layer(
        work,
        torch.empty((2, 128, 512)),
        torch.empty((2, 128, 64)),
        torch.empty((2, 128, 128)),
        source_page_ids=(0, 1),
        destination_pages={
            component: (5, 6) for component in modules.protocol.KvComponent
        },
    )
    records = adapter.descriptors(work)
    assert {record.component for record in records} == {
        modules.protocol.KvComponent.LATENT_K,
        modules.protocol.KvComponent.ROPE_K,
        modules.protocol.KvComponent.INDEXER_K,
    }
    assert all(
        record.source_kind == modules.protocol.PdSourceKind.NPU_REGISTERED
        for record in records
    )
    assert len(calls) == 3
    assert all(record.source_page_ids == (0, 1) for record in records)
    assert all(record.destination_page_ids == (5, 6) for record in records)
    assert records[0].bytes_per_page == 128 * 512 * 4


def test_adapter_uses_bound_token_indices_and_valid_range(modules):
    allocation = modules.protocol.HostPageAllocation("req", (2,), 128, 1)
    calls = []

    class FakeHostPool:
        def allocate_component(self, *args, **kwargs):
            pass

        def copy_to_host(self, *args):
            calls.append(args)

        def copy_from_host(self, *args):
            pass

    adapter = modules.adapter.NpuPrefillOffloadAdapter(
        FakeHostPool(),
        page_size=128,
        record_event=lambda: SimpleNamespace(query=lambda: True),
    )
    adapter.register_layer_components(
        0, latent_shape=(384, 8), rope_shape=(384, 4), dtype=torch.float32
    )
    work = SimpleNamespace(
        token=modules.protocol.LayerSlotToken(0, "req", 0, 1),
        host_allocation=allocation,
        valid_range=modules.protocol.KvPageRange(2, 4),
    )
    token_indices = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    adapter.bind_layer(
        work,
        torch.empty((4, 8)),
        torch.empty((4, 4)),
        token_indices=token_indices,
    )
    adapter.save_layer(work)
    assert len(calls) == 2
    for call in calls:
        assert call[2].tolist() == [0, 1, 2, 3]
        assert call[2].dtype == torch.int64
        assert call[3].tolist() == [257, 258, 259, 260]
        assert call[4].tolist() == [False, True, True, False]


def test_adapter_binds_components_through_bounded_slot_pool(modules):
    allocation = modules.protocol.HostPageAllocation("req", (0,), 4, 1)
    slot_pool_module = modules.slot_pool
    slot_pool = slot_pool_module.NpuLayerwiseKvSlotPool(1, device="cpu")
    slot_pool.allocate_component("latent_k", (2, 4), torch.float32)
    slot_pool.allocate_component("rope_k", (2, 2), torch.float32)
    adapter = modules.adapter.NpuPrefillOffloadAdapter(
        SimpleNamespace(allocate_component=lambda *args, **kwargs: None),
        page_size=4,
        slot_pool=slot_pool,
    )
    adapter.register_layer_components(
        0, latent_shape=(4, 4), rope_shape=(4, 2), dtype=torch.float32
    )
    work = SimpleNamespace(
        token=modules.protocol.LayerSlotToken(0, "req", 0, 1),
        host_allocation=allocation,
        valid_range=modules.protocol.KvPageRange(0, 2),
    )
    latent = torch.ones((2, 4), dtype=torch.float32)
    rope = torch.ones((2, 2), dtype=torch.float32)
    binding = adapter.bind_layer(work, latent, rope)
    assert binding.latent_k.data_ptr() == slot_pool.slot_view("latent_k", 0).data_ptr()
    assert torch.equal(binding.latent_k, latent)
    adapter.release_layer(work)
    slot_pool.reserve_slot(0, object())


class CopyingHostPool:
    def __init__(self):
        self.storage = {}

    def allocate_component(self, name, shape, dtype, **kwargs):
        self.storage[name] = torch.full(shape, -999.0, dtype=dtype)

    def copy_to_host(self, name, src, src_rows, dst_rows, valid):
        assert src_rows.dtype == dst_rows.dtype == torch.int64
        self.storage[name][dst_rows[valid]] = src[src_rows[valid]]

    def copy_from_host(self, name, dst, src_rows, dst_rows, valid):
        assert src_rows.dtype == dst_rows.dtype == torch.int64
        dst[dst_rows[valid]] = self.storage[name][src_rows[valid]]


def make_work(modules, allocation, layer, generation, start, end):
    return SimpleNamespace(
        token=modules.protocol.LayerSlotToken(
            0, allocation.request_id, layer, generation
        ),
        host_allocation=allocation,
        valid_range=modules.protocol.KvPageRange(start, end),
    )


def test_chunk_restore_is_layer_local_and_only_reads_completed_prefix(modules):
    host = CopyingHostPool()
    event = SimpleNamespace(query=lambda: True)
    adapter = modules.adapter.NpuPrefillOffloadAdapter(
        host, page_size=2, record_event=lambda: event
    )
    allocation = modules.protocol.HostPageAllocation("req", (2, 0), 2, 1)
    for layer in range(2):
        adapter.register_layer_components(
            layer,
            latent_shape=(6, 3),
            rope_shape=(6, 2),
            indexer_shape=(6, 4),
            dtype=torch.float32,
        )
        work = make_work(modules, allocation, layer, layer + 1, 0, 3)
        values = [torch.full((3, width), float(layer + 10)) for width in (3, 2, 4)]
        adapter.bind_layer(work, *values)
        adapter.restore_layer(work)
        adapter.save_layer(work)
        adapter.complete_host_save(work)
        adapter.release_layer(work)

    for layer in range(2):
        work = make_work(modules, allocation, layer, layer + 3, 3, 4)
        values = [torch.full((4, width), -1.0) for width in (3, 2, 4)]
        adapter.bind_layer(work, *values)
        adapter.restore_layer(work)
        for tensor in values:
            assert torch.all(tensor[:3] == layer + 10)
            assert torch.all(tensor[3] == -1)
            tensor[3].fill_(layer + 20)
        adapter.save_layer(work)
        adapter.complete_host_save(work)
        adapter.release_layer(work)
        for component in modules.protocol.KvComponent:
            stored = host.storage[f"layer:{layer}:{component.value}"]
            assert torch.all(stored[[4, 5, 0]] == layer + 10)
            assert torch.all(stored[1] == layer + 20)
            assert torch.all(stored[2:4] == -999)


def test_paged_device_rows_are_independent_of_logical_and_host_rows(modules):
    host = CopyingHostPool()
    adapter = modules.adapter.NpuPrefillOffloadAdapter(
        host, page_size=2, record_event=lambda: SimpleNamespace(query=lambda: True)
    )
    adapter.register_layer_components(
        0, latent_shape=(8, 3), rope_shape=(8, 2), dtype=torch.float32
    )
    allocation = modules.protocol.HostPageAllocation("req", (3, 1), 2, 7)
    work = make_work(modules, allocation, 0, 1, 0, 4)
    latent = torch.arange(24, dtype=torch.float32).view(4, 2, 1, 3)
    rope = torch.arange(16, dtype=torch.float32).view(4, 2, 1, 2)
    device_rows = torch.tensor([4, 5, 0, 1], dtype=torch.int64)
    adapter.bind_layer(
        work,
        latent,
        rope,
        token_indices=torch.arange(4),
        device_row_indices=device_rows,
    )
    adapter.save_layer(work)
    assert torch.equal(
        host.storage["layer:0:latent_k"][[6, 7, 2, 3]],
        latent.view(-1, 3)[device_rows],
    )


def test_host_save_event_and_attention_event_are_independent(modules):
    host = CopyingHostPool()
    save_event = SimpleNamespace(query=lambda: False)
    attention_event = SimpleNamespace(query=lambda: False)
    events = iter([save_event, attention_event])
    adapter = modules.adapter.NpuPrefillOffloadAdapter(
        host, page_size=2, record_event=lambda: next(events)
    )
    adapter.register_layer_components(
        0, latent_shape=(4, 3), rope_shape=(4, 2), dtype=torch.float32
    )
    work = make_work(
        modules, modules.protocol.HostPageAllocation("req", (0,), 2, 1), 0, 1, 0, 2
    )
    adapter.bind_layer(work, torch.ones(2, 3), torch.ones(2, 2))
    adapter.save_layer(work)
    adapter.record_attention_done(work)
    with pytest.raises(modules.protocol.SlotBusyError, match="DMA"):
        adapter.complete_host_save(work)
    save_event.query = lambda: True
    adapter.complete_host_save(work)
    with pytest.raises(modules.protocol.SlotBusyError, match="attention"):
        adapter.complete_attention(work)
    with pytest.raises(modules.protocol.SlotBusyError):
        adapter.release_layer(work)
    attention_event.query = lambda: True
    adapter.complete_attention(work)
    adapter.release_layer(work)


def test_request_reuse_cannot_restore_previous_lifetime(modules):
    host = CopyingHostPool()
    adapter = modules.adapter.NpuPrefillOffloadAdapter(
        host, page_size=2, record_event=lambda: SimpleNamespace(query=lambda: True)
    )
    adapter.register_layer_components(
        0, latent_shape=(2, 3), rope_shape=(2, 2), dtype=torch.float32
    )
    old = modules.protocol.HostPageAllocation("req", (0,), 2, 1)
    work = make_work(modules, old, 0, 1, 0, 1)
    adapter.bind_layer(work, torch.ones(1, 3), torch.ones(1, 2))
    adapter.save_layer(work)
    adapter.complete_host_save(work)
    adapter.release_layer(work)
    adapter.release_request(old)
    new = modules.protocol.HostPageAllocation("req", (0,), 2, 2)
    work = make_work(modules, new, 0, 2, 1, 2)
    adapter.bind_layer(work, torch.ones(2, 3), torch.ones(2, 2))
    with pytest.raises(RuntimeError, match="historical KV"):
        adapter.restore_layer(work)


def test_fixture_restores_framework_modules_after_use():
    names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.hardware_backend.npu.hisparse",
        "sglang.srt.hardware_backend.npu.hisparse.prefill_offload",
        "sglang.srt.hardware_backend.npu.registered_host_kv",
        "sglang.srt.hardware_backend.npu.hisparse.prefill_offload_adapter",
    )
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    with pytest.MonkeyPatch.context() as monkeypatch:
        loaded = modules.__wrapped__(monkeypatch)
        assert loaded.adapter.KvComponent is loaded.protocol.KvComponent
        assert (
            loaded.adapter.NpuLayerwiseKvSlotPool
            is loaded.slot_pool.NpuLayerwiseKvSlotPool
        )
    for name, module in previous.items():
        assert sys.modules.get(name, missing) is module


def test_import_preserves_framework_packages_and_python_path(modules):
    packages = {
        name: module
        for name, module in sys.modules.items()
        if name == "sglang" or name.startswith("sglang.")
    }
    package_paths = {
        name: tuple(module.__path__)
        for name, module in packages.items()
        if hasattr(module, "__path__")
    }
    python_path = list(sys.path)
    spec = importlib.util.spec_from_file_location(
        "prefill_adapter_import_check", __file__
    )
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    assert sys.path == python_path
    for name, module in packages.items():
        assert sys.modules[name] is module
        if name in package_paths:
            assert tuple(module.__path__) == package_paths[name]
    assert not hasattr(imported, "protocol")
    assert not hasattr(imported, "adapter_module")


def test_fixture_packages_resolve_host_descriptors_without_package_initialization(
    modules,
):
    package = sys.modules["sglang.srt"]
    assert package.__path__ == [str(ROOT / "python/sglang/srt")]
    spec = PathFinder.find_spec("sglang.srt.mem_cache", package.__path__)
    assert spec is not None
    assert str(ROOT / "python/sglang/srt/mem_cache") in spec.submodule_search_locations
    assert (Path(spec.submodule_search_locations[0]) / "device_pool_info.py").is_file()


if __name__ == "__main__":
    from sglang.test.npu_test_paths import run_pytest

    run_pytest(__file__)
