"""Dependency and data checks for the two-slot Prefill pipeline without NPU hardware."""

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu"


class Stream:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.tail = frozenset()

    def mark(self, name):
        self.tail = self.tail | {name}

    def wait_event(self, event):
        assert event.recorded
        self.tail = self.tail | event.dependencies

    def synchronize(self):
        self.model.synchronized.update(self.tail)


class Event:
    def __init__(self, model):
        self.model = model
        self.recorded = False

    def record(self, stream):
        self.recorded = True
        self.dependencies = stream.tail

    def query(self):
        return self.dependencies <= self.model.synchronized

    def synchronize(self):
        self.model.synchronized.update(self.dependencies)


class Device:
    def __init__(self):
        self.synchronized = set()
        self.capturing = False
        self.current = Stream(self, "compute")
        self.compute = self.current
        self.created = []

    def Stream(self, device):
        stream = Stream(self, f"copy{len(self.created)}")
        self.created.append(stream)
        return stream

    def Event(self):
        return Event(self)

    def current_stream(self):
        return self.current

    def is_current_stream_capturing(self):
        return self.capturing

    @contextlib.contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous


@pytest.fixture
def setup(monkeypatch):
    device = Device()
    monkeypatch.setattr(torch, "npu", device, raising=False)
    prefix = "sglang.srt.hardware_backend.npu"
    for name in ("sglang", "sglang.srt", "sglang.srt.hardware_backend", prefix):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    loaded = {}
    for name in ("integrated_offload", "prefill_overlap"):
        qualified = prefix + "." + name
        spec = importlib.util.spec_from_file_location(qualified, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        loaded[name] = module

    def build(h2d=True, d2h=True, slots=2):
        host = torch.zeros(4, 24, 2)
        storage = torch.zeros(slots, 24, 2)
        copies = []
        publications = []

        def restore(name, target, src, dst, valid):
            stream = device.current
            copies.append(
                ("restore", name, runtime.generation, stream.name, stream.tail)
            )
            target[dst[valid]] = host[name, src[valid]]
            stream.mark(f"restore-{runtime.generation}-{name}")

        def save(layer, components, rows, valid):
            stream = device.current
            assert f"scatter-{runtime.generation}-{layer}" in stream.tail
            copies.append(("save", layer, runtime.generation, stream.name, stream.tail))
            host[layer, rows[valid]] = components[0].view(-1, 2)[rows[valid]]
            stream.mark(f"save-{runtime.generation}-{layer}")

        def publish():
            publications.append(device.current.tail)

        backing = SimpleNamespace(
            pages=12,
            page_size=2,
            _component_names=[(layer, None) for layer in range(4)],
            pool=SimpleNamespace(copy_from_host=restore),
            enqueue_prefill_rows_from_slot=save,
            publish_prefill_writes=publish,
        )

        def physical(layer):
            if not 7 <= layer < 11:
                raise IndexError(layer)
            return (layer - 7) % slots

        pool = SimpleNamespace(
            integrated_offload=True,
            decode_host_backing=backing,
            prefill_layer_slots=slots,
            start_layer=7,
            layer_num=4,
            kv_lora_rank=2,
            qk_rope_head_dim=0,
            k_buffer=storage,
            physical_layer_index=physical,
            raw_layer_components=lambda layer: (
                storage[physical(layer)],
                torch.empty(0),
            ),
        )
        request_pool = SimpleNamespace(
            req_to_token=torch.tensor([[8, 3, 10, 5, 12, 7]])
        )
        runtime = loaded["prefill_overlap"].PipelinedIntegratedOffloadRuntime(
            pool, request_pool, h2d_prefetch=h2d, d2h_async=d2h
        )
        return SimpleNamespace(
            runtime=runtime,
            pool=pool,
            host=host,
            copies=copies,
            publications=publications,
            device=device,
            mapping=request_pool.req_to_token[0],
        )

    return build


@pytest.mark.parametrize(
    "h2d,d2h", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("slots", [1, 2])
def test_prefix_and_slot_reuse_dependencies(setup, h2d, d2h, slots):
    case = setup(h2d, d2h, slots)
    runtime = case.runtime
    for prefix, length in ((0, 2), (2, 4), (4, 6)):
        batch = SimpleNamespace(
            extend_prefix_lens_cpu=[prefix],
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=case.mapping[prefix:length],
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        with runtime.forward(batch):
            for layer in range(7, 11):
                runtime.prepare_layer(layer)
                runtime.prepare_layer(layer)
                current = case.pool.raw_layer_components(layer)[0]
                torch.testing.assert_close(
                    current[case.mapping[:prefix]],
                    case.host[layer - 7, case.mapping[:prefix]],
                )
                current[case.mapping[prefix:length]] = layer * 10 + prefix
                case.device.current.mark(f"scatter-{runtime.generation}-{layer - 7}")
                if d2h:
                    runtime.on_kv_written(layer, case.mapping[prefix:length])
                else:
                    case.host[layer - 7, case.mapping[prefix:length]] = current[
                        case.mapping[prefix:length]
                    ]
                runtime.on_compute_start(layer)
                case.device.current.mark(f"attention-{runtime.generation}-{layer - 7}")
                runtime.finish_layer(layer)
        if d2h:
            assert all(
                f"save-{runtime.generation}-{layer}" in case.publications[-1]
                for layer in range(4)
            )
    for operation, layer, generation, stream, dependencies in case.copies:
        if operation == "restore" and layer >= slots:
            assert f"attention-{generation}-{layer - slots}" in dependencies
            if d2h:
                assert f"save-{generation}-{layer - slots}" in dependencies
        if operation == "restore" and generation > 1:
            assert f"attention-{generation - 1}-3" in dependencies
            if d2h:
                assert all(
                    f"save-{generation - 1}-{previous}" in dependencies
                    for previous in range(4)
                )
    runtime.drain()
    assert not runtime.retired and not runtime.active and not runtime.failed


def test_prefetch_does_not_wait_for_current_attention_completion(setup):
    case = setup()
    runtime = case.runtime
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[2],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[2:4],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="stop test"):
        with runtime.forward(batch):
            runtime.prepare_layer(7)
            case.device.current.mark("scatter-1-0")
            runtime.on_kv_written(7, case.mapping[2:4])
            runtime.on_compute_start(7)
            restore_b = next(item for item in case.copies if item[:2] == ("restore", 1))
            assert "attention-1-0" not in restore_b[4]
            assert "save-1-0" not in restore_b[4]
            with pytest.raises(RuntimeError, match="Duplicate"):
                runtime.validate_write(7)
            raise RuntimeError("stop test")
    assert runtime.failed and not runtime.active
    with pytest.raises(RuntimeError, match="failed"):
        with runtime.forward(batch):
            pass


def test_missing_predecessor_hook_rejected_before_future_wait(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="not prefetched"):
        with case.runtime.forward(batch):
            case.runtime.prepare_layer(8)


def test_empty_write_generation_and_request_release(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:0],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with case.runtime.forward(batch):
        with pytest.raises(RuntimeError, match="active"):
            case.runtime.release_request("request")
        for layer in range(7, 11):
            case.runtime.prepare_layer(layer)
            case.device.current.mark(f"scatter-1-{layer - 7}")
            case.runtime.on_compute_start(layer)
            case.runtime.finish_layer(layer)
    case.runtime.release_request("request")
    assert not case.runtime.retired and case.runtime.host_visible is None


def test_nonempty_generation_requires_real_scatter(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="KV-written"):
        with case.runtime.forward(batch):
            case.runtime.prepare_layer(7)
            case.runtime.on_compute_start(7)


def test_prefill_capture_rejected_before_copy_submission(setup):
    case = setup()
    case.device.capturing = True
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
    )
    with pytest.raises(RuntimeError, match="graph capture"):
        with case.runtime.forward(batch):
            pass
    assert not case.copies and not case.runtime.active


def test_decode_on_another_stream_waits_for_host_visibility(setup):
    case = setup()
    writer = case.device.current
    writer.mark("last-generation-published")
    case.runtime.host_visible = case.runtime._record(writer)
    decoder = case.device.Stream("cpu")
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False)
    )
    with case.device.stream(decoder), case.runtime.forward(batch):
        assert "last-generation-published" in case.device.current.tail
    assert not case.runtime.active


def test_layer_hook_cannot_change_compute_stream(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="forward compute stream"):
        with case.runtime.forward(batch):
            with case.device.stream(case.device.Stream("cpu")):
                case.runtime.prepare_layer(7)
    assert case.runtime.failed and not case.runtime.active
