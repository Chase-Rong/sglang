"""Production NPU pool methods and batched layer lifetime with CPU copy oracles."""

import ast
import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[4]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu"


@pytest.fixture
def modules(monkeypatch):
    prefix = "sglang.srt.hardware_backend.npu"
    for name in ("sglang", "sglang.srt", "sglang.srt.hardware_backend", prefix):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    modules = {}
    for name in (
        "prefill_offload",
        "registered_host_kv",
        "prefill_offload_adapter",
        "prefill_offload_runtime",
    ):
        qualified = f"{prefix}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, NPU / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def production_pool_type():
    class BasePool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.start_layer = kwargs.get("start_layer") or 0
            self.store_dtype = self.dtype
            self.layer_transfer_counter = None
            self.memory_saver_adapter = SimpleNamespace(
                region=lambda *args: contextlib.nullcontext()
            )

        def _finalize_allocation_log(self, size):
            self.mem_usage = self.get_kv_size_bytes()

    class MLAPool(BasePool):
        pass

    source = ast.parse((NPU / "memory_pool_npu.py").read_text(encoding="utf-8"))
    node = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUMLATokenToKVPool"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    namespace = dict(
        torch=torch,
        MLATokenToKVPool=MLAPool,
        GPU_MEMORY_TYPE_KV_CACHE="kv",
        get_tensor_size_bytes=lambda tensor: tensor.nbytes,
        unwrap_write_loc=lambda value: (value, None, None),
        torch_npu=SimpleNamespace(
            npu_scatter_nd_update_=lambda dst, indices, src: dst.index_copy_(
                0, indices.flatten().long(), src
            )
        ),
    )
    exec(
        compile(ast.fix_missing_locations(module), "production_npu_pool", "exec"),
        namespace,
    )
    return namespace["NPUMLATokenToKVPool"]


class HostCopies:
    def __init__(self):
        self.storage = {}

    def allocate_component(self, name, shape, dtype, **kwargs):
        self.storage[name] = torch.full(shape, -999.0, dtype=dtype)

    def copy_to_host(self, name, source, source_rows, target_rows, valid):
        self.storage[name][target_rows[valid]] = source[source_rows[valid]]

    def copy_from_host(self, name, target, source_rows, target_rows, valid):
        target[target_rows[valid]] = self.storage[name][source_rows[valid]]


def make_runtime(modules, *, layers=4, physical_slots=2, publish=None, start_layer=0):
    pool = production_pool_type()(
        size=16,
        page_size=2,
        dtype=torch.float32,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        index_head_dim=3,
        layer_num=layers,
        device="cpu",
        enable_memory_saver=False,
        prefill_layer_slots=physical_slots,
        start_layer=start_layer,
        end_layer=start_layer + layers,
    )
    host = HostCopies()
    adapter = modules["prefill_offload_adapter"].NpuPrefillOffloadAdapter(
        host, page_size=2, record_event=lambda: SimpleNamespace(query=lambda: True)
    )
    request_pool = SimpleNamespace(
        req_to_token=torch.tensor(
            [
                [0, 0, 0, 0],
                [2, 3, 8, 9],
                [4, 5, 10, 11],
            ]
        )
    )
    runtime = modules["prefill_offload_runtime"].NpuPrefillOffloadRuntime(
        pool,
        request_pool,
        adapter,
        publish=publish or (lambda *args: [SimpleNamespace(state="acknowledged")]),
        timeout_s=0.001,
    )
    runtime.register_host_layers()
    pool.prefill_offload_runtime = runtime
    return runtime, pool, host


def make_batch(length, prefix, request_count=2):
    sender = SimpleNamespace(
        require_prefill_layer_targets=lambda layers: None,
        prefill_layer_operations=lambda: (),
    )
    return SimpleNamespace(
        rids=[f"req-{index}" for index in range(request_count)],
        seq_lens_cpu=torch.full((request_count,), length),
        extend_prefix_lens_cpu=[prefix] * request_count,
        req_pool_indices=torch.arange(1, request_count + 1),
        prefill_offload_senders=[sender] * request_count,
        forward_mode=SimpleNamespace(
            is_extend=lambda: True,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        ),
    )


def test_actual_pool_allocation_is_bounded_and_registration_preserves_logical_layers(
    modules,
):
    runtime, pool, host = make_runtime(modules)
    assert pool.k_buffer.shape[0] == 2
    assert pool.layer_num == 4
    assert pool.get_kv_size_bytes() == 2 * 18 * (4 + 2 + 3) * 4
    pointers, lengths, page_bytes = pool.get_contiguous_buf_infos()
    assert len(pointers) == 12
    assert pointers[0] == pointers[2] and pointers[1] == pointers[3]
    assert len(set(pointers)) == 6
    with pytest.raises(RuntimeError, match="active"):
        pool.get_index_k_buffer(0)


@pytest.mark.parametrize("start_layer", [0, 8])
def test_two_requests_four_layers_two_chunks_restore_before_indexer_access(
    modules, start_layer
):
    runtime, pool, host = make_runtime(modules, start_layer=start_layer)
    for prefix, length in ((0, 2), (2, 3)):
        batch = make_batch(length, prefix)
        with runtime.forward(batch):
            for layer_id in range(start_layer, start_layer + 4):
                indexer = pool.get_index_k_buffer(layer_id).view(-1, 3)
                latent, rope = pool.get_kv_buffer(layer_id)
                views = (latent.view(-1, 4), rope.view(-1, 2), indexer)
                for request_index, request in enumerate(runtime.requests):
                    for component_index, view in enumerate(views):
                        value = 10 * layer_id + 100 * component_index + request_index
                        rows = request.device_rows
                        if prefix:
                            assert torch.all(view[rows[:prefix]] == value)
                        view[rows[prefix:]] = value
                runtime.finish_layer(layer_id)
        assert runtime.slots.active_tokens() == ()
    for layer_id in range(start_layer, start_layer + 4):
        for request_index in range(2):
            allocation = runtime.host_pages.allocation(f"req-{request_index}")
            rows = [
                allocation.page_ids[index // 2] * 2 + index % 2 for index in range(3)
            ]
            for component_index, name in enumerate(("latent_k", "rope_k", "indexer_k")):
                assert torch.all(
                    host.storage[f"layer:{layer_id}:{name}"][rows]
                    == 10 * layer_id + 100 * component_index + request_index
                )
    runtime.release_request("req-0")
    runtime.release_request("req-1")
    assert runtime.host_pages.free_page_count == runtime.host_pages.page_count


def test_receiver_ack_timeout_retains_the_whole_batch_slot(modules):
    operation = SimpleNamespace(state="transferred")
    runtime, pool, host = make_runtime(
        modules, layers=1, physical_slots=1, publish=lambda *args: [operation]
    )
    with pytest.raises(TimeoutError, match="ACK"):
        with runtime.forward(make_batch(2, 0)):
            pool.get_index_k_buffer(0)
            runtime.finish_layer(0)
    assert runtime.failed
    assert len(runtime.layers[0].works) == 2
    assert runtime.slots.owner(0) is not None
    with pytest.raises(modules["prefill_offload"].SlotBusyError):
        runtime.release_request("req-0")
    with pytest.raises(modules["prefill_offload"].SlotBusyError, match="transfer"):
        runtime.drain_failed_batch()
    operation.state = "drained"
    runtime.drain_failed_batch()
    assert runtime.slots.active_tokens() == ()
    runtime.release_request("req-0")
    runtime.release_request("req-1")


def test_layer_access_after_attention_completion_is_rejected(modules):
    runtime, pool, host = make_runtime(modules, layers=1, physical_slots=1)
    runtime.begin_batch(make_batch(2, 0))
    pool.get_kv_buffer(0)
    runtime.finish_layer(0)
    with pytest.raises(RuntimeError, match="after attention"):
        pool.get_index_k_buffer(0)
    runtime.finish_batch()


def test_prefill_memory_budget_counts_physical_device_layers_and_all_host_layers(
    modules,
):
    budget = modules["prefill_offload"].PrefillKvMemoryBudget(
        logical_layers=64,
        physical_slots=2,
        bytes_per_token_per_layer=1408,
        host_bytes=64 * 1408 * 1024,
    )
    assert budget.device_bytes_per_token == 2816
    assert budget.host_bytes_per_token == 90112
    assert budget.max_tokens(device_bytes=2816 * 4096, page_size=128) == 896
    assert budget.max_tokens(device_bytes=2816 * 512, page_size=128) == 384
    budget.validate_host_capacity(896, 128)
    with pytest.raises(MemoryError, match="Host KV"):
        budget.validate_host_capacity(1024, 128)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_replicated_prefill_cp_layout(modules, cp_size):
    validate = modules["prefill_offload_runtime"].validate_prefill_parallel_layout
    validate(
        cp_size=cp_size,
        dcp_size=1,
        replicated_cp=cp_size > 1,
        layer_split=False,
        all_cp_transfer=False,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"dcp_size": 2},
        {"layer_split": True},
        {"replicated_cp": False},
        {"all_cp_transfer": True},
    ],
)
def test_prefill_cp_rejects_unmatched_source_layout(modules, override):
    values = dict(
        cp_size=2,
        dcp_size=1,
        replicated_cp=True,
        layer_split=False,
        all_cp_transfer=False,
    )
    values.update(override)
    with pytest.raises(ValueError):
        modules["prefill_offload_runtime"].validate_prefill_parallel_layout(**values)


def test_draft_runtime_negotiates_wire_layers_without_changing_local_host_storage(
    modules,
):
    runtime, pool, host = make_runtime(modules, layers=1, physical_slots=1)
    pool.fsa_transfer_layer_offset = 64
    requested = []
    batch = make_batch(2, 0)
    for sender in batch.prefill_offload_senders:
        sender.require_prefill_layer_targets = lambda layers: requested.append(
            tuple(layers)
        )
    with runtime.forward(batch):
        pool.get_kv_buffer(0)
        runtime.finish_layer(0)
    assert requested == [(64,), (64,)]
    assert set(host.storage) == {
        "layer:0:latent_k",
        "layer:0:rope_k",
        "layer:0:indexer_k",
    }


def test_completed_request_can_release_while_an_unrelated_batch_is_active(modules):
    runtime, pool, _ = make_runtime(modules, layers=1, physical_slots=1)
    first = make_batch(2, 0, request_count=1)
    with runtime.forward(first):
        pool.get_kv_buffer(0)
        runtime.finish_layer(0)
    second = make_batch(2, 0, request_count=1)
    second.rids = ["another-request"]
    second.req_pool_indices = torch.tensor([2])
    with runtime.forward(second):
        assert runtime.can_release_request("req-0")
        assert not runtime.can_release_request("another-request")
        runtime.release_request("req-0")
        runtime.release_request("req-0")
        pool.get_kv_buffer(0)
        runtime.finish_layer(0)


def test_release_poll_waits_for_target_draft_and_transport_drain(modules, monkeypatch):
    connection = types.ModuleType("sglang.srt.disaggregation.base.conn")
    connection.KVPoll = SimpleNamespace(WaitingForInput=1, Success=2)
    monkeypatch.setitem(sys.modules, connection.__name__, connection)
    ready = [True, False]
    runtimes = [
        SimpleNamespace(can_release_request=lambda request, index=index: ready[index])
        for index in range(2)
    ]
    operation = SimpleNamespace(state="transferred")
    sender = SimpleNamespace(prefill_layer_operations=lambda: (operation,))
    check = modules["prefill_offload_runtime"].PrefillReleaseCheck(
        runtimes, "req", sender
    )
    assert check.poll() == 1
    ready[1] = True
    assert check.poll() == 1
    operation.state = "failed"
    assert check.poll() == 1
    operation.state = "drained"
    assert check.poll() == 2
