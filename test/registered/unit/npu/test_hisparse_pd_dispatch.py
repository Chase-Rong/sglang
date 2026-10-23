"""Exercise production PD dispatch on CPU without accelerator imports."""

import ast
import dataclasses
import importlib.util
import logging
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import test_layer_receive_group as receive_tests
import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
PD_NAME = "sglang.srt.hardware_backend.npu.hisparse.pd"
receive_inputs = receive_tests.inputs


def production_definitions(path, classes, functions=(), **namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            node.bases = []
            if classes[node.name] is not None:
                node.body = [
                    method
                    for method in node.body
                    if isinstance(method, ast.FunctionDef)
                    and method.name in classes[node.name]
                ]
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
    namespace.update(__name__=__name__)
    definitions = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(definitions, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def native_dispatch(native):
    module = production_definitions(
        SRT / "managers/hisparse_factory.py",
        {},
        functions={"uses_native_hisparse"},
        get_memory=lambda: SimpleNamespace(enable_hisparse=True),
        get_platform=lambda: SimpleNamespace(is_npu=not native),
    )
    return module.uses_native_hisparse


def decode_definitions(native=False):
    return production_definitions(
        SRT / "disaggregation/decode.py",
        {
            "DecodePreallocQueue": {
                "_init_kv_manager",
                "_pre_alloc",
                "_pre_alloc_fill_len",
                "_rebootstrap_prefill_len",
                "_allocatable_token_budgets",
                "_check_if_req_exceed_kv_capacity",
                "pop_preallocated",
            },
            "DecodeTransferQueue": {"pop_transferred"},
            "SchedulerDisaggregationDecodeMixin": {"process_decode_queue"},
        },
        functions={"alloc_for_decode_prealloc", "alloc_for_decode_prealloc_hisparse"},
        uses_native_hisparse=native_dispatch(native),
        torch=torch,
        np=np,
        time=time,
        HTTPStatus=HTTPStatus,
        logger=logging.getLogger(__name__),
        FINISH_ABORT=type("FinishAbort", (), {}),
        DeepSeekV4TokenToKVPool=type("DeepSeekV4Pool", (), {}),
        TransferBackend=SimpleNamespace(NIXL="nixl", MOONCAKE="mooncake"),
        DisaggregationMode=SimpleNamespace(DECODE="decode"),
        KVClassType=SimpleNamespace(KVARGS="args", MANAGER="manager"),
        CLIP_MAX_NEW_TOKEN=32,
        _is_npu=True,
    )


@pytest.fixture
def pd(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        PD_NAME, SRT / "hardware_backend/npu/hisparse/pd.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, PD_NAME, module)
    spec.loader.exec_module(module)
    return module


def test_pd_reuses_pool_and_target_draft_lease_namespaces(pd, receive_inputs):
    _, pool, events = receive_inputs
    target, draft = pool(0, 4, "target"), pool(16, 1, "draft")
    assert pd.get_pd_receive_pool(SimpleNamespace(), draft_layer_base=4) is None
    assert pd.get_pd_receive_pool(target, draft_layer_base=4) is target
    group = pd.get_pd_receive_pool(target, draft, draft_layer_base=4)
    assert group.host_backings == (
        target.decode_host_backing,
        draft.decode_host_backing,
    )
    lease = group.layer_receive_lease(5, "session", (1, 3), (0, 2), prefix=1)
    assert set(lease.layers) == {0, 1, 2, 3, 4}
    assert lease.prefix == 1
    lease.visibility(4, 1, 3)
    assert events == [("draft", 16, 1, 3)]
    lease.close()


@pytest.mark.parametrize("native", [False, True])
def test_manager_keeps_target_draft_and_state_registration(native, pd, receive_inputs):
    _, pool, _ = receive_inputs
    module = decode_definitions(native)
    namespace = module.DecodePreallocQueue._init_kv_manager.__globals__
    target, draft = pool(0, 4, "target"), pool(16, 1, "draft")
    target.get_contiguous_buf_infos = lambda: ([100], [64], [16])
    draft.get_contiguous_buf_infos = lambda: ([200], [32], [16])
    host = SimpleNamespace(get_contiguous_buf_infos=lambda: ([900], [128], [16]))
    args = SimpleNamespace()
    manager = Mock()
    manager_class = Mock(return_value=manager)
    state_setup = Mock()
    namespace.update(
        get_kv_class=lambda backend, kind: (lambda: args)
        if kind == "args"
        else manager_class,
        get_parallel=lambda: SimpleNamespace(attn_tp_size=2),
        get_disagg=lambda: SimpleNamespace(disaggregation_ib_device="ib0"),
        build_kv_layer_ids=Mock(return_value=[0, 4]),
        setup_state_kv_args=state_setup,
    )
    queue = module.DecodePreallocQueue()
    queue.token_to_kv_pool, queue.draft_token_to_kv_pool = target, draft
    queue.token_to_kv_pool.page_size = 2
    queue.scheduler = SimpleNamespace(
        enable_hisparse=True,
        hisparse_coordinator=SimpleNamespace(mem_pool_host=host),
        ps=SimpleNamespace(dp_rank=0, gpu_id=0),
        model_config=SimpleNamespace(num_hidden_layers=4),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(kv_cache_dtype_str="bf16")
        ),
        server_args=SimpleNamespace(),
    )
    queue.tp_rank, queue.pp_rank = 0, 0
    queue.enable_staging, queue.is_mla_backend = False, True
    queue.transfer_backend = "nixl"
    queue.req_to_token_pool = object()
    queue.metadata_buffers = SimpleNamespace(get_buf_infos=lambda: ([300], [16], [16]))
    assert queue._init_kv_manager() is manager
    assert args.kv_data_ptrs == ([900, 200] if native else [100, 200])
    assert args.kv_data_mem_kinds == (["DRAM", "VRAM"] if native else ["VRAM", "VRAM"])
    assert args.kv_layer_ids == [0, 4]
    assert len(args.npu_layer_receive_pool.host_backings) == 2
    state_setup.assert_called_once_with(
        args,
        target,
        draft,
        total_kv_layers=4,
        req_to_token_pool=queue.req_to_token_pool,
    )


@pytest.mark.parametrize("configured", [0, 4096])
@pytest.mark.parametrize("host_offload", [False, True])
def test_decode_manager_creates_layer_staging_with_actual_service_enum(
    monkeypatch, configured, host_offload
):
    service_source = SRT / "disaggregation/utils.py"
    service_tree = ast.parse(service_source.read_text(encoding="utf-8"))
    service_class = next(
        node
        for node in service_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DisaggregationMode"
    )
    service_namespace = {"Enum": Enum}
    exec(
        compile(
            ast.Module(body=[service_class], type_ignores=[]),
            str(service_source),
            "exec",
        ),
        service_namespace,
    )
    service = service_namespace["DisaggregationMode"]
    assert isinstance(service.DECODE, Enum)
    config_name = "sglang.srt.hardware_backend.npu.hisparse.offload_config"
    config_spec = importlib.util.spec_from_file_location(
        config_name, SRT / "hardware_backend/npu/hisparse/offload_config.py"
    )
    config = importlib.util.module_from_spec(config_spec)
    monkeypatch.setitem(sys.modules, config_name, config)
    config_spec.loader.exec_module(config)
    window = SimpleNamespace(address=500, window_bytes=configured or 64 * 1024 * 1024)
    create_window = Mock(return_value=window)
    staging_name = "sglang.srt.hardware_backend.npu.hisparse.layer_staging"
    staging = ModuleType(staging_name)
    staging.create_npu_layer_staging = create_window
    monkeypatch.setitem(sys.modules, staging_name, staging)

    class CommonManager:
        def __init__(self, args, disaggregation_mode, server_args, is_mla_backend):
            self.kv_args = args
            self.disaggregation_mode = disaggregation_mode

    source = SRT / "disaggregation/mooncake/conn.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MooncakeKVManager"
    )
    manager.bases = [ast.Name(id="CommonManager", ctx=ast.Load())]
    manager.body = [
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    namespace = dict(
        __name__=__name__,
        CommonManager=CommonManager,
        threading=threading,
        DisaggregationMode=service,
        envs=SimpleNamespace(
            SGLANG_NPU_LAYER_STAGING_BYTES=SimpleNamespace(get=lambda: configured),
            SGLANG_DISAGG_STAGING_BUFFER=SimpleNamespace(get=lambda: False),
        ),
        get_observability=lambda: SimpleNamespace(enable_trace=False),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            manager,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    manager_class = namespace["MooncakeKVManager"]
    manager_class.init_engine = Mock()
    manager_class.register_buffer_to_engine = Mock()
    manager_class.start_decode_thread = Mock()
    region = SimpleNamespace(host_ptr=100, nbytes=64)
    backing = SimpleNamespace(pool=SimpleNamespace(device_id=4), regions=[[region]])
    pool = SimpleNamespace(host_backings=(backing,)) if host_offload else None
    actual = manager_class(
        SimpleNamespace(npu_layer_receive_pool=pool),
        service.DECODE,
        SimpleNamespace(),
        True,
    )
    if host_offload:
        create_window.assert_called_once_with(configured or 64 * 1024 * 1024, 4)
        assert actual._layer_staging is window
        assert actual._layer_staged_host_regions == {(100, 64)}
    else:
        create_window.assert_not_called()
        assert actual._layer_staging is None
    manager_class.register_buffer_to_engine.assert_called_once()
    manager_class.start_decode_thread.assert_called_once()


def prefix_queue(module):
    queue = module.DecodePreallocQueue()
    queue.scheduler = SimpleNamespace(
        enable_hisparse=True,
        server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
        last_batch=None,
    )
    queue.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=1,
        alloc=Mock(side_effect=lambda count: torch.arange(50, 50 + count)),
        available_size=lambda: 20,
    )
    queue._uses_swa_tail_prealloc = lambda: False
    queue._swa_tail_len = lambda length: length
    queue._required_alloc_tokens = (
        lambda **kwargs: kwargs["fill_len"] - kwargs["prefix_len"]
    )
    queue._radix_full_available = lambda: 20
    queue._radix_full_evictable = lambda: 7
    rows = torch.zeros((2, 16), dtype=torch.int64)
    queue.req_to_token_pool = SimpleNamespace(
        alloc=lambda reqs: [1],
        write=lambda key, value: rows.__setitem__(key, value),
        req_to_token=rows,
    )
    queue._need_space_for_single_req = lambda retractable: 3
    queue.retracted_queue = []
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=1),
        rid="prefix",
        origin_input_ids=list(range(8)),
        output_ids=[9],
        set_extend_range=Mock(),
    )
    return queue, req


@pytest.mark.parametrize("prefix_len,total_prefix_len", [(0, 0), (2, 2), (2, 4)])
def test_npu_prealloc_preserves_device_prefix_and_hicache_restore_gap(
    prefix_len, total_prefix_len
):
    module = decode_definitions()
    queue, req = prefix_queue(module)
    prefix = torch.arange(10, 10 + prefix_len)
    result = queue._pre_alloc(req, prefix, prefix_len, total_prefix_len)
    queue.token_to_kv_pool_allocator.alloc.assert_called_once_with(8 - total_prefix_len)
    assert result.tolist() == list(range(50, 58 - total_prefix_len))
    assert (
        queue.req_to_token_pool.req_to_token[1, :prefix_len].tolist() == prefix.tolist()
    )
    assert (
        queue.req_to_token_pool.req_to_token[1, prefix_len:total_prefix_len].eq(0).all()
    )
    req.set_extend_range.assert_called_once_with(total_prefix_len, 8)


@pytest.mark.parametrize("native,expected", [(False, 21), (True, 94)])
def test_admission_uses_backend_allocator_budget(native, expected):
    module = decode_definitions(native)
    queue, _ = prefix_queue(module)
    if native:
        queue.token_to_kv_pool_allocator.logical_attn_allocator = SimpleNamespace(
            available_size=lambda: 100
        )
    assert (
        queue._allocatable_token_budgets(reserved_tokens=6, hicache_reserved_tokens=0)
        == expected
    )


@pytest.mark.parametrize("native,exceeds", [(False, True), (True, False)])
def test_request_capacity_keeps_npu_resolved_budget(native, exceeds):
    module = decode_definitions(native)
    queue, req = prefix_queue(module)
    queue.max_total_num_tokens = 4
    queue.scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(max_token_pool_size=16)
    )
    queue.scheduler.output_streamer = Mock()
    queue._check_if_req_exceed_kv_capacity.__globals__["prepare_abort"] = Mock()
    req.return_logprob = False
    assert queue._check_if_req_exceed_kv_capacity(req) == exceeds


def test_native_prealloc_retains_logical_allocation_and_host_destinations():
    module = decode_definitions(True)
    queue, req = prefix_queue(module)
    allocator = queue.token_to_kv_pool_allocator
    allocator.device = "cpu"
    allocator.alloc_logical_only = Mock(return_value=torch.arange(20, 28))
    host_slots = torch.arange(40, 48)
    host_pool = SimpleNamespace(alloc_paged_token_slots=Mock(return_value=host_slots))
    queue.scheduler.hisparse_coordinator = SimpleNamespace(
        mem_pool_host=host_pool,
        req_to_host_pool=object(),
        req_to_host_pool_allocated_len=object(),
        host_token_len=lambda length: length,
    )
    assert queue._pre_alloc(req) is host_slots
    assert queue.req_to_token_pool.req_to_token[1, :8].tolist() == list(range(20, 28))
    allocator.alloc.assert_not_called()
    assert allocator.alloc_logical_only.call_args.kwargs["extend_num_tokens"] == 8
    host_pool.alloc_paged_token_slots.assert_called_once()


def test_npu_preallocation_publishes_prefix_suffix_and_kda_metadata():
    module = decode_definitions()
    queue, req = prefix_queue(module)
    namespace = queue.pop_preallocated.__globals__
    state_names = (
        "MAMBA",
        "SWA",
        "DSA",
        "DSA_TAIL",
        "MINIMAX_INDEX_K",
        "SWA_RING",
        "C128_STATE",
        "BLOCK_SCALE",
        "BLOCK_SCALE_SWA",
    )
    namespace.update(
        StateType=SimpleNamespace(**{name: name for name in state_names}),
        kv_to_page_indices=lambda indices, page: np.asarray(indices)[::page] // page,
    )
    queue.pp_size = 1
    queue._resolve_pending_reqs = Mock()
    queue._update_handshake_waiters = Mock()
    queue._hicache_pending_restore_tokens = lambda: 0
    queue._allocatable_token_budgets = lambda **kwargs: 32
    queue._match_prefix_and_lock = lambda request: SimpleNamespace(
        prefix_indices=torch.tensor([10, 11]),
        l1_prefix_len=2,
        decode_prefix_len=4,
        restore_token_count=2,
    )
    queue.scheduler.running_batch = SimpleNamespace(reqs=[])
    queue.scheduler.enable_priority_scheduling = False
    queue.scheduler.enable_decode_hicache = True
    queue._start_hicache_prefetch = Mock()
    queue.req_to_token_pool.available_size = lambda: 1
    queue.req_to_token_pool.req_index_to_mamba_index_mapping = {1: torch.tensor([3])}
    queue.req_to_token_pool.translate_mamba_indices = lambda indices: indices + 10
    queue.token_to_kv_pool_allocator.translate_kv_indices_for_transfer = (
        lambda indices: indices + 100
    )
    queue.token_to_kv_pool = SimpleNamespace(page_size=1)
    queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
        available_size=lambda: 1, alloc=lambda: 0
    )
    queue.kv_manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            state_types=["MAMBA", "DSA"],
            npu_layer_receive_pool=object(),
        )
    )
    queue.num_reserved_decode_tokens = 0
    queue.transfer_queue = SimpleNamespace(enable_staging=False)
    queue._num_published_destinations = 0
    req.finished_reason = None
    req.sampling_params = SimpleNamespace(max_new_tokens=1)
    req.time_stats = SimpleNamespace(set_decode_transfer_queue_entry_time=Mock())
    receiver = SimpleNamespace(send_metadata=Mock())
    request = SimpleNamespace(
        req=req, kv_receiver=receiver, waiting_for_input=True, is_rebootstrap=False
    )
    queue.queue, queue.pending_reqs = [request], []
    admitted, failed = queue.pop_preallocated()
    assert admitted == [request] and not failed
    args, kwargs = receiver.send_metadata.call_args
    np.testing.assert_array_equal(args[0], [150, 151, 152, 153])
    np.testing.assert_array_equal(args[2][0][0], [13])
    assert kwargs["decode_prefix_len"] == 4
    assert kwargs["layer_token_count"] == 8
    np.testing.assert_array_equal(
        kwargs["layer_device_kv_indices"], [10, 11, 0, 0, 50, 51, 52, 53]
    )
    assert "device_kv_indices" not in kwargs
    queue._start_hicache_prefetch.assert_called_once()


@pytest.mark.parametrize("native", [False, True])
def test_transferred_npu_requests_skip_native_host_admission(native):
    module = decode_definitions(native)
    process = module.SchedulerDisaggregationDecodeMixin.process_decode_queue
    process.__globals__["get_disagg"] = lambda: SimpleNamespace(
        disaggregation_decode_enable_offload_kvcache=False
    )
    req = object()
    coordinator = Mock()
    scheduler = SimpleNamespace(
        enable_hisparse=True,
        enable_decode_hicache=False,
        hisparse_coordinator=coordinator,
        disagg_decode_transfer_queue=SimpleNamespace(
            resolve_deferred_releases=Mock(),
            extend=Mock(),
            pop_transferred=lambda: [req],
        ),
        disagg_decode_prealloc_queue=SimpleNamespace(
            resume_retracted_reqs=lambda: [],
            retracted_queue=[],
            pop_preallocated=lambda: ([], []),
        ),
        waiting_queue=[],
        polling_count=0,
        polling_interval=1,
    )
    process(scheduler)
    assert scheduler.waiting_queue == [req]
    if native:
        coordinator.admit_request_direct.assert_called_once_with(req)
    else:
        coordinator.admit_request_direct.assert_not_called()


@pytest.mark.parametrize("failure", ["transfer", "corruption"])
def test_npu_transfer_abort_still_calls_coordinator_cleanup(failure):
    module = decode_definitions()
    queue = module.DecodeTransferQueue()
    namespace = queue.pop_transferred.__globals__
    namespace.update(
        KVPoll=SimpleNamespace(Failed="failed", Success="success"),
        HiCacheRestoreResult=SimpleNamespace(FAILED="failed", PENDING="pending"),
        prepare_abort=Mock(),
        release_kv_cache=Mock(),
    )
    req = SimpleNamespace(
        rid="failed",
        bootstrap_room=5,
        return_logprob=False,
        finished_reason=namespace["FINISH_ABORT"](),
    )
    receiver = SimpleNamespace(
        failure_exception=Mock(),
        clear=Mock(),
        kv_mgr=SimpleNamespace(kv_args=SimpleNamespace()),
    )
    queue.queue = [
        SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            hicache_restore_status="done",
            metadata_buffer_index=0,
        )
    ]
    queue.scheduler = SimpleNamespace(
        enable_hisparse=True,
        enable_decode_hicache=False,
        hisparse_coordinator=Mock(),
        output_streamer=Mock(),
        metrics_reporter=SimpleNamespace(enable_metrics=False),
    )
    queue.enable_staging = queue.enable_deferred_kv_release = False
    queue._poll_with_metadata_gate = lambda: [
        "failed" if failure == "transfer" else "success"
    ]
    queue._clean_hicache_prefetch_resources = Mock()
    queue._commit_transfer_to_req = Mock()
    queue.tp_rank, queue.tree_cache = 0, object()
    queue.metadata_buffers = SimpleNamespace(bootstrap_room=[5])
    queue.req_to_metadata_buffer_idx_allocator = Mock()
    assert queue.pop_transferred() == []
    queue.scheduler.hisparse_coordinator.request_finished.assert_called_once_with(req)
    namespace["release_kv_cache"].assert_called_once_with(
        req, queue.tree_cache, is_insert=False
    )


def observer_definitions(native):
    return production_definitions(
        SRT / "managers/scheduler_components/pool_stats_observer.py",
        {"PoolStats": None, "SchedulerPoolStatsObserver": None},
        dataclasses=dataclasses,
        dataclass=dataclass,
        uses_native_hisparse=native_dispatch(native),
        UnifiedMambaSWATokenToKVPoolAllocator=type("UnifiedAllocator", (), {}),
    )


def observer(module, **changes):
    fields = dict(
        tree_cache=SimpleNamespace(
            full_evictable_size=lambda: 4, swa_evictable_size=lambda: 4
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            full_available_size=lambda: 8, swa_available_size=lambda: 8
        ),
        req_to_token_pool=None,
        session_controller=None,
        hisparse_coordinator=None,
        is_hybrid_swa=True,
        is_hybrid_ssm=False,
        enable_hisparse=True,
        full_tokens_per_layer=10,
        swa_tokens_per_layer=10,
        max_total_num_tokens=10,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )
    fields.update(changes)
    return module.SchedulerPoolStatsObserver(**fields)


@pytest.mark.parametrize("native,expected", [(False, -2), (True, 0)])
def test_only_native_hisparse_clamps_swa_accounting(native, expected):
    module = observer_definitions(native)
    stats = observer(module)._get_swa_token_info()
    assert stats.full_num_used == stats.swa_num_used == expected


@pytest.mark.parametrize("reported", [False, True])
def test_npu_stats_preserve_joint_pool_capacity_and_optional_overlay(reported):
    module = observer_definitions(False)
    breakdown = SimpleNamespace(
        device_tokens=2, device_token_usage=0.2, host_tokens=4, host_token_usage=0.4
    )
    coordinator = SimpleNamespace(
        get_token_stats=lambda: breakdown if reported else None
    )
    stats = observer(
        module,
        hisparse_coordinator=coordinator,
        token_to_kv_pool_allocator=SimpleNamespace(
            full_available_size=lambda: 4,
            swa_available_size=lambda: 3,
        ),
    ).get_pool_stats()
    assert stats.full_num_used == 2 and stats.swa_num_used == 3
    assert stats.is_hisparse == reported
    if reported:
        assert stats.hisparse_host_tokens == 4


@pytest.mark.parametrize("int8_checkpoints", [False, True])
def test_npu_stats_keep_mamba_capacity_and_checkpoint_accounting(int8_checkpoints):
    module = observer_definitions(False)
    req_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(size=8),
        mamba_allocator=SimpleNamespace(available_size=lambda: 3),
        schedulable_token_capacity=lambda size: 20,
        mamba_ckpt_pool=object() if int8_checkpoints else None,
    )
    stats = observer(
        module,
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        req_to_token_pool=req_pool,
        tree_cache=SimpleNamespace(
            supports_mamba=lambda: True,
            is_tree_cache=lambda: True,
            full_evictable_size=lambda: 4,
            mamba_evictable_size=lambda: 2,
        ),
        token_to_kv_pool_allocator=SimpleNamespace(size=100, available_size=lambda: 12),
    ).get_pool_stats()
    assert stats.full_num_used == 4 and stats.full_token_usage == 0.2
    assert stats.mamba_num_used == (5 if int8_checkpoints else 3)
    assert stats.mamba_evictable_size == (0 if int8_checkpoints else 2)
