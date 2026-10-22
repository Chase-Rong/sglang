"""Exercise production PD dispatch on CPU without accelerator imports."""

import ast
import importlib.util
import logging
import sys
import threading
import time
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import test_layer_receive_group as receive_tests
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
PD_NAME = "sglang.srt.hardware_backend.npu.hisparse.pd"
receive_inputs = receive_tests.inputs


def production_definitions(path, classes, functions=(), bases=None, **namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            base = (bases or {}).get(node.name)
            if base is None:
                node.bases = []
            else:
                base_name = "Base" + node.name
                namespace[base_name] = base
                node.bases = [ast.Name(id=base_name, ctx=ast.Load())]
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
    for definition in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definition.decorator_list = []
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
        get_kv_class=lambda backend, kind: (
            (lambda: args) if kind == "args" else manager_class
        ),
        get_parallel=lambda: SimpleNamespace(attn_tp_size=2, dp_rank=0),
        get_device=lambda: SimpleNamespace(gpu_id=0),
        get_pd_kv_layout=lambda pool: None,
        get_kv_transfer_buf_infos=lambda pool: pool.get_contiguous_buf_infos(),
        get_disagg=lambda: SimpleNamespace(
            disaggregation_ib_device="ib0",
            disaggregation_decode_host_receive_threshold=1,
        ),
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
            SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=SimpleNamespace(
                get=lambda: 65536
            ),
        ),
        get_observability=lambda: SimpleNamespace(enable_trace=False),
        check_mooncake_custom_mem_pool_enabled=lambda: (False, None),
        _validate_efa_allocator_compatibility=lambda *args: None,
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


if __name__ == "__main__":
    from sglang.test.npu_test_paths import run_pytest

    run_pytest(__file__)
