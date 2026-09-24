"""Integrated MTP admission and graph dispatch without accelerator execution."""

import ast
import contextlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from test_hicache_memfabric_pool import runtime as runtime
from test_offload_config import MODULE as config
from test_offload_config import service
from test_prefill_overlap import setup as setup

ROOT = Path(__file__).parents[4] / "python/sglang/srt"


def production_method(relative_path, name, namespace):
    source = ROOT / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]




def test_hicache_plan_binds_draft_before_attention_and_graph_initialization(runtime):
    target_backing, target_pool = runtime.make()
    draft_backing, draft_pool = runtime.make(layers=1)
    target_backing.bind_hicache(target_pool)
    target_runner = SimpleNamespace(
        token_to_kv_pool=target_pool, spec_algorithm=object()
    )
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft_pool,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[])),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        _draft_model_runners=lambda: (draft_runner,),
    )
    build = production_method(
        "speculative/base_spec_worker.py",
        "_build_hicache_draft_plan",
        {
            "HiCacheDraftPlan": SimpleNamespace,
            "HiCacheDraftMode": SimpleNamespace(PACKED="packed", SIDECAR="sidecar"),
            "get_memory": lambda: SimpleNamespace(enable_hierarchical_cache=True),
            "_can_pack_hicache_mtp": lambda *args: True,
        },
    )
    plan = build(worker)
    assert plan.mode == "packed"
    assert draft_backing.page_binding is target_backing.page_binding
    assert target_backing.hicache_pool.mtp_draft_device_pools == (draft_pool,)


def test_hicache_assembler_borrows_bound_target_and_draft_without_another_pool(runtime):
    target, target_pool = runtime.make()
    draft, draft_pool = runtime.make(layers=1)
    host = target.bind_hicache(target_pool)
    host.attach_mtp_drafts((draft_pool,))
    target_pool.tail_extra_slots = 4
    namespace = {
        "Any": Any,
        "Optional": Optional,
        "get_parallel": lambda: SimpleNamespace(
            attn_cp_size=1, attn_dcp_size=1, dcp_enabled=False
        ),
    }
    validate = production_method(
        "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        "_validate_npu_hicache_parallel",
        namespace,
    )
    namespace["_validate_npu_hicache_parallel"] = validate
    build = production_method(
        "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        "build_kv_host_pool",
        namespace,
    )
    before = list(runtime.events)
    assert (
        build(
            kv_pool=target_pool,
            page_size=8,
            use_mla=True,
            mtp_draft_device_pools=(draft_pool,),
        )
        is host
    )
    assert runtime.events == before
    draft.page_binding = None
    with pytest.raises((ValueError, RuntimeError), match="binding|ownership"):
        validate(target_pool, (draft_pool,))


@pytest.mark.parametrize("decode_graph", ["disabled", "full"])
def test_startup_admits_fixed_width_nextn_hicache_and_keeps_prefix(decode_graph):
    args = SimpleNamespace(
        enable_unified_cache_external_linker=False,
        hicache_storage_backend=None,
        speculative_algorithm="EAGLE",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        speculative_adaptive=False,
        page_size=128,
        chunked_prefill_size=8192,
        tp_size=8,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend=decode_graph, max_bs=1),
            prefill=SimpleNamespace(backend="disabled"),
        ),
        enable_hierarchical_cache=True,
        disable_radix_cache=False,
    )

    def resolve(cfg, reason, **values):
        for key, value in values.items():
            setattr(cfg, key, value)

    startup = production_method(
        "hardware_backend/npu/utils.py",
        "set_default_server_args",
        {
            "resolving_view": lambda cfg: cfg,
            "declare_resolution": resolve,
            "envs": SimpleNamespace(
                SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=SimpleNamespace(get=lambda: True)
            ),
            "get_npu_memory_capacity": lambda: 96 * 1024,
            "use_mla_backend": lambda cfg: True,
        },
    )
    startup(args)
    assert args.enable_hierarchical_cache and not args.disable_radix_cache
    assert args.hicache_io_backend == "kernel_ascend"
    # NPU admission precedes the speculative hook, which supplies steps + 1.
    args.speculative_num_draft_tokens = None
    startup(args)
    args.speculative_num_draft_tokens = 4
    for key, value in [
        ("speculative_eagle_topk", 2),
        ("speculative_num_draft_tokens", 7),
        ("speculative_adaptive", True),
        ("speculative_algorithm", "EAGLE3"),
    ]:
        old = getattr(args, key)
        setattr(args, key, value)
        with pytest.raises(ValueError):
            startup(args)
        setattr(args, key, old)


@pytest.mark.parametrize("speculative", [False, True])
def test_integrated_pool_geometry_preserves_mtp_without_enabling_copy_overlap(
    monkeypatch, speculative
):
    # This tests pool options after argument resolution, not startup admission.
    # Integrated HiCache requires radix and shared target/draft page ownership.
    values = dict(SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=True,
                  SGLANG_NPU_DECODE_HOST_KV_BYTES=1 << 30,
                  SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=True,
                  SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=True,
                  SGLANG_NPU_PREFILL_LAYER_SLOTS=2)
    envs = SimpleNamespace(**{key: SimpleNamespace(get=lambda value=value: value)
                             for key, value in values.items()})
    group = object()
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", SimpleNamespace(envs=envs))
    monkeypatch.setitem(sys.modules, "sglang.srt.hardware_backend.npu.fsa_native",
                        SimpleNamespace(native_ops=lambda: None))
    monkeypatch.setitem(sys.modules, "sglang.srt.hardware_backend.npu.shared_host_kv",
                        SimpleNamespace(shared_host_group_from_runtime=lambda device: group))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    schedule = SimpleNamespace(enable_mixed_chunk=False)
    parallel = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(
        get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
        get_schedule=lambda: schedule, get_parallel=lambda: parallel,
        get_spec=lambda: SimpleNamespace(speculative_algorithm="EAGLE" if speculative else None)))
    kvc = SimpleNamespace(
        device="npu",
        use_mla_backend=True,
        server_args=service(disable_radix_cache=False, enable_hierarchical_cache=True),
        kv_cache_dtype=torch.bfloat16,
    )
    options = config.fsa_pool_options(kvc, 4)
    expected = dict(
        prefill_layer_slots=2,
        decode_host_budget=1 << 30,
        shared_host_group=group,
        integrated_offload=True,
    )
    assert options == expected
    # Admission still rejects layouts for which the bounded slots cannot be reused.
    schedule.enable_mixed_chunk = True
    with pytest.raises(ValueError):
        config.fsa_pool_options(kvc, 4)
    schedule.enable_mixed_chunk = False
    parallel.attn_cp_size = 2
    with pytest.raises(ValueError):
        config.fsa_pool_options(kvc, 4)


@pytest.mark.parametrize("speculative", [False, True])
def test_runner_keeps_async_prefill_with_mtp(setup, speculative):
    case = setup()
    namespace = dict(envs=SimpleNamespace(
        SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=SimpleNamespace(get=lambda: True),
        SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=SimpleNamespace(get=lambda: True)))
    initialize = production_method("model_executor/model_runner.py",
                                   "maybe_init_prefill_layerwise_offload", namespace)
    runner = SimpleNamespace(token_to_kv_pool=case.pool,
                             req_to_token_pool=case.runtime.request_pool,
                             spec_algorithm=SimpleNamespace(is_none=lambda: not speculative))
    initialize(runner)
    runtime = runner.prefill_offload_runtime
    assert case.pool.prefill_offload_runtime is runtime
    assert runtime.h2d_prefetch
    assert runtime.d2h_async
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0], req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:0],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True))
    with runtime.forward(batch):
        for layer in range(7, 11):
            runtime.prepare_layer(layer)
            runtime.on_compute_start(layer)
            runtime.finish_layer(layer)
    assert len(case.device.created) == 2
    runtime.release_request("request")


@pytest.mark.parametrize("extend", [False, True])
def test_draft_graph_waits_for_async_prefill_on_another_stream(setup, extend):
    case = setup()
    runtime = case.runtime
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0], req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True))
    with runtime.forward(batch):
        for layer in range(7, 11):
            runtime.prepare_layer(layer)
            case.pool.raw_layer_components(layer)[0][case.mapping[:2]] = layer
            case.device.current.mark(f"scatter-1-{layer - 7}")
            runtime.on_kv_written(layer, case.mapping[:2])
            runtime.on_compute_start(layer)
            runtime.finish_layer(layer)
    run = production_method("speculative/eagle_worker_v2.py", "_execute_draft_graph", {})
    reader = case.device.Stream("cpu")
    graph_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False))
    selection = object()
    result = object()

    def execute(actual_batch, *args):
        assert actual_batch is graph_batch
        assert args == ((selection,) if extend else ())
        assert not runtime.active
        assert all(f"save-1-{layer}" in case.device.current.tail for layer in range(4))
        assert runtime.host_visible.dependencies <= case.device.current.tail
        assert not case.device.synchronized  # Device-side dependency, no host barrier.
        return result

    worker = SimpleNamespace(draft_runner=SimpleNamespace(prefill_offload_forward=runtime.forward))
    with case.device.stream(reader):
        assert not reader.tail
        assert run(worker, SimpleNamespace(execute=execute), graph_batch,
                   *((selection,) if extend else ())) is result
    assert runtime.generation == 1
    runtime.release_request("request")


def test_draft_graph_does_not_execute_after_prefill_failure(setup):
    case = setup()
    run = production_method("speculative/eagle_worker_v2.py", "_execute_draft_graph", {})
    case.runtime.failed = True
    worker = SimpleNamespace(draft_runner=SimpleNamespace(prefill_offload_forward=case.runtime.forward))
    with pytest.raises(RuntimeError, match="failed"):
        run(worker, SimpleNamespace(execute=lambda *args: pytest.fail("unsafe replay")),
            SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False)))


@pytest.mark.parametrize("extend", [False, True])
def test_real_mtp_dispatch_gates_both_graph_entries(setup, extend):
    case = setup()
    case.device.current.mark("published-prefill")
    case.runtime.host_visible = case.runtime._record(case.device.current)
    forward = SimpleNamespace(forward_mode=SimpleNamespace(
        is_extend_without_speculative=lambda: False, is_idle=lambda: False))

    def execute(batch, *args):
        assert batch is forward
        assert "published-prefill" in case.device.current.tail
        if extend:
            assert args[0].tolist() == [1]
            return SimpleNamespace(next_token_logits=torch.tensor([[0.0, 2.0]]),
                                   hidden_states=torch.tensor([[7.0]]))
        assert not args
        return (1, 2, 3, 4)

    graph = SimpleNamespace(execute=execute, can_run_graph=lambda batch: True)
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(canary_manager=None,
            prefill_offload_forward=case.runtime.forward),
        cuda_graph_runner=graph, cuda_graph_runner_for_draft_extend=graph,
        req_to_token_pool=case.runtime.request_pool, topk=1, speculative_num_steps=3,
        speculative_num_draft_tokens=4, seed_dsa_topk_from_draft_extend=False,
        target_worker=object(), tree_mask_mode=None, device="cpu",
        plan_stream=None, plan_stream_ctx=contextlib.nullcontext())
    gate = production_method("speculative/eagle_worker_v2.py", "_execute_draft_graph", {})
    worker._execute_draft_graph = lambda *args: gate(worker, *args)
    namespace = dict(contextlib=contextlib, torch=torch, ScheduleBatch=object,
                     GenerationBatchResult=object, EagleDraftExtendInput=SimpleNamespace,
                     prepare_for_draft=lambda *args: (forward, True),
                     prepare_for_draft_extend=lambda *args, **kwargs: forward,
                     build_eagle_verify_input=lambda *args, **kwargs: args[2:6],
                     maybe_detect_nan=lambda *args: None, maybe_detect_inf=lambda *args: None,
                     _is_hip=False, get_spec=lambda: SimpleNamespace(speculative_use_rejection_sampling=False))
    method = production_method("speculative/eagle_worker_v2.py",
        "_draft_extend_for_decode" if extend else "draft", namespace)
    batch = SimpleNamespace(spec_info=object(), seq_lens=torch.tensor([4]))
    reader = case.device.Stream("cpu")
    with case.device.stream(reader):
        if extend:
            output = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=None),
                accept_lens=torch.tensor([2]), next_token_ids=torch.tensor([1, 2, 3, 4]),
                next_draft_input=SimpleNamespace())
            method(worker, batch, output)
            assert output.next_draft_input.topk_index.tolist() == [[1]]
        else:
            assert method(worker, batch) == (1, 2, 3, 4)


@pytest.mark.parametrize("integrated", [False, True])
def test_integrated_draft_captures_decode_and_extend_graphs(integrated):
    class DeviceGraph:
        def __init__(self, worker):
            self.worker = worker

    class ExtendGraph(DeviceGraph):
        pass

    disabled = SimpleNamespace(get=lambda: False)
    namespace = dict(
        _is_cpu=False, _is_hip=False, _is_cuda=False, _is_musa=False,
        _is_npu=True, _is_xpu=False, time=time, logger=None,
        Phase=SimpleNamespace(DECODE="decode"), Backend=SimpleNamespace(DISABLED="disabled"),
        check_cuda_graph_backend=lambda *args: False,
        get_model=lambda: SimpleNamespace(model_impl="auto"),
        get_exec=lambda: SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=
            SimpleNamespace(decode=SimpleNamespace(backend="full")))),
        get_batch_sizes_to_capture=lambda runner: ([1, 2, 4, 8, 16, 32], 32),
        get_available_gpu_memory=lambda *args: 1.0,
        log_info_on_rank0=lambda *args: None,
        envs=SimpleNamespace(SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=disabled),
    )
    for name in ("EAGLEDraftCudaGraphRunner", "EAGLEDraftNpuGraphRunner"):
        namespace[name] = DeviceGraph
    for name in ("EAGLEDraftExtendCudaGraphRunner", "EAGLEDraftExtendNpuGraphRunner"):
        namespace[name] = ExtendGraph
    for name in ("TritonAttnBackend", "TRTLLMMLABackend", "TRTLLMHAAttnBackend",
                 "TokenspeedMLABackend", "FlashInferAttnBackend"):
        namespace[name] = type(name, (), {})
    capture = production_method("speculative/eagle_worker_v2.py", "_capture_cuda_graphs", namespace)
    runtime = SimpleNamespace(pool=SimpleNamespace(integrated_offload=integrated))
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(prefill_offload_runtime=runtime),
        target_worker=SimpleNamespace(device="npu"), device="npu", gpu_id=0,
        speculative_num_steps=3, speculative_num_draft_tokens=4, topk=1,
        draft_extend_attn_backend=object(),
        _specialized_graph_memory_usage={}, _specialized_graph_time_usage={})
    capture(worker)
    if integrated:
        assert isinstance(worker.cuda_graph_runner, DeviceGraph)
        assert isinstance(worker.cuda_graph_runner_for_draft_extend, ExtendGraph)
        assert set(worker._specialized_graph_time_usage) == {"draft_decode", "draft_extend"}
    else:
        assert worker.cuda_graph_runner is None
        assert worker.cuda_graph_runner_for_draft_extend is None
