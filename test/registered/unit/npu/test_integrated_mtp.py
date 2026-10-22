"""Integrated MTP admission and graph dispatch without accelerator execution."""

import ast
import contextlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import test_hicache_memfabric_pool as fabric_tests
import test_prefill_overlap as overlap_tests
import torch
from test_decode_memory_budget import MODULE as budget_module
from test_hisparse_config import CONFIG
from test_offload_config import MODULE as config
from test_offload_config import service
from test_speculative_sparse import MODULE as speculative_module

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import npu_module_name, run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

runtime = fabric_tests.runtime
setup = overlap_tests.setup

ROOT = Path(__file__).parents[4] / "python/sglang/srt"


def production_method(relative_path, name, namespace):
    source = ROOT / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        method,
                    ],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("host_tokens", [0, 131072])
def test_host_capacity_budget_route_includes_mtp_layers(
    monkeypatch, integrated, host_tokens
):
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.hardware_backend.npu.hisparse.offload_config"),
        config,
    )
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.hardware_backend.npu.hisparse.speculative_sparse"),
        speculative_module,
    )

    values = dict(
        SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=integrated,
        SGLANG_NPU_HICACHE_HOST_TOKENS=host_tokens,
        SGLANG_NPU_PREFILL_LAYER_SLOTS=2,
        SGLANG_NPU_FSA_RESIDENT_CAPACITY=4096,
        SGLANG_NPU_DECODE_HOST_KV_BYTES=1 << 34,
        SGLANG_NPU_SHARED_HOST_KV=True,
        SGLANG_NPU_LAYER_STAGING_BYTES=0,
        SGLANG_NPU_DSA_OFFLOAD_BACKEND="external_plan",
    )
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.environ"),
        SimpleNamespace(
            envs=SimpleNamespace(
                **{
                    key: SimpleNamespace(get=lambda value=value: value)
                    for key, value in values.items()
                }
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.configs.model_config"),
        SimpleNamespace(
            is_deepseek_dsa=lambda config: True,
            dsa_layer_skips_topk=lambda *args: False,
            get_dsa_index_kpool_compress=lambda config: True,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.model_executor.pool_configurator"),
        SimpleNamespace(
            _get_dsa_cache_layer_ids=lambda kvc, layers: list(range(layers))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.mem_cache.kv_cache_configurator"),
        SimpleNamespace(_should_elide_dsa_index_k=lambda **kwargs: False),
    )
    spec = SimpleNamespace(
        speculative_num_steps=5,
        speculative_num_draft_tokens=6,
        speculative_eagle_topk=1,
    )
    graph = SimpleNamespace(backend="full", bs=[1, 2, 4, 8, 12, 16, 24], max_bs=24)
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.runtime_context"),
        SimpleNamespace(
            get_disagg=lambda: SimpleNamespace(
                disaggregation_mode="null" if integrated else "decode"
            ),
            get_spec=lambda: spec,
            max_speculative_num_draft_tokens=lambda: 6,
            get_exec=lambda: SimpleNamespace(
                graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=graph))
            ),
        ),
    )
    hf_config = SimpleNamespace(index_topk=2048, index_kpool=4)
    model = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
        num_nextn_predict_layers=1,
    )
    kvc = SimpleNamespace(
        device="npu",
        use_mla_backend=True,
        model_config=model,
        kv_cache_dtype=torch.bfloat16,
        is_draft_worker=False,
        mambaish_config=None,
        spec_algorithm=SimpleNamespace(is_none=lambda: False, name="EAGLE"),
        spec_aux_config=SimpleNamespace(eagle_draft_num_layers=1),
    )
    actual = budget_module.budget_from_config(kvc, 11)
    assert actual.layers == 12
    assert actual.host_token_capacity == host_tokens
    assert actual.staging_bytes == (0 if integrated else 64 * 1024 * 1024)
    assert actual.host_bytes_per_token == 12 * 512 * 2
    assert actual.prefill_layer_slots == (3 if integrated else 0)
    assert actual.graph_rows == 24 and actual.rows_per_request == 6
    assert actual.index_layers == 12
    assert actual.host_index_sidecar
    assert (
        actual.host_indexer_bytes(8192, 128)
        == (max(8192, host_tokens) + 128) * 12 * 128 * 2 // 4
    )


def test_hicache_plan_registers_draft_for_host_pool_assembly(runtime):
    target_backing, target_pool = runtime.make()
    draft_backing, draft_pool = runtime.make(layers=1)
    target_backing.bind_hicache(target_pool)
    target_runner = SimpleNamespace(
        token_to_kv_pool=target_pool,
        spec_algorithm=object(),
        req_to_token_pool=SimpleNamespace(),
    )
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft_pool,
        req_to_token_pool=SimpleNamespace(),
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
    assert plan.device_pools == (draft_pool,)
    assert target_runner.mtp_draft_device_pools == (draft_pool,)
    assert target_runner.mtp_draft_mamba_pools == ()
    assert draft_backing.page_binding is None


def test_hicache_assembler_borrows_bound_target_and_draft_without_another_pool(runtime):
    target, target_pool = runtime.make()
    draft, draft_pool = runtime.make(layers=1)
    host = target.bind_hicache(target_pool)
    target_pool.tail_extra_slots = 4
    namespace = {
        "Any": Any,
        "Optional": Optional,
        "get_parallel": lambda: SimpleNamespace(
            attn_cp_size=1, attn_dcp_size=1, dcp_enabled=False
        ),
    }
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
    assert draft.page_binding is target.page_binding
    assert host.mtp_draft_device_pools == (draft_pool,)
    with pytest.raises(ValueError, match="physical KV pages"):
        build(kv_pool=target_pool, page_size=3, mtp_draft_device_pools=(draft_pool,))


@pytest.fixture
def draft_page_ownership(runtime):
    module = sys.modules[runtime.binding_type.__module__]

    def create(*, wrapped=False, target_bound=True, **draft_geometry):
        target, target_pool = runtime.make()
        draft, draft_pool = runtime.make(**({"layers": 1} | draft_geometry))
        if target_bound:
            target.bind_hicache(target_pool)
        target_view = (
            SimpleNamespace(full_kv_pool=target_pool) if wrapped else target_pool
        )
        draft_view = SimpleNamespace(full_kv_pool=draft_pool) if wrapped else draft_pool
        return SimpleNamespace(
            module=module,
            target=target,
            target_pool=target_pool,
            draft=draft,
            draft_pool=draft_pool,
            draft_view=draft_view,
            allocator=SimpleNamespace(get_kvcache=lambda: target_view),
            stages=[],
        )

    return create


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("target_bound", [False, True])
def test_early_draft_binding_borrows_target_owner_without_payload_allocation(
    runtime, draft_page_ownership, wrapped, target_bound
):
    case = draft_page_ownership(wrapped=wrapped, target_bound=target_bound)
    before = list(runtime.events)
    allocations = dict(runtime.allocations)
    draft_pointer = case.draft.layer_views(0)[0].data_ptr()
    assert case.draft.page_binding is None

    case.module.bind_draft_page_ownership(case.allocator, case.draft_view)

    host = case.target.hicache_pool
    binding = case.target.page_binding
    assert binding is not None
    assert case.draft.page_binding is binding
    assert case.draft.hicache_pool.binding is binding
    assert host.mtp_draft_device_pools == (case.draft_pool,)
    assert host.layer_num == case.target_pool.layer_num + 1
    assert host.layer_views(case.target_pool.layer_num)[0].data_ptr() == draft_pointer
    assert runtime.events == before
    assert runtime.allocations.keys() == allocations.keys()
    assert all(runtime.allocations[key] is value for key, value in allocations.items())
    assert binding.references.sum().item() == 0
    case.module.bind_draft_page_ownership(case.allocator, case.draft_view)
    assert case.draft.page_binding is binding
    assert host.layer_num == case.target_pool.layer_num + 1
    assert runtime.events == before


@pytest.mark.parametrize(
    "draft_geometry",
    [{"layers": 2}, {"rope_dim": 0}, {"page_size": 8}, {"pages": 6}],
)
def test_early_draft_binding_rejects_incompatible_backing_geometry(
    runtime, draft_page_ownership, draft_geometry
):
    case = draft_page_ownership(wrapped=True, **draft_geometry)
    binding = case.target.page_binding
    before = list(runtime.events)
    with pytest.raises(ValueError, match="geometry"):
        case.module.bind_draft_page_ownership(case.allocator, case.draft_view)
    assert case.target.page_binding is binding
    assert case.draft.page_binding is None
    assert case.target.hicache_pool.mtp_draft_device_pools == ()
    assert runtime.events == before


@pytest.mark.parametrize("missing", ["target", "draft"])
def test_early_draft_binding_rejects_one_sided_offload(
    runtime, draft_page_ownership, missing
):
    case = draft_page_ownership(wrapped=True)
    pool = case.target_pool if missing == "target" else case.draft_pool
    pool.decode_host_backing = None
    before = list(runtime.events)
    with pytest.raises((ValueError, RuntimeError), match="backing|offload|ownership"):
        case.module.bind_draft_page_ownership(case.allocator, case.draft_view)
    assert case.draft.page_binding is None
    assert case.target.hicache_pool.mtp_draft_device_pools == ()
    assert runtime.events == before


def test_early_draft_binding_leaves_nonoffload_pools_unchanged(runtime):
    module = sys.modules[runtime.binding_type.__module__]
    target = SimpleNamespace()
    draft = SimpleNamespace()
    allocator = SimpleNamespace(
        get_kvcache=lambda: SimpleNamespace(full_kv_pool=target)
    )
    before = list(runtime.events)
    module.bind_draft_page_ownership(allocator, SimpleNamespace(full_kv_pool=draft))
    assert vars(target) == {}
    assert vars(draft) == {}
    assert runtime.events == before


def initialize_draft_test_pools(
    case, monkeypatch, *, is_draft_worker=True, is_npu=True
):
    real_bind = case.module.bind_draft_page_ownership

    def bind(allocator, pool):
        assert case.stages[-1] == "allocator"
        case.stages.append("bind")
        return real_bind(allocator, pool)

    monkeypatch.setattr(case.module, "bind_draft_page_ownership", bind)
    request_pool = SimpleNamespace(
        req_to_token=torch.tensor([[4, 5, 6, 7, 0, 0, 0, 0]])
    )

    def build_pool(**kwargs):
        case.stages.append("pool")
        return case.draft_view

    def build_allocator(**kwargs):
        assert kwargs["token_to_kv_pool"] is case.draft_view
        case.stages.append("allocator")
        return case.allocator

    def initialized(**kwargs):
        if is_draft_worker and is_npu:
            assert case.draft.page_binding is case.target.page_binding
            assert case.stages[-1] == "bind"
        else:
            assert case.draft.page_binding is None
            assert "bind" not in case.stages
        case.stages.append("pools_ready")
        return SimpleNamespace(**kwargs)

    unrelated_allocator = type("UnrelatedAllocator", (), {})
    namespace = dict(
        current_platform=SimpleNamespace(is_npu=lambda: is_npu),
        _is_npu=is_npu,
        get_memory=lambda: SimpleNamespace(enable_unified_memory=False),
        get_schedule=lambda: SimpleNamespace(prefill_only_disable_kv_cache=False),
        UnifiedMambaTokenToKVPoolAllocator=unrelated_allocator,
        UnifiedSWAAllocatorBase=unrelated_allocator,
        HiSparseTokenToKVPoolAllocator=unrelated_allocator,
        is_deepseek_dsa=lambda config: True,
        is_deepseek_v4=lambda config: False,
        _InitializedPools=initialized,
    )
    initialize = production_method(
        "mem_cache/kv_cache_configurator.py", "_init_pools", namespace
    )
    configurator = SimpleNamespace(
        device="npu" if is_npu else "cuda",
        is_draft_worker=is_draft_worker,
        is_hybrid_swa_mtp_draft=False,
        kv_shard_rank=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        _build_req_to_token_pool=lambda **kwargs: request_pool,
        _validate_prefill_only_disable_kv_cache_pool_family=lambda *args: None,
        _build_token_to_kv_pool=build_pool,
        _build_token_to_kv_pool_allocator=build_allocator,
    )
    return initialize(
        configurator,
        sizes=SimpleNamespace(max_running_requests=1),
        req_to_token_pool=request_pool if is_draft_worker else None,
        token_to_kv_pool_allocator=case.allocator if is_draft_worker else None,
    )


@pytest.mark.parametrize("is_draft_worker", [False, True])
@pytest.mark.parametrize("is_npu", [False, True])
def test_configurator_binds_only_npu_draft_before_returning_pools(
    draft_page_ownership, monkeypatch, is_draft_worker, is_npu
):
    case = draft_page_ownership(wrapped=True)
    result = initialize_draft_test_pools(
        case, monkeypatch, is_draft_worker=is_draft_worker, is_npu=is_npu
    )
    assert result.token_to_kv_pool is case.draft_view
    assert result.token_to_kv_pool_allocator is case.allocator
    expected = ["pool", "allocator"]
    if is_draft_worker and is_npu:
        expected.append("bind")
    assert case.stages == [*expected, "pools_ready"]


def test_draft_graph_metadata_has_page_owner_before_scheduler_assembler(
    runtime, draft_page_ownership, monkeypatch
):
    case = draft_page_ownership(wrapped=True)
    result = initialize_draft_test_pools(case, monkeypatch)
    binding = case.target.page_binding
    assert binding.ensure_pages(torch.tensor([1]))
    owner_references = binding.references.clone()
    before = list(runtime.events)
    populate = production_method(
        "hardware_backend/npu/attention/ascend_dsa_backend.py",
        "_populate_dsa_metadata",
        dict(torch=torch, AscendDSAForwardMetadata=SimpleNamespace),
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([1]),
        req_pool_indices=torch.tensor([0]),
        batch_size=1,
        forward_mode=SimpleNamespace(
            is_extend=lambda: False,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        ),
    )

    class DecodeGraph:
        phase = "draft_decode"

        def __init__(self, worker):
            assert "assembler" not in case.stages
            assert case.draft.page_binding is binding
            metadata = SimpleNamespace()
            backend = SimpleNamespace(
                forward_metadata=metadata,
                page_size=4,
                dsa_index_kpool=1,
                req_to_token_pool=result.req_to_token_pool,
                token_to_kv_pool=result.token_to_kv_pool,
                _cache_seqlens=lambda batch: batch.seq_lens.to(torch.int32),
                _query_lens_device=lambda batch: torch.ones_like(batch.seq_lens),
                _page_table_max_seq_len=lambda batch: 4,
            )
            populate(backend, batch, _graph_capture=True)
            torch.testing.assert_close(
                metadata.host_page_table,
                binding.host_table(metadata.real_page_table),
            )
            assert metadata.host_page_table[0, 0].item() > 0
            case.stages.append(self.phase)

    class ExtendGraph(DecodeGraph):
        phase = "draft_extend"

    capture = draft_graph_capture_method(DecodeGraph, ExtendGraph)
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(
            prefill_offload_runtime=SimpleNamespace(pool=case.draft_pool)
        ),
        target_worker=SimpleNamespace(device="npu"),
        device="npu",
        gpu_id=0,
        speculative_num_steps=5,
        speculative_num_draft_tokens=6,
        topk=1,
        draft_extend_attn_backend=object(),
        _specialized_graph_memory_usage={},
        _specialized_graph_time_usage={},
    )
    capture(worker)
    assert isinstance(worker.cuda_graph_runner, DecodeGraph)
    assert isinstance(worker.cuda_graph_runner_for_draft_extend, ExtendGraph)
    assemble = production_method(
        "mem_cache/hybrid_cache/hybrid_pool_assembler.py", "build_kv_host_pool", {}
    )
    case.stages.append("assembler")
    for _ in range(2):
        host = assemble(
            kv_pool=case.target_pool,
            page_size=8,
            mtp_draft_device_pools=(case.draft_pool,),
        )
        assert host is case.target.hicache_pool
        assert host.layer_num == case.target_pool.layer_num + 1
        assert case.draft.page_binding is binding
    torch.testing.assert_close(binding.references, owner_references)
    assert runtime.events == before
    assert case.stages == [
        "pool",
        "allocator",
        "bind",
        "pools_ready",
        "draft_decode",
        "draft_extend",
        "assembler",
    ]
    binding.release_device_pages(torch.tensor([1]))


@pytest.mark.parametrize("decode_graph", ["disabled", "full"])
def test_startup_admits_fixed_width_nextn_hicache_and_keeps_prefix(
    decode_graph, monkeypatch
):
    monkeypatch.setitem(
        sys.modules, "sglang.srt.hardware_backend.npu.hisparse.config", CONFIG
    )
    for name, default in CONFIG.ENV_FIELDS.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD", "True")
    monkeypatch.setenv("SGLANG_NPU_DECODE_HOST_KV_BYTES", str(1 << 30))
    args = SimpleNamespace(
        disaggregation_mode="null",
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
                SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=SimpleNamespace(get=lambda: True),
                SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=SimpleNamespace(get=lambda: False),
            ),
            "get_npu_memory_capacity": lambda: 96 * 1024,
            "use_mla_backend": lambda cfg: True,
        },
    )

    def validate_startup():
        startup(args)
        CONFIG.validate_npu_hisparse(
            args,
            CONFIG.resolve_npu_hisparse_config(args),
            SimpleNamespace(
                index_topk=2048,
                index_kpool=4,
                kv_lora_rank=512,
                qk_rope_head_dim=0,
            ),
            is_dsa=True,
            is_v4=False,
        )

    validate_startup()
    assert args.enable_hierarchical_cache and not args.disable_radix_cache
    assert args.hicache_io_backend == "kernel_ascend"
    # NPU admission precedes the speculative hook, which supplies steps + 1.
    args.speculative_num_draft_tokens = None
    validate_startup()
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
            validate_startup()
        setattr(args, key, old)


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("host_tokens", [0, 131072])
@pytest.mark.parametrize("backend", ["lim_copy_sfa", "external_plan"])
def test_integrated_pool_geometry_preserves_mtp_without_enabling_copy_overlap(
    monkeypatch, speculative, host_tokens, backend
):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.arg_groups.overrides",
        SimpleNamespace(resolved_view=lambda value: value),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.hardware_backend.npu.hisparse.config", CONFIG
    )
    # This tests pool options after argument resolution, not startup admission.
    # Integrated HiCache requires radix and shared target/draft page ownership.
    values = dict(
        SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=True,
        SGLANG_NPU_DECODE_HOST_KV_BYTES=1 << 30,
        SGLANG_NPU_HICACHE_HOST_TOKENS=host_tokens,
        SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=True,
        SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=True,
        SGLANG_NPU_PREFILL_LAYER_SLOTS=2,
        SGLANG_NPU_DSA_OFFLOAD_BACKEND=backend,
    )
    envs = SimpleNamespace(
        **{
            key: SimpleNamespace(get=lambda value=value: value)
            for key, value in values.items()
        }
    )
    group = object()
    monkeypatch.setitem(
        sys.modules, npu_module_name("sglang.srt.environ"), SimpleNamespace(envs=envs)
    )
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.hardware_backend.npu.fsa_native"),
        SimpleNamespace(native_ops=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.hardware_backend.npu.shared_host_kv"),
        SimpleNamespace(shared_host_group_from_runtime=lambda device: group),
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False
    )
    schedule = SimpleNamespace(enable_mixed_chunk=False)
    parallel = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1)
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.runtime_context"),
        SimpleNamespace(
            get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
            get_schedule=lambda: schedule,
            get_parallel=lambda: parallel,
            get_spec=lambda: SimpleNamespace(
                speculative_algorithm="EAGLE" if speculative else None
            ),
        ),
    )
    host_plan = budget_module.HostMemoryBudgetPlan(1 << 30, 2 << 30)
    kvc = SimpleNamespace(
        device="npu",
        use_mla_backend=True,
        server_args=service(disable_radix_cache=False, enable_hierarchical_cache=True),
        kv_cache_dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        _offload_host_memory_plan=host_plan,
        is_draft_worker=False,
    )
    options = config.fsa_pool_options(kvc, 4)
    expected = dict(
        prefill_layer_slots=2,
        decode_host_budget=1 << 30,
        shared_host_group=group,
        integrated_offload=True,
        host_memory_budget_plan=host_plan,
    )
    if host_tokens:
        expected["decode_host_token_capacity"] = host_tokens
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
    namespace = dict(
        envs=SimpleNamespace(
            SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=SimpleNamespace(
                get=lambda: True
            ),
            SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=SimpleNamespace(get=lambda: True),
        )
    )
    initialize = production_method(
        "hardware_backend/npu/hisparse/prefill.py",
        "initialize_prefill_runtime",
        namespace,
    )
    runner = SimpleNamespace(
        token_to_kv_pool=case.pool,
        req_to_token_pool=case.runtime.request_pool,
        spec_algorithm=SimpleNamespace(is_none=lambda: not speculative),
    )
    initialize(runner)
    runtime = runner.prefill_offload_runtime
    assert case.pool.prefill_offload_runtime is runtime
    assert runtime.h2d_prefetch
    assert runtime.d2h_async
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:0],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
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
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with runtime.forward(batch):
        for layer in range(7, 11):
            runtime.prepare_layer(layer)
            case.pool.raw_layer_components(layer)[0][case.mapping[:2]] = layer
            case.device.current.mark(f"scatter-1-{layer - 7}")
            runtime.on_kv_written(layer, case.mapping[:2])
            runtime.on_compute_start(layer)
            runtime.finish_layer(layer)
    run = production_method(
        "speculative/eagle_worker_v2.py", "_execute_draft_graph", {}
    )
    reader = case.device.Stream("cpu")
    graph_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False)
    )
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

    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(prefill_offload_forward=runtime.forward)
    )
    with case.device.stream(reader):
        assert not reader.tail
        assert (
            run(
                worker,
                SimpleNamespace(execute=execute),
                graph_batch,
                *((selection,) if extend else ()),
            )
            is result
        )
    assert runtime.generation == 1
    runtime.release_request("request")


def test_draft_graph_does_not_execute_after_prefill_failure(setup):
    case = setup()
    run = production_method(
        "speculative/eagle_worker_v2.py", "_execute_draft_graph", {}
    )
    case.runtime.failed = True
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(prefill_offload_forward=case.runtime.forward)
    )
    with pytest.raises(RuntimeError, match="failed"):
        run(
            worker,
            SimpleNamespace(execute=lambda *args: pytest.fail("unsafe replay")),
            SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_extend_without_speculative=lambda: False
                )
            ),
        )


@pytest.mark.parametrize("extend", [False, True])
def test_real_mtp_dispatch_gates_both_graph_entries(setup, extend):
    case = setup()
    case.device.current.mark("published-prefill")
    case.runtime.host_visible = case.runtime._record(case.device.current)
    forward = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: False, is_idle=lambda: False
        )
    )
    draft_probs = torch.tensor([4.0])

    def execute(batch, *args):
        assert batch is forward
        assert "published-prefill" in case.device.current.tail
        if extend:
            assert args[0].tolist() == [1]
            return SimpleNamespace(
                next_token_logits=torch.tensor([[0.0, 2.0]]),
                hidden_states=torch.tensor([[7.0]]),
            )
        assert not args
        return (1, 2, 3, draft_probs)

    graph = SimpleNamespace(execute=execute, can_run_graph=lambda batch: True)
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(
            canary_manager=None, prefill_offload_forward=case.runtime.forward
        ),
        cuda_graph_runner=graph,
        cuda_graph_runner_for_draft_extend=graph,
        req_to_token_pool=case.runtime.request_pool,
        topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        seed_dsa_topk_from_draft_extend=False,
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(kv_index_translator=None)
        ),
        tree_mask_mode=None,
        device="cpu",
        plan_stream=None,
        plan_stream_ctx=contextlib.nullcontext(),
    )
    gate = production_method(
        "speculative/eagle_worker_v2.py", "_execute_draft_graph", {}
    )
    worker._execute_draft_graph = lambda *args: gate(worker, *args)
    namespace = dict(
        contextlib=contextlib,
        torch=torch,
        ScheduleBatch=object,
        GenerationBatchResult=object,
        EagleDraftExtendInput=SimpleNamespace,
        prepare_for_draft=lambda *args, **kwargs: (forward, True, None),
        prepare_for_draft_extend=lambda *args, **kwargs: forward,
        build_eagle_verify_input=lambda *args, **kwargs: args[2:6],
        maybe_detect_nan=lambda *args: None,
        maybe_detect_inf=lambda *args: None,
        _is_hip=False,
        get_spec=lambda: SimpleNamespace(speculative_use_rejection_sampling=False),
    )
    method = production_method(
        "speculative/eagle_worker_v2.py",
        "_draft_extend_for_decode" if extend else "draft",
        namespace,
    )
    batch = SimpleNamespace(spec_info=object(), seq_lens=torch.tensor([4]))
    reader = case.device.Stream("cpu")
    with case.device.stream(reader):
        if extend:
            output = SimpleNamespace(
                prepared_draft_extend_inputs=None,
                logits_output=SimpleNamespace(hidden_states=None),
                accept_lens=torch.tensor([2]),
                next_token_ids=torch.tensor([1, 2, 3, 4]),
                next_draft_input=SimpleNamespace(),
            )
            method(worker, batch, output)
            assert output.next_draft_input.topk_index.tolist() == [[1]]
        else:
            result = method(worker, batch)
            assert result[:3] == (1, 2, 3)
            torch.testing.assert_close(result[3], draft_probs)
            assert result[3].data_ptr() != draft_probs.data_ptr()


def draft_graph_capture_method(device_graph, extend_graph):
    disabled = SimpleNamespace(get=lambda: False)
    namespace = dict(
        _is_cpu=False,
        _is_hip=False,
        _is_cuda=False,
        _is_musa=False,
        _is_npu=True,
        _is_xpu=False,
        time=time,
        logger=None,
        Phase=SimpleNamespace(DECODE="decode"),
        Backend=SimpleNamespace(DISABLED="disabled"),
        check_cuda_graph_backend=lambda *args: False,
        get_model=lambda: SimpleNamespace(model_impl="auto"),
        get_exec=lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full")
                )
            )
        ),
        get_batch_sizes_to_capture=lambda runner: ([1, 2, 4, 8, 16, 32], 32),
        get_available_gpu_memory=lambda *args: 1.0,
        log_info_on_rank0=lambda *args: None,
        envs=SimpleNamespace(SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=disabled),
    )
    for name in ("EAGLEDraftCudaGraphRunner", "EAGLEDraftNpuGraphRunner"):
        namespace[name] = device_graph
    for name in ("EAGLEDraftExtendCudaGraphRunner", "EAGLEDraftExtendNpuGraphRunner"):
        namespace[name] = extend_graph
    for name in (
        "TritonAttnBackend",
        "TRTLLMMLABackend",
        "TRTLLMHAAttnBackend",
        "TokenspeedMLABackend",
        "FlashInferAttnBackend",
        "QwenSparseAttnBackend",
    ):
        namespace[name] = type(name, (), {})
    return production_method(
        "speculative/eagle_worker_v2.py", "_capture_cuda_graphs", namespace
    )


@pytest.mark.parametrize("integrated", [False, True])
def test_integrated_draft_captures_decode_and_extend_graphs(integrated):
    class DeviceGraph:
        def __init__(self, worker):
            self.worker = worker

    class ExtendGraph(DeviceGraph):
        pass

    capture = draft_graph_capture_method(DeviceGraph, ExtendGraph)
    runtime = SimpleNamespace(pool=SimpleNamespace(integrated_offload=integrated))
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(prefill_offload_runtime=runtime),
        target_worker=SimpleNamespace(device="npu"),
        device="npu",
        gpu_id=0,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        topk=1,
        draft_extend_attn_backend=object(),
        _specialized_graph_memory_usage={},
        _specialized_graph_time_usage={},
    )
    capture(worker)
    if integrated:
        assert isinstance(worker.cuda_graph_runner, DeviceGraph)
        assert isinstance(worker.cuda_graph_runner_for_draft_extend, ExtendGraph)
        assert set(worker._specialized_graph_time_usage) == {
            "draft_decode",
            "draft_extend",
        }
    else:
        assert worker.cuda_graph_runner is None
        assert worker.cuda_graph_runner_for_draft_extend is None


if __name__ == "__main__":
    run_pytest(__file__)
