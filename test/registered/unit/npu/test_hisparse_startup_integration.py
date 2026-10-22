"""Exercise startup declarations through pool and attention initialization."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import test_hisparse_config as config_tests
import torch
from test_hisparse_config import CONFIG, args, budget, model, startup_args

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

startup = config_tests.startup

ROOT = Path(__file__).resolve().parents[4]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu"


def load_source(monkeypatch, name, relative):
    spec = importlib.util.spec_from_file_location(name, NPU / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def stub_module(monkeypatch, name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize("backend", ["lim_copy_sfa", "external_plan"])
def test_legacy_prefill_preserves_disabled_fusion_and_ordinary_attention(backend):
    server = args("prefill", enable_hisparse=False, hisparse_config=None)
    config = CONFIG.resolve_npu_hisparse_config(
        server,
        {
            "SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD": "1",
            "SGLANG_NPU_PREFILL_HOST_KV_BYTES": str(1 << 30),
            "SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION": "0",
            "SGLANG_NPU_DSA_OFFLOAD_BACKEND": backend,
        },
    )
    assert not config.fused_sparse_attention
    CONFIG.validate_npu_hisparse(
        server,
        config,
        model(index_topk=4096, index_kpool=1, qk_rope_head_dim=64),
        is_dsa=True,
        is_v4=False,
    )


def runtime_inputs(startup, monkeypatch, role, **changes):
    server = startup_args(
        startup,
        role,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        optimistic_prefill_attempts=0,
        **changes,
    )
    startup.utils.set_default_server_args(server)
    startup.hook.validate_hisparse(server)
    view = startup.overrides.resolving_view(server)
    context = sys.modules["sglang.srt.runtime_context"]
    context.get_disagg = lambda: view
    context.get_memory = lambda: view
    context.get_spec = lambda: view
    context.get_parallel = lambda: SimpleNamespace(
        attn_cp_size=1,
        attn_dcp_size=1,
        enable_prefill_cp=False,
        enable_dsa_cache_layer_split=False,
    )
    context.get_schedule = lambda: SimpleNamespace(enable_mixed_chunk=False)
    stub_module(
        monkeypatch,
        "sglang.srt.mem_cache.pool_host.base",
        host_memory_budget_bytes=lambda **kwargs: 1 << 40,
        ranks_per_host=lambda: 1,
    )
    stub_module(
        monkeypatch,
        "sglang.srt.distributed.parallel_state",
        get_attn_tp_group=lambda: SimpleNamespace(world_size=1),
    )
    context.attention_backends = lambda: (
        view.prefill_attention_backend,
        view.decode_attention_backend,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False
    )
    native = Mock()
    stub_module(
        monkeypatch, "sglang.srt.hardware_backend.npu.fsa_native", native_ops=native
    )
    shared = object()
    stub_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.shared_host_kv",
        shared_host_group_from_runtime=lambda device: shared,
    )
    load_source(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.hisparse.config",
        "hisparse/config.py",
    )
    return server, view, native, shared


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
@pytest.mark.parametrize("legacy", [False, True])
def test_minimal_startup_reaches_pool_options_without_repeating_derived_flags(
    startup, monkeypatch, role, legacy
):
    changes = {}
    if legacy:
        config = CONFIG.resolve_npu_hisparse_config(args(role, budget(role)), {})
        for name, value in config.environment().items():
            monkeypatch.setenv(name, value)
        changes = dict(enable_hisparse=False, hisparse_config=None)
    server, view, native, shared = runtime_inputs(startup, monkeypatch, role, **changes)
    admission = load_source(
        monkeypatch, "startup_pool_options", "hisparse/offload_config.py"
    )
    options = admission.fsa_pool_options(
        SimpleNamespace(
            server_args=server,
            device="npu",
            use_mla_backend=True,
            kv_cache_dtype=torch.bfloat16,
            model_dtype=torch.bfloat16,
        ),
        4,
    )
    native.assert_called_once_with()
    assert not server.enable_hierarchical_cache
    assert not server.disable_radix_cache
    if role == "prefill":
        assert options["prefill_layer_slots"] == 2
        assert options["host_memory_budget_plan"].domain_reserve_bytes == 1 << 30
        assert options["host_memory_budget_plan"].reserve_alignment_bytes == 1
        assert view.disable_radix_cache
    else:
        assert options["decode_host_budget"] == 1 << 30
        assert options["shared_host_group"] is shared
        assert view.enable_hierarchical_cache


def test_prefill_backend_never_constructs_decode_planner(startup, monkeypatch):
    runtime_inputs(startup, monkeypatch, "prefill")
    state = Mock(side_effect=AssertionError("unexpected Decode allocation"))
    stub_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention",
        FusedSparseAttentionState=state,
        paged_kv_view=Mock(),
        run_fused_sparse_attention=Mock(),
    )
    stub_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.hisparse.speculative_sparse",
        sparse_mtp_rows=Mock(),
        sparse_mtp_tail=Mock(),
    )
    sys.modules["sglang.srt.configs.model_config"].dsa_layer_skips_topk = Mock()
    adapter = load_source(monkeypatch, "startup_fsa_backend", "hisparse/fsa_backend.py")
    backend = SimpleNamespace()
    adapter.initialize_fsa_backend(backend, None)
    assert backend.fused_sparse_state is None
    state.assert_not_called()
    sys.modules[
        "sglang.srt.hardware_backend.npu.fsa_native"
    ].native_ops.assert_not_called()


@pytest.mark.parametrize("registration_fails", [False, True])
def test_prefill_builder_consumes_resolved_radix_configuration(
    startup, monkeypatch, registration_fails
):
    server, _, _, _ = runtime_inputs(startup, monkeypatch, "prefill")
    stub_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.hisparse.speculative_sparse",
        sparse_mtp_tail=lambda *unused: 0,
    )
    for name in (
        "prefill_offload",
        "prefill_offload_adapter",
        "prefill_offload_runtime",
    ):
        load_source(
            monkeypatch,
            "sglang.srt.hardware_backend.npu.hisparse." + name,
            "hisparse/" + name + ".py",
        )
    host = SimpleNamespace(allocate_component=Mock(), close=Mock())
    if registration_fails:
        host.allocate_component.side_effect = [
            None,
            RuntimeError("registration failed"),
        ]
    stub_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.registered_host_kv",
        NpuRegisteredHostKvPool=Mock(return_value=host),
        NpuLayerwiseKvSlotPool=object,
    )
    builder = load_source(monkeypatch, "startup_prefill_builder", "hisparse/prefill.py")
    pool = SimpleNamespace(
        layer_num=2,
        prefill_layer_slots=2,
        size=256,
        page_size=128,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
        _kpool_use_compress=True,
        store_dtype=torch.bfloat16,
        start_layer=0,
    )
    runner = SimpleNamespace(
        server_args=server,
        device="npu",
        token_to_kv_pool=pool,
        req_to_token_pool=object(),
        ps=SimpleNamespace(attn_cp_size=1, attn_dcp_size=1),
        spec_algorithm=object(),
        model_config=object(),
        page_size=128,
    )
    if registration_fails:
        with pytest.raises(RuntimeError, match="registration failed"):
            builder.initialize_prefill_runtime(runner)
        host.close.assert_called_once_with(drained=True)
        assert not hasattr(runner, "prefill_offload_runtime")
        return
    builder.initialize_prefill_runtime(runner)
    assert runner.prefill_offload_runtime.pool is pool
    assert runner.prefill_offload_runtime.adapter.host_pool is host
    assert host.allocate_component.call_count == 2
    host.close.assert_not_called()


@pytest.mark.parametrize("role", ["null", "decode"])
@pytest.mark.parametrize(
    "changes",
    [
        {"kv_cache_dtype": "float16"},
        {"dtype": "float16"},
    ],
)
def test_lim_rejects_explicit_fp16_before_allocation(role, changes):
    server = args(role, budget(role), **changes)
    config = CONFIG.resolve_npu_hisparse_config(server, {})
    with pytest.raises(ValueError, match="BF16"):
        CONFIG.validate_npu_hisparse(server, config, model(), is_dsa=True, is_v4=False)


@pytest.mark.parametrize("backend", ["lim_copy_sfa", "external_plan"])
def test_prefill_keeps_fp16_support(backend):
    server = args(
        "prefill",
        {**budget("prefill"), "dsa_offload_backend": backend},
        kv_cache_dtype="float16",
        dtype="float16",
    )
    CONFIG.validate_npu_hisparse(
        server,
        CONFIG.resolve_npu_hisparse_config(server, {}),
        model(),
        is_dsa=True,
        is_v4=False,
    )


@pytest.mark.parametrize("dtype", ["auto", "bfloat16"])
def test_lim_auto_checks_model_query_dtype(dtype):
    server = args(npu=budget("null"), kv_cache_dtype=dtype)
    config = CONFIG.resolve_npu_hisparse_config(server, {})
    with pytest.raises(ValueError, match="BF16 query"):
        CONFIG.validate_npu_hisparse(
            server, config, model(torch_dtype="float16"), is_dsa=True, is_v4=False
        )


@pytest.mark.parametrize("backend", ["lim_copy_sfa", "external_plan"])
def test_pool_options_validate_actual_resolved_dtype(startup, monkeypatch, backend):
    server, _, native, _ = runtime_inputs(startup, monkeypatch, "null")
    monkeypatch.setenv("SGLANG_NPU_DSA_OFFLOAD_BACKEND", backend)
    admission = load_source(
        monkeypatch, "startup_pool_dtype", "hisparse/offload_config.py"
    )
    pool = SimpleNamespace(
        server_args=server,
        device="npu",
        use_mla_backend=True,
        kv_cache_dtype=torch.float16,
        model_dtype=torch.float16,
    )
    if backend == "lim_copy_sfa":
        with pytest.raises(ValueError, match="BF16"):
            admission.fsa_pool_options(pool, 4)
        native.assert_not_called()
    else:
        assert admission.fsa_pool_options(pool, 4)["decode_host_budget"] == 1 << 30


if __name__ == "__main__":
    run_pytest(__file__)
