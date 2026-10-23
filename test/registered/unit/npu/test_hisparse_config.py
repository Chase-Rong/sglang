"""CPU configuration contracts; no device libraries or services are required."""

import importlib.util
import json
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[4]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/hisparse/config.py"
SPEC = importlib.util.spec_from_file_location("npu_hisparse_config_under_test", SOURCE)
CONFIG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONFIG
SPEC.loader.exec_module(CONFIG)


def args(role="null", npu=None, **changes):
    values = dict(
        enable_hisparse=True,
        disaggregation_mode=role,
        hisparse_config={"npu": npu or {}},
        enable_hierarchical_cache=role != "prefill",
        disable_radix_cache=role == "prefill",
        disaggregation_transfer_backend="ascend",
        kv_cache_dtype="bfloat16",
        dsa_prefill_backend=None,
        dsa_decode_backend=None,
        page_size=128,
        speculative_algorithm=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def budget(role):
    return {
        "prefill_host_kv_bytes" if role == "prefill" else "decode_host_kv_bytes": 1
        << 30
    }


def model(**changes):
    values = dict(index_topk=2048, index_kpool=4, kv_lora_rank=512, qk_rope_head_dim=0)
    values.update(changes)
    return SimpleNamespace(**values)


def validate(server, config=None, hf=None, **changes):
    CONFIG.validate_npu_hisparse(
        server,
        config or CONFIG.resolve_npu_hisparse_config(server, {}),
        hf or model(),
        is_dsa=changes.get("is_dsa", True),
        is_v4=changes.get("is_v4", False),
    )


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
@pytest.mark.parametrize("backend", ["lim_copy_sfa", "external_plan"])
def test_new_and_legacy_inputs_resolve_to_one_immutable_configuration(role, backend):
    server = args(role, {**budget(role), "dsa_offload_backend": backend})
    actual = CONFIG.resolve_npu_hisparse_config(server, {})
    legacy = args(role, enable_hisparse=False, hisparse_config=None)
    assert CONFIG.resolve_npu_hisparse_config(legacy, actual.environment()) == actual
    assert actual.role == role
    assert actual.integrated == (role == "null")
    assert actual.prefill_layerwise_offload == (role != "decode")
    assert actual.backend == backend
    with pytest.raises(FrozenInstanceError):
        actual.device_buffer_size = 1


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
def test_flag_requires_explicit_positive_byte_budget_without_allocating(role):
    environment = {}
    with pytest.raises(ValueError, match="explicit positive.*Host byte budget"):
        CONFIG.resolve_npu_hisparse_config(args(role), environment)
    assert environment == {}


def test_non_offload_npu_is_unchanged():
    server = args(enable_hisparse=False, hisparse_config=None)
    assert CONFIG.resolve_npu_hisparse_config(server, {}) is None
    assert (
        CONFIG.resolve_npu_hisparse_config(
            server, {"SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION": "1"}
        )
        is None
    )


@pytest.mark.parametrize("field", list(CONFIG.ENV_FIELDS))
def test_each_explicit_legacy_alias_accepts_equal_json_and_rejects_conflict(field):
    name, default = CONFIG.ENV_FIELDS[field]
    value = {
        "integrated": True,
        "prefill_layerwise_offload": True,
        "fused_sparse_attention": True,
        "decode_host_kv_bytes": 1 << 30,
    }.get(field, default)
    npu = {**budget("null"), field: value}
    actual = CONFIG.resolve_npu_hisparse_config(args(npu=npu), {name: str(value)})
    assert getattr(actual, field) == value
    if type(value) is bool:
        conflicting = not value
    elif type(value) in (int, float):
        conflicting = value + 1
    else:
        conflicting = value + "_different"
    with pytest.raises(ValueError, match=f"conflicts with explicit {name}"):
        CONFIG.resolve_npu_hisparse_config(args(npu=npu), {name: str(conflicting)})


def test_unset_legacy_defaults_do_not_override_json_and_export_is_explicit():
    server = args(
        npu={**budget("null"), "prefill_layer_slots": 3, "shared_host_kv": False}
    )
    server.hisparse_config.update(device_buffer_size=8192, host_to_device_ratio=999)
    environment = {}
    actual = CONFIG.resolve_npu_hisparse_config(server, environment)
    assert not environment
    actual.apply_environment(environment)
    assert environment["SGLANG_NPU_FSA_RESIDENT_CAPACITY"] == "8192"
    assert environment["SGLANG_NPU_PREFILL_LAYER_SLOTS"] == "3"
    assert environment["SGLANG_NPU_SHARED_HOST_KV"] == "False"
    assert actual.decode_host_kv_bytes == 1 << 30
    assert CONFIG.resolve_npu_hisparse_config(server, environment) == actual


def test_cli_json_string_preserves_all_budget_inputs():
    server = args(
        npu={
            **budget("null"),
            "hicache_host_tokens": 131072,
            "prefill_layer_slots": 1,
            "layer_staging_bytes": 4096,
            "prefill_layer_timeout_s": 45,
        }
    )
    original = CONFIG.resolve_npu_hisparse_config(server, {})
    server.hisparse_config = json.dumps(server.hisparse_config)
    assert CONFIG.resolve_npu_hisparse_config(server, {}) == original
    assert original.hicache_host_tokens == 131072
    assert original.prefill_layer_slots == 1


def test_top_level_resident_input_conflicts_identify_both_sources():
    server = args(npu={**budget("null"), "device_buffer_size": 8192})
    server.hisparse_config["device_buffer_size"] = 4096
    with pytest.raises(ValueError, match="device_buffer_size conflicts"):
        CONFIG.resolve_npu_hisparse_config(server, {})
    server.hisparse_config.pop("device_buffer_size")
    with pytest.raises(ValueError, match="SGLANG_NPU_FSA_RESIDENT_CAPACITY"):
        CONFIG.resolve_npu_hisparse_config(
            server, {"SGLANG_NPU_FSA_RESIDENT_CAPACITY": "4096"}
        )


def test_npu_json_does_not_silently_enable_an_omitted_flag():
    with pytest.raises(ValueError, match="requires --enable-hisparse"):
        CONFIG.resolve_npu_hisparse_config(
            args(npu=budget("null"), enable_hisparse=False), {}
        )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_roles_reject_budgets_from_the_other_service(role):
    with pytest.raises(ValueError, match="different services|Prefill Host byte budget"):
        CONFIG.resolve_npu_hisparse_config(
            args(role, {"prefill_host_kv_bytes": 4096, "decode_host_kv_bytes": 4096}),
            {},
        )


@pytest.mark.parametrize("raw", ["{", "[]", "null", "1", {"npu": []}, {"npu": None}])
def test_malformed_json_and_non_object_config_fail(raw):
    with pytest.raises(ValueError, match="hisparse_config"):
        CONFIG.resolve_npu_hisparse_config(args(hisparse_config=raw), {})


@pytest.mark.parametrize(
    "npu,match",
    [
        ({"unknown": 1}, "Unknown"),
        ({"decode_host_kv_bytes": True}, "expected int"),
        ({"decode_host_kv_bytes": -1}, "nonnegative"),
        ({"device_buffer_size": 0}, "positive"),
        ({"prefill_layer_slots": 0}, "positive"),
        ({"prefill_layer_slots": 1.5}, "expected int"),
        ({"shared_host_kv": "false"}, "expected bool"),
        ({"layer_staging_bytes": 33}, "32-byte"),
        ({"prefill_layer_timeout_s": float("nan")}, "finite"),
        ({"prefill_layer_timeout_s": 0}, "positive"),
        ({"dsa_offload_backend": "cuda"}, "external_plan"),
        ({"fused_sparse_attention_mode": "reference"}, "requires.*fused"),
        ({"integrated": False}, "role=null"),
        ({"fused_sparse_attention": False}, "role=null"),
    ],
)
def test_invalid_fields_are_rejected_before_export(npu, match):
    with pytest.raises(ValueError, match=match):
        CONFIG.resolve_npu_hisparse_config(args(npu={**budget("null"), **npu}), {})


@pytest.mark.parametrize(
    "name,value",
    [
        ("SGLANG_NPU_SHARED_HOST_KV", "maybe"),
        ("SGLANG_NPU_DECODE_HOST_KV_BYTES", "lots"),
        ("SGLANG_NPU_PREFILL_LAYER_TIMEOUT_S", "nan"),
    ],
)
def test_invalid_legacy_values_are_errors_not_silent_defaults(name, value):
    with pytest.raises(ValueError):
        CONFIG.resolve_npu_hisparse_config(args(npu=budget("null")), {name: value})


def test_reference_fsa_remains_available_outside_lim():
    validate(
        args(
            npu={
                **budget("null"),
                "dsa_offload_backend": "external_plan",
                "fused_sparse_attention_mode": "reference",
            }
        )
    )


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
def test_role_specific_cache_contracts(role):
    server = args(role, budget(role))
    validate(server)
    server.disable_radix_cache = not server.disable_radix_cache
    with pytest.raises(ValueError, match="radix"):
        validate(server)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"kv_cache_dtype": "fp8_e4m3"}, "storage"),
        ({"attention_backend": "flashmla_sparse"}, "ascend"),
        ({"dsa_decode_backend": "flashmla_sparse"}, "Ascend"),
        ({"enable_unified_cache_external_linker": True}, "Host tier"),
        ({"hicache_storage_backend": "mooncake"}, "Host tier"),
        ({"enable_hierarchical_cache": False}, "HiCache"),
        ({"enable_mixed_chunk": True}, "CP1/DCP1"),
        ({"attn_cp_size": 2}, "CP1/DCP1"),
        ({"dcp_size": 2}, "CP1/DCP1"),
        ({"page_size": 64}, "page128"),
        (
            {"speculative_algorithm": "EAGLE", "speculative_num_steps": 6},
            "six query routes",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 2,
            },
            "topk=1",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3,
                "speculative_adaptive": True,
            },
            "fixed-width",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3,
                "speculative_num_draft_tokens": 3,
            },
            "draft_tokens",
        ),
    ],
)
def test_existing_unsupported_service_combinations_still_fail(changes, match):
    with pytest.raises(ValueError, match=match):
        validate(args(npu=budget("null"), **changes))


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_fixed_chain_mtp_and_larger_resident_inputs_are_accepted(steps):
    validate(
        args(
            npu={**budget("null"), "device_buffer_size": 8192},
            speculative_algorithm="NEXTN",
            speculative_num_steps=steps,
        )
    )


def test_prefill_replicated_cp_is_not_subject_to_integrated_cp1():
    validate(args("prefill", budget("prefill"), attn_cp_size=4))
    with pytest.raises(ValueError, match="DCP=1"):
        validate(args("prefill", budget("prefill"), dcp_size=2))


def test_external_plan_retains_non_lim_model_geometry():
    server = args(
        npu={**budget("null"), "dsa_offload_backend": "external_plan"}, page_size=64
    )
    server.hisparse_config["top_k"] = 1024
    validate(server, hf=model(index_topk=1024, index_kpool=1, qk_rope_head_dim=64))


def test_topk_is_model_index_units_and_not_expanded_kpool_tokens():
    server = args(npu=budget("null"))
    server.hisparse_config["top_k"] = 2051
    with pytest.raises(ValueError, match="before KPool expansion"):
        validate(server)
    server.hisparse_config.update(top_k=2048, device_buffer_size=2048)
    with pytest.raises(ValueError, match="expanded KPool"):
        validate(server)


def test_glm53_nope_uses_effective_model_kv_geometry():
    validate(args(npu=budget("null")), hf=model(qk_rope_head_dim=64, use_mla_nope=True))


@pytest.mark.parametrize("is_dsa,is_v4", [(False, False), (False, True), (True, True)])
def test_npu_dsv4_and_non_dsa_are_not_blanket_waived(is_dsa, is_v4):
    with pytest.raises(ValueError, match="DeepSeek V4 is unsupported"):
        validate(args(npu=budget("null")), is_dsa=is_dsa, is_v4=is_v4)


@pytest.fixture
def startup(monkeypatch):
    """Load real declaration/views/env/startup code, mocking device dependencies."""
    for name, default in CONFIG.ENV_FIELDS.values():
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)
    packages = [
        "sglang",
        "sglang.srt",
        "sglang.srt.arg_groups",
        "sglang.srt.arg_groups.model_overrides",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.hardware_backend.npu.hisparse",
        "sglang.srt.model_executor",
        "sglang.srt.configs",
        "sglang.srt.utils",
    ]
    for name in packages:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, package)

    def stub(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    platform = SimpleNamespace(is_npu=True, is_hip=False)
    stub(
        "sglang.srt.runtime_context",
        get_platform=lambda: platform,
        get_exec=lambda: None,
        get_context=lambda: None,
    )
    stub("sglang.srt.platforms", current_platform=SimpleNamespace())
    stub(
        "sglang.srt.utils.common",
        is_mps=lambda: False,
        is_no_spec_infer_or_topk_one=lambda value: True,
        get_quantization_config=lambda: None,
        is_gfx95_supported=lambda: False,
        xpu_has_xmx_support=lambda: False,
    )
    stub(
        "sglang.srt.utils",
        get_npu_memory_capacity=lambda: 128 * 1024,
        is_npu=lambda: True,
        is_cuda=lambda: False,
    )
    stub("torch", Tensor=object)
    attention_arch = SimpleNamespace(MLA="mla")
    stub(
        "sglang.srt.configs.model_config",
        ModelConfig=object,
        AttentionArch=attention_arch,
        is_deepseek_dsa=lambda hf: hasattr(hf, "index_topk"),
        is_deepseek_v4=lambda hf: getattr(hf, "v4", False),
    )

    def load(name):
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "python" / (name.replace(".", "/") + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("sglang.srt.environ")
    graph = load("sglang.srt.model_executor.cuda_graph_config")
    load("sglang.srt.arg_groups.model_override_base")
    load("sglang.srt.arg_groups.arg_utils")
    overrides = load("sglang.srt.arg_groups.overrides")
    load("sglang.srt.hardware_backend.npu.hisparse.config")
    utils = load("sglang.srt.hardware_backend.npu.utils")
    hook = load("sglang.srt.arg_groups.hisparse_hook")
    return SimpleNamespace(
        utils=utils, overrides=overrides, graph=graph, hook=hook, platform=platform
    )


def startup_args(startup, role="null", **changes):
    return args(
        role,
        budget(role),
        page_size=None,
        enable_unified_cache_external_linker=False,
        hicache_storage_backend=None,
        cuda_graph_config=startup.graph.CudaGraphConfig(),
        _model_config=SimpleNamespace(hf_config=model(), attention_arch="mla"),
        **changes,
    )


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
@pytest.mark.parametrize("legacy", [False, True])
def test_real_startup_normalizes_before_hicache_and_declares_same_route(
    startup, monkeypatch, role, legacy
):
    server = startup_args(startup, role)
    if legacy:
        environment = CONFIG.resolve_npu_hisparse_config(server, {}).environment()
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        server.enable_hisparse = False
        server.hisparse_config = None
    startup.utils.set_default_server_args(server)
    view = startup.overrides.resolving_view(server)
    assert view.enable_hisparse
    assert view.disable_radix_cache == (role == "prefill")
    assert view.page_size == 128
    if role != "prefill":
        assert view.enable_hierarchical_cache
        assert view.hicache_io_backend == "kernel_ascend"
        assert view.hicache_host_memory_mode == "cache"
        assert view.hicache_write_policy == "write_through"
        assert view.hicache_mem_layout == "page_first_kv_split"
    startup.hook.validate_hisparse(server)
    assert startup.hook._hisparse_default_backend("bfloat16") == "ascend"


def test_real_startup_fails_before_declarations_and_environment_mutation(startup):
    server = startup_args(startup)
    server.hisparse_config = None
    with pytest.raises(ValueError, match="explicit positive"):
        startup.utils.set_default_server_args(server)
    assert not getattr(server, "_resolved_overrides", None)
    assert "SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD" not in os.environ


def test_native_cuda_radix_and_dtype_checks_remain_intact(startup):
    startup.platform.is_npu = False
    server = startup_args(startup)
    assert startup.hook._hisparse_default_backend("bfloat16") == "flashmla_sparse"
    assert startup.hook._hisparse_default_backend("fp8_e4m3") == "flashmla_kv"
    with pytest.raises(AssertionError, match="disable-radix-cache"):
        startup.hook.validate_hisparse(server)
    server.disable_radix_cache = True
    startup.hook.validate_hisparse(server)
    server.kv_cache_dtype = "float16"
    with pytest.raises(ValueError, match="KV cache dtypes"):
        startup.hook.validate_hisparse(server)
    startup.platform.is_hip = True
    assert startup.hook._hisparse_default_backend("bfloat16") == "tilelang"
