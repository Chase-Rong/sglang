"""NPU HiSparse configuration interface (stdlib only).

Startup calls resolve_npu_hisparse_config(resolving_view(args)), then
config.apply_environment() before declaring enable_hisparse=True and setting up
HiCache. Runtime consumers can resolve the same immutable object from a view
with enable_hisparse, hisparse_config and disaggregation_mode attributes.

JSON fields live in hisparse_config.npu and use the names in ENV_FIELDS below.
Top-level device_buffer_size aliases the existing FSA resident-capacity input;
top_k remains the model's index_topk, not the expanded KPool token count.
Host byte budgets are required explicitly. Existing budget/kernel code resolves
actual MTP/page-aligned capacity; host_to_device_ratio never sizes NPU backing.
No device imports, allocations, or environment mutations occur during parsing.
"""

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

ENV_FIELDS = {
    "dsa_offload_backend": ("SGLANG_NPU_DSA_OFFLOAD_BACKEND", "lim_copy_sfa"),
    "device_buffer_size": ("SGLANG_NPU_FSA_RESIDENT_CAPACITY", 4096),
    "shared_host_kv": ("SGLANG_NPU_SHARED_HOST_KV", True),
    "integrated": ("SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD", False),
    "layer_staging_bytes": ("SGLANG_NPU_LAYER_STAGING_BYTES", 0),
    "prefill_layer_timeout_s": ("SGLANG_NPU_PREFILL_LAYER_TIMEOUT_S", 30.0),
    "prefill_layerwise_offload": ("SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD", False),
    "prefill_layer_slots": ("SGLANG_NPU_PREFILL_LAYER_SLOTS", 2),
    "prefill_host_kv_bytes": ("SGLANG_NPU_PREFILL_HOST_KV_BYTES", 0),
    "decode_host_kv_bytes": ("SGLANG_NPU_DECODE_HOST_KV_BYTES", 0),
    "hicache_host_tokens": ("SGLANG_NPU_HICACHE_HOST_TOKENS", 0),
    "fused_sparse_attention": ("SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION", False),
    "fused_sparse_attention_mode": ("SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE", "fused"),
}


@dataclass(frozen=True)
class NPUHiSparseConfig:
    role: str
    top_k: int | None
    dsa_offload_backend: str
    device_buffer_size: int
    shared_host_kv: bool
    integrated: bool
    layer_staging_bytes: int
    prefill_layer_timeout_s: float
    prefill_layerwise_offload: bool
    prefill_layer_slots: int
    prefill_host_kv_bytes: int
    decode_host_kv_bytes: int
    hicache_host_tokens: int
    fused_sparse_attention: bool
    fused_sparse_attention_mode: str

    @property
    def backend(self):
        return self.dsa_offload_backend

    def environment(self):
        """Return the compatibility inputs consumed by existing kernels/budgets."""
        return {
            name: str(getattr(self, field))
            for field, (name, default) in ENV_FIELDS.items()
        }

    def apply_environment(self, environ=None):
        """Publish normalized inputs before constructing any offload resources."""
        destination = os.environ if environ is None else environ
        destination.update(self.environment())


def _parse_value(value, default, source, *, environment=False):
    if environment:
        try:
            if type(default) is bool:
                if value.lower() in ("true", "1", "yes", "y"):
                    return True
                if value.lower() in ("false", "0", "no", "n"):
                    return False
                raise ValueError("expected a boolean")
            return type(default)(value)
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"Invalid {source}: {value!r}") from error
    valid = type(value) is type(default)
    if type(default) is float:
        valid = type(value) in (int, float)
    if not valid:
        raise ValueError(f"Invalid {source}: expected {type(default).__name__}")
    return value


def resolve_npu_hisparse_config(server_args, environ=None):
    """Resolve new JSON and explicit legacy aliases, rejecting conflicting inputs.

    Returns None for non-offload NPU configurations. Call only on NPU: platform
    selection belongs to the startup/factory caller, not this pure parser.
    """
    environment = os.environ if environ is None else environ
    raw = getattr(server_args, "hisparse_config", None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as error:
            raise ValueError(f"Invalid hisparse_config JSON: {error}") from error
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("hisparse_config must be a JSON object")
    npu = raw.get("npu", {})
    if not isinstance(npu, Mapping):
        raise ValueError("hisparse_config.npu must be an object")
    values = {}
    for field, (name, default) in ENV_FIELDS.items():
        values[field] = (
            _parse_value(environment[name], default, name, environment=True)
            if name in environment
            else default
        )
    enabled = getattr(server_args, "enable_hisparse", False)
    legacy = (
        values["integrated"]
        or values["prefill_layerwise_offload"]
        or values["prefill_host_kv_bytes"] > 0
        or values["decode_host_kv_bytes"] > 0
    )
    if not enabled and not legacy:
        if npu:
            raise ValueError("hisparse_config.npu requires --enable-hisparse")
        return None
    unknown = sorted(set(npu) - set(ENV_FIELDS))
    if unknown:
        raise ValueError(f"Unknown hisparse_config.npu fields: {unknown}")
    supplied = dict(npu)
    if "device_buffer_size" in raw:
        if (
            "device_buffer_size" in supplied
            and supplied["device_buffer_size"] != raw["device_buffer_size"]
        ):
            raise ValueError(
                "hisparse_config.device_buffer_size conflicts with "
                "hisparse_config.npu.device_buffer_size"
            )
        supplied["device_buffer_size"] = raw["device_buffer_size"]
    for field, value in supplied.items():
        name, default = ENV_FIELDS[field]
        source = f"hisparse_config.npu.{field}"
        value = _parse_value(value, default, source)
        if name in environment and value != values[field]:
            raise ValueError(f"{source} conflicts with explicit {name}")
        values[field] = value
    role = getattr(server_args, "disaggregation_mode", "null")
    if role not in ("null", "prefill", "decode"):
        raise ValueError(f"Unsupported NPU HiSparse service role: {role}")
    for field, required in (
        ("integrated", role == "null"),
        ("prefill_layerwise_offload", role != "decode"),
        ("fused_sparse_attention", role != "prefill"),
    ):
        name, default = ENV_FIELDS[field]
        if (field in supplied or name in environment) and values[field] != required:
            raise ValueError(
                f"hisparse_config.npu.{field} / {name} conflicts with "
                f"NPU HiSparse role={role} (requires {required})"
            )
        values[field] = required
    for field in (
        "device_buffer_size",
        "prefill_layer_slots",
        "prefill_host_kv_bytes",
        "decode_host_kv_bytes",
        "hicache_host_tokens",
        "layer_staging_bytes",
    ):
        if values[field] < 0:
            raise ValueError(f"NPU HiSparse {field} must be nonnegative")
    if values["device_buffer_size"] == 0 or values["prefill_layer_slots"] == 0:
        raise ValueError(
            "NPU HiSparse requires positive device_buffer_size and prefill_layer_slots"
        )
    if values["layer_staging_bytes"] % 32:
        raise ValueError("NPU HiSparse layer_staging_bytes must be 32-byte aligned")
    if (
        not math.isfinite(values["prefill_layer_timeout_s"])
        or values["prefill_layer_timeout_s"] <= 0
    ):
        raise ValueError(
            "NPU HiSparse prefill_layer_timeout_s must be finite and positive"
        )
    if values["dsa_offload_backend"] not in ("lim_copy_sfa", "external_plan"):
        raise ValueError(
            "NPU HiSparse dsa_offload_backend must be lim_copy_sfa or external_plan"
        )
    if values["fused_sparse_attention_mode"] not in ("fused", "reference"):
        raise ValueError(
            "NPU HiSparse fused_sparse_attention_mode must be fused or reference"
        )
    if (
        role != "prefill"
        and values["dsa_offload_backend"] == "lim_copy_sfa"
        and values["fused_sparse_attention_mode"] != "fused"
    ):
        raise ValueError("lim_copy_sfa requires fused_sparse_attention_mode=fused")
    budget = "prefill_host_kv_bytes" if role == "prefill" else "decode_host_kv_bytes"
    if values[budget] <= 0:
        raise ValueError(
            f"NPU HiSparse role={role} requires an explicit positive "
            f"hisparse_config.npu.{budget} or {ENV_FIELDS[budget][0]}; "
            "no Host byte budget is allocated by default"
        )
    if role == "prefill" and values["decode_host_kv_bytes"]:
        raise ValueError(
            "Prefill and Decode Host byte budgets belong to different services"
        )
    if role == "decode" and values["prefill_host_kv_bytes"]:
        raise ValueError("Decode cannot use a Prefill Host byte budget")
    top_k = raw.get("top_k")
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        raise ValueError("hisparse_config.top_k must be a positive integer")
    if top_k is not None and values["device_buffer_size"] < top_k:
        raise ValueError("device_buffer_size must cover top_k")
    return NPUHiSparseConfig(role=role, top_k=top_k, **values)


def validate_lim_dtypes(kv_dtype, query_dtype=None):
    for name, dtype in (("KV", kv_dtype), ("query", query_dtype)):
        if dtype is not None and str(dtype) not in (
            "auto",
            "bfloat16",
            "torch.bfloat16",
        ):
            raise ValueError(f"lim_copy_sfa requires BF16 {name} storage, got {dtype}")


def validate_npu_hisparse(server_args, config, hf_config, *, is_dsa, is_v4):
    """Validate NPU-specific model/service contracts without importing devices."""
    if is_v4 or not is_dsa:
        raise ValueError(
            "NPU HiSparse supports existing DSA offload models; DeepSeek V4 is unsupported"
        )
    dtype = getattr(server_args, "kv_cache_dtype", "auto")
    if dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("NPU HiSparse Host KV requires float16 or bfloat16 storage")
    for attr in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    ):
        if getattr(server_args, attr, None) not in (None, "ascend"):
            raise ValueError(f"NPU HiSparse requires {attr}=ascend")
    for attr in ("dsa_prefill_backend", "dsa_decode_backend"):
        if getattr(server_args, attr, None) not in (None, "ascend"):
            raise ValueError(
                f"NPU HiSparse uses Ascend attention, not {attr}={getattr(server_args, attr)}"
            )
    if config.role == "prefill":
        if not getattr(server_args, "disable_radix_cache", False):
            raise ValueError("NPU HiSparse Prefill requires disabled radix cache")
        if getattr(server_args, "optimistic_prefill_attempts", 0):
            raise ValueError(
                "NPU HiSparse Prefill requires disabled optimistic Prefill"
            )
    else:
        if getattr(server_args, "disable_radix_cache", False) or not getattr(
            server_args, "enable_hierarchical_cache", False
        ):
            raise ValueError(
                "NPU HiSparse integrated/Decode requires HiCache and enabled radix cache"
            )
        if (
            getattr(server_args, "enable_unified_cache_external_linker", False)
            or getattr(server_args, "hicache_storage_backend", None) is not None
        ):
            raise ValueError(
                "NPU HiSparse requires the HiCache Host tier without external linker/Store"
            )
    if (
        config.role != "null"
        and getattr(server_args, "disaggregation_transfer_backend", None) != "ascend"
    ):
        raise ValueError("NPU HiSparse PD requires the Ascend MemFabric backend")
    cp = getattr(server_args, "attn_cp_size", 1)
    dcp = getattr(server_args, "dcp_size", getattr(server_args, "attn_dcp_size", 1))
    if config.integrated and (
        getattr(server_args, "enable_mixed_chunk", False) or cp != 1 or dcp != 1
    ):
        raise ValueError(
            "Integrated NPU HiSparse requires separate Prefill batches and CP1/DCP1"
        )
    if config.shared_host_kv and dcp != 1:
        raise ValueError("shared MLA Host backing requires replicated KV with DCP=1")
    algorithm = getattr(server_args, "speculative_algorithm", None)
    steps = getattr(server_args, "speculative_num_steps", None)
    if algorithm is not None and (
        algorithm not in ("EAGLE", "NEXTN")
        or getattr(server_args, "speculative_eagle_topk", 1) != 1
        or type(steps) is not int
        or steps < 1
        or getattr(server_args, "speculative_num_draft_tokens", None)
        not in (None, steps + 1)
        or getattr(server_args, "speculative_adaptive", False)
        or getattr(server_args, "enable_multi_layer_eagle", False)
        or getattr(server_args, "speculative_draft_window_size", None) is not None
    ):
        raise ValueError(
            "NPU HiSparse requires fixed-width NEXTN/EAGLE, topk=1, draft_tokens=steps+1 and disabled adaptive/windowed drafting"
        )
    text_config = getattr(hf_config, "text_config", hf_config)

    def model_value(name, default=None):
        return (
            text_config.get(name, default)
            if isinstance(text_config, Mapping)
            else getattr(text_config, name, default)
        )

    model_topk = model_value("index_topk")
    if config.top_k is not None and config.top_k != model_topk:
        raise ValueError(
            "hisparse_config.top_k must equal model index_topk (before KPool expansion)"
        )
    if config.role == "prefill":
        return
    expanded = int(model_topk) + int(model_value("index_kpool", 1)) - 1
    if config.device_buffer_size < expanded:
        raise ValueError(
            "NPU HiSparse device_buffer_size must cover expanded KPool TopK and tail"
        )
    if config.dsa_offload_backend == "lim_copy_sfa":
        query_dtype = getattr(server_args, "dtype", "auto")
        if query_dtype == "auto":
            query_dtype = model_value("torch_dtype", model_value("dtype"))
        validate_lim_dtypes(dtype, query_dtype)
    if config.dsa_offload_backend == "lim_copy_sfa" and (
        model_topk != 2048
        or model_value("index_kpool", 1) != 4
        or model_value("kv_lora_rank") != 512
        or (
            0 if model_value("use_mla_nope", False) else model_value("qk_rope_head_dim")
        )
        != 0
        or getattr(server_args, "page_size", 128) != 128
        or (steps + 1 if algorithm else 1) > 6
    ):
        raise ValueError(
            "lim_copy_sfa requires TopK2048, KPool4, NoPE512, page128 and one to six query routes"
        )
