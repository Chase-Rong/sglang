"""Admission contracts for the separate Ascend sparse offload lifecycle."""


def validate_offload_service(server_args, role, *, integrated=False):
    if integrated and role != "null":
        raise ValueError("integrated offload requires one non-disaggregated service")
    if not integrated and role not in ("prefill", "decode"):
        raise ValueError(
            "NPU sparse offload requires a separate Prefill or Decode service"
        )
    if hasattr(server_args, "get_attention_backends"):
        backends = server_args.get_attention_backends()
    else:
        from sglang.srt.runtime_context import attention_backends

        backends = attention_backends()
    if tuple(backends) != ("ascend", "ascend"):
        raise ValueError(
            "NPU sparse offload requires Ascend for both attention backends"
        )
    if (
        (
            not integrated
            and server_args.disaggregation_transfer_backend
            not in ("ascend", "mooncake")
        )
        or not server_args.disable_radix_cache
        or server_args.enable_hisparse
    ):
        raise ValueError(
            "NPU sparse offload requires Ascend/Mooncake transport, disabled radix cache and no HiSparse coordinator"
        )
    if role == "prefill" and server_args.optimistic_prefill_attempts:
        raise ValueError(
            "bounded Prefill must finish layer bootstrap before forward; disable optimistic Prefill"
        )


def validate_mtp_pool_pair(target_pool, draft_pool):
    for name in (
        "page_size",
        "size",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "index_head_dim",
        "store_dtype",
    ):
        if getattr(target_pool, name) != getattr(draft_pool, name):
            raise ValueError(f"offload target/draft pools require matching {name}")
    target_runtime = getattr(target_pool, "prefill_offload_runtime", None)
    draft_runtime = getattr(draft_pool, "prefill_offload_runtime", None)
    if target_runtime is not None and draft_runtime is not None:
        if target_runtime.request_pool is not draft_runtime.request_pool:
            raise ValueError(
                "offload target/draft pools must share request page ownership"
            )


def resolve_transfer_device(logical_device, physical_override, visible_devices):
    if type(logical_device) is not int or logical_device < 0 or physical_override < -1:
        raise ValueError("invalid Ascend transfer device ID")
    if visible_devices is None:
        return logical_device if physical_override == -1 else physical_override
    entries = [entry.strip() for entry in visible_devices.split(",")]
    if (
        not entries
        or any(not entry.isdecimal() for entry in entries)
        or len(set(map(int, entries))) != len(entries)
    ):
        raise ValueError(
            "offload requires a distinct numeric ASCEND_RT_VISIBLE_DEVICES list"
        )
    if logical_device >= len(entries):
        raise ValueError("Ascend transfer device is outside the visible device list")
    physical_device = int(entries[logical_device])
    if physical_override != -1 and physical_override != physical_device:
        raise ValueError(
            "ASCEND_NPU_PHY_ID conflicts with this rank's visible device mapping"
        )
    return physical_device


def fsa_pool_options(kvc, layer_count):
    import torch

    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.shared_host_kv import (
        shared_host_group_from_runtime,
    )
    from sglang.srt.runtime_context import get_disagg

    prefill = envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.get()
    host_budget = envs.SGLANG_NPU_DECODE_HOST_KV_BYTES.get()
    integrated = envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get()
    if not prefill and not host_budget and not integrated:
        return {}
    role = get_disagg().disaggregation_mode
    if str(kvc.device).split(":")[0] != "npu" or not kvc.use_mla_backend:
        raise ValueError("FSA Host offload requires an NPU MLA pool")
    validate_offload_service(kvc.server_args, role, integrated=integrated)
    if kvc.kv_cache_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FSA Host KV requires FP16 or BF16 storage")
    native_ops()
    if integrated:
        if (
            not prefill
            or host_budget <= 0
            or not envs.SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION.get()
        ):
            raise ValueError(
                "integrated offload requires Prefill slots, Host budget and fused attention"
            )
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().attn_cp_size != 1 or get_parallel().attn_dcp_size != 1:
            raise ValueError("integrated offload requires CP=1 and DCP=1")
        slots = min(envs.SGLANG_NPU_PREFILL_LAYER_SLOTS.get(), layer_count)
        if slots <= 0:
            raise ValueError("integrated offload requires positive Prefill layer slots")
        return {
            "prefill_layer_slots": slots,
            "decode_host_budget": host_budget,
            "shared_host_group": shared_host_group_from_runtime(
                torch.npu.current_device()
            ),
            "integrated_offload": True,
        }
    if prefill:
        if role != "prefill" or host_budget:
            raise ValueError(
                "Prefill layer slots and Decode Host budget belong to different services"
            )
        slots = min(envs.SGLANG_NPU_PREFILL_LAYER_SLOTS.get(), layer_count)
        if slots <= 0 or envs.SGLANG_NPU_PREFILL_HOST_KV_BYTES.get() <= 0:
            raise ValueError("Prefill offload requires positive slot and Host budgets")
        return {"prefill_layer_slots": slots}
    if role != "decode" or not envs.SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION.get():
        raise ValueError("Decode Host backing requires the fused/reference FSA path")
    if envs.SGLANG_NPU_LAYER_STAGING_BYTES.get() <= 0:
        raise ValueError("Decode Host receive requires a bounded NPU staging window")
    return {
        "decode_host_budget": host_budget,
        "shared_host_group": shared_host_group_from_runtime(torch.npu.current_device()),
    }
