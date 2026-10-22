"""Admission contracts for the separate Ascend sparse offload lifecycle."""

from contextlib import contextmanager


@contextmanager
def offload_host_memory_budget_scope(device_pool):
    """Wrap one complete sidecar build using the preallocation Host snapshot."""
    from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope

    pool = getattr(device_pool, "full_kv_pool", device_pool)
    backing = getattr(pool, "decode_host_backing", None)
    account = getattr(backing, "host_memory_account", None)
    if account is not None:
        from sglang.srt.mem_cache.pool_host.base import ensure_host_memory_budget_scope

        with ensure_host_memory_budget_scope(account):
            yield
        return
    plan = getattr(pool, "host_memory_budget_plan", None)
    if plan is None:
        yield
        return
    plan.validate()
    with host_memory_budget_scope(plan.sidecar_budget_bytes):
        yield


@contextmanager
def offload_host_backing_budget_scope(plan):
    if plan is None:
        yield
        return
    from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope

    plan.validate()
    with host_memory_budget_scope(plan.scope_available_bytes // plan.shared_ranks):
        yield


def bind_host_memory_budget_plan(backing, plan):
    backing.host_memory_budget_plan = plan
    if plan is not None:
        try:
            backing.host_memory_account.charge(
                "offload_planner", plan.planner_bytes_per_rank
            )
        except BaseException:
            backing.pool.close(drained=True)
            raise


def layer_staging_bytes(configured, *, role, host_offload, integrated=False):
    if type(configured) is not int or configured < 0 or configured % 32:
        raise ValueError(
            "NPU layer staging bytes must be nonnegative and 32-byte aligned"
        )
    if integrated or role != "decode" or not host_offload:
        return 0
    return configured or 64 * 1024 * 1024


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
    if integrated:
        if server_args.disable_radix_cache or not getattr(
            server_args, "enable_hierarchical_cache", False
        ):
            raise ValueError(
                "Integrated offload requires HiCache and enabled radix cache"
            )
    if (
        not integrated
        and server_args.disaggregation_transfer_backend not in ("ascend", "mooncake")
    ) or (not integrated and role == "prefill" and not server_args.disable_radix_cache):
        raise ValueError(
            "NPU sparse offload requires Ascend/Mooncake transport and disabled Prefill radix cache"
        )
    if not integrated and server_args.disaggregation_transfer_backend != "ascend":
        raise ValueError("NPU PD offload requires the Ascend MemFabric backend")
    if (
        role == "decode"
        and not integrated
        and (
            server_args.disable_radix_cache
            or not getattr(server_args, "enable_hierarchical_cache", False)
        )
    ):
        raise ValueError("PD offload Decode requires HiCache and enabled radix cache")
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
    for name in ("index_size", "index_page_size", "index_kpool", "_kpool_use_compress"):
        if getattr(target_pool, name, None) != getattr(draft_pool, name, None):
            raise ValueError(f"offload target/draft pools require matching {name}")
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


def validate_integrated_layout(*, integrated, prefill, mixed_chunk, cp_size, dcp_size):
    if not integrated or not prefill:
        raise ValueError("Integrated offload requires layerwise Prefill")
    if mixed_chunk or cp_size != 1 or dcp_size != 1:
        raise ValueError(
            "Integrated offload requires separate Prefill batches and CP1/DCP1"
        )


def validate_prefill_overlap(
    *, integrated, prefill, speculative, mixed_chunk, cp_size, dcp_size
):
    try:
        validate_integrated_layout(
            integrated=integrated,
            prefill=prefill,
            mixed_chunk=mixed_chunk,
            cp_size=cp_size,
            dcp_size=dcp_size,
        )
    except ValueError as error:
        raise ValueError(f"Prefill copy overlap: {error}") from error
    # MTP graph replay enters the Prefill runtime context before consuming KV;
    # its Host-visibility event orders asynchronous copies on the replay stream.


def host_memory_scope_identity():
    from pathlib import Path

    root = Path("/proc")
    membership = tuple(
        sorted(
            line
            for line in (root / "self/cgroup").read_text().splitlines()
            if not line.split(":", 2)[1] or "memory" in line.split(":", 2)[1].split(",")
        )
    )
    mounts = []
    for line in (root / "self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields = after.split()
        if fields[0] == "cgroup2" or (
            fields[0] == "cgroup" and "memory" in fields[2].split(",")
        ):
            mounts.append((fields[0], before.split()[3]))
    namespace = (root / "self/ns/cgroup").stat()
    return (
        (root / "sys/kernel/random/boot_id").read_text().strip(),
        namespace.st_dev,
        namespace.st_ino,
        membership,
        tuple(sorted(mounts)),
    )


def prepare_host_memory_budget_plan(reserve_bytes, *, alignment_bytes=1 << 30):
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        HostMemoryBudgetPlan,
    )
    from sglang.srt.mem_cache.pool_host.base import (
        host_memory_budget_bytes,
        ranks_per_host,
    )

    group = None
    if envs.SGLANG_NPU_SHARED_HOST_KV.get():
        from sglang.srt.distributed.parallel_state import get_attn_tp_group

        group = get_attn_tp_group()
    shared_ranks = group.world_size if group is not None else 1
    try:
        local_ranks = ranks_per_host()
        if shared_ranks > local_ranks or local_ranks % shared_ranks:
            raise ValueError("Shared Host ranks must partition the local launch ranks")
        identity = host_memory_scope_identity() if shared_ranks > 1 else ()
        available = max(0, host_memory_budget_bytes(auto_size=True))
        report = (
            (identity, reserve_bytes, alignment_bytes, local_ranks),
            available,
            None,
        )
    except Exception as error:
        report = ((), 0, str(error))
    reports = group.all_gather_object(report) if shared_ranks > 1 else [report]
    if any(report[2] is not None for report in reports):
        raise RuntimeError(f"Host memory admission failed across ranks: {reports}")
    if any(report[0] != reports[0][0] for report in reports):
        raise ValueError(
            "Shared Host ranks disagree on the host/cgroup scope, domain reserve or rank placement"
        )
    plan = HostMemoryBudgetPlan(
        reserve_bytes,
        min(report[1] for report in reports) * shared_ranks,
        shared_ranks=shared_ranks,
        scope_identity=reports[0][0][0],
        reserve_alignment_bytes=alignment_bytes,
    )
    plan.validate()
    return plan


def _pool_host_memory_plan(
    kvc, reserve_bytes, *, alignment_bytes=1 << 30, required_bytes=None
):
    plan = getattr(kvc, "_offload_host_memory_plan", None)
    if plan is None:
        plan = getattr(
            getattr(kvc, "memory_pool_config", None), "offload_host_memory_plan", None
        )
    if plan is None and getattr(kvc, "is_draft_worker", False):
        allocator = kvc.token_to_kv_pool_allocator
        pool = allocator.get_kvcache()
        pool = getattr(pool, "full_kv_pool", pool)
        plan = getattr(pool, "host_memory_budget_plan", None)
        if plan is None:
            raise RuntimeError(
                "Draft Host allocation requires the target's Host memory admission"
            )
    if plan is None:
        claim = reserve_bytes if required_bytes is None else required_bytes
        if claim > reserve_bytes:
            raise MemoryError("Prefill Host payload exceeds its configured limit")
        plan = prepare_host_memory_budget_plan(claim, alignment_bytes=alignment_bytes)
    if (
        plan.domain_reserve_bytes > reserve_bytes
        or plan.reserve_alignment_bytes != alignment_bytes
        or (alignment_bytes > 1 and plan.domain_reserve_bytes != reserve_bytes)
    ):
        raise ValueError("Target/draft Host memory reservations must match")
    kvc._offload_host_memory_plan = plan
    return plan


def fsa_pool_options(kvc, layer_count, *, token_capacity=None):
    import torch

    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg

    prefill = envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.get()
    host_budget = envs.SGLANG_NPU_DECODE_HOST_KV_BYTES.get()
    integrated = envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get()
    if integrated:
        from sglang.srt.runtime_context import get_parallel, get_schedule

        validate_integrated_layout(
            integrated=integrated,
            prefill=prefill,
            mixed_chunk=get_schedule().enable_mixed_chunk,
            cp_size=get_parallel().attn_cp_size,
            dcp_size=get_parallel().attn_dcp_size,
        )
    if not prefill and not host_budget and not integrated:
        return {}
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.shared_host_kv import (
        shared_host_group_from_runtime,
    )

    role = get_disagg().disaggregation_mode
    if str(kvc.device).split(":")[0] != "npu" or not kvc.use_mla_backend:
        raise ValueError("FSA Host offload requires an NPU MLA pool")
    validate_offload_service(
        resolved_view(kvc.server_args), role, integrated=integrated
    )
    if kvc.kv_cache_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FSA Host KV requires FP16 or BF16 storage")
    if (
        role != "prefill"
        and envs.SGLANG_NPU_DSA_OFFLOAD_BACKEND.get() == "lim_copy_sfa"
    ):
        from sglang.srt.hardware_backend.npu.hisparse.config import validate_lim_dtypes

        validate_lim_dtypes(kvc.kv_cache_dtype, kvc.model_dtype)
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
        options = {
            "prefill_layer_slots": slots,
            "decode_host_budget": host_budget,
            "shared_host_group": shared_host_group_from_runtime(
                torch.npu.current_device()
            ),
            "integrated_offload": True,
            "host_memory_budget_plan": _pool_host_memory_plan(kvc, host_budget),
        }
        host_tokens = envs.SGLANG_NPU_HICACHE_HOST_TOKENS.get()
        if host_tokens:
            options["decode_host_token_capacity"] = host_tokens
        return options
    if prefill:
        if role != "prefill" or host_budget:
            raise ValueError(
                "Prefill layer slots and Decode Host budget belong to different services"
            )
        slots = min(envs.SGLANG_NPU_PREFILL_LAYER_SLOTS.get(), layer_count)
        if slots <= 0 or envs.SGLANG_NPU_PREFILL_HOST_KV_BYTES.get() <= 0:
            raise ValueError("Prefill offload requires positive slot and Host budgets")
        required = None
        if token_capacity is not None and not kvc.is_draft_worker:
            from sglang.srt.configs.model_config import (
                get_dsa_index_kpool,
                get_dsa_index_kpool_compress,
            )
            from sglang.srt.hardware_backend.npu.hisparse.speculative_sparse import (
                sparse_mtp_draft_layers,
            )
            from sglang.srt.runtime_context import get_schedule

            model = kvc.model_config
            pooled = get_dsa_index_kpool(
                model.hf_config
            ) > 1 and get_dsa_index_kpool_compress(model.hf_config)
            required = (
                (token_capacity + get_schedule().page_size)
                * (layer_count + sparse_mtp_draft_layers(kvc))
                * (
                    model.kv_lora_rank
                    + model.qk_rope_head_dim
                    + (0 if pooled else model.index_head_dim)
                )
                * torch._utils._element_size(kvc.kv_cache_dtype)
            )
        return {
            "prefill_layer_slots": slots,
            "host_memory_budget_plan": _pool_host_memory_plan(
                kvc,
                envs.SGLANG_NPU_PREFILL_HOST_KV_BYTES.get(),
                alignment_bytes=1,
                required_bytes=required,
            ),
        }
    if role != "decode" or not envs.SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION.get():
        raise ValueError("Decode Host backing requires the fused/reference FSA path")
    host_tokens = envs.SGLANG_NPU_HICACHE_HOST_TOKENS.get()
    return {
        "decode_host_token_capacity": host_tokens,
        "decode_host_budget": host_budget,
        "shared_host_group": shared_host_group_from_runtime(torch.npu.current_device()),
        "host_memory_budget_plan": _pool_host_memory_plan(kvc, host_budget),
    }
