"""Device dispatch for HiSparse without importing native GPU kernels on NPU."""

from sglang.srt.runtime_context import get_memory, get_platform, get_spec


def uses_native_hisparse(enabled: bool | None = None) -> bool:
    if enabled is None:
        enabled = get_memory().enable_hisparse
    return enabled and not get_platform().is_npu


def uses_hisparse_staging(coordinator) -> bool:
    """Native coordinators use a ready queue; NPU keeps ordinary batch merge."""
    return coordinator is not None and getattr(coordinator, "uses_staging", True)


def has_pending_hisparse_transfers(scheduler) -> bool:
    if uses_hisparse_staging(scheduler.hisparse_coordinator):
        return False
    for name in ("disagg_prefill_bootstrap_queue", "disagg_decode_prealloc_queue"):
        manager = getattr(getattr(scheduler, name, None), "kv_manager", None)
        pending = getattr(manager, "has_pending_layer_transfers", None)
        if pending is not None and pending():
            return True
    return False


def release_hisparse_resources(scheduler):
    coordinator = scheduler.hisparse_coordinator
    if coordinator is not None and not uses_hisparse_staging(coordinator):
        from sglang.srt.hardware_backend.npu.hisparse.shutdown import (
            release_npu_hisparse_resources,
        )

        release_npu_hisparse_resources(scheduler)
        return
    if coordinator is not None:
        coordinator.destroy()
    scheduler.tree_cache.release_host_resources()


def create_hisparse_coordinator(model_runner):
    if not get_memory().enable_hisparse:
        return None
    if get_platform().is_npu:
        from sglang.srt.hardware_backend.npu.hisparse.runtime import (
            NpuHiSparseCoordinator,
        )

        return NpuHiSparseCoordinator(model_runner)

    from sglang.srt.distributed.parallel_state import get_dp_tp_group
    from sglang.srt.managers.hisparse_coordinator import (
        HiSparseCoordinator,
        resolve_shared_index_layers,
    )
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config

    config = parse_hisparse_config()
    return HiSparseCoordinator(
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        top_k=getattr(
            model_runner.model_config.hf_text_config, "index_topk", config.top_k
        ),
        device_buffer_size=config.device_buffer_size,
        device=model_runner.device,
        tp_group=get_dp_tp_group().cpu_group,
        host_to_device_ratio=config.host_to_device_ratio,
        swap_in_block_size=config.swap_in_block_size,
        shared_index_layers=resolve_shared_index_layers(
            hf_text_config=model_runner.model_config.hf_text_config,
        ),
        num_draft_tokens=(
            0
            if model_runner.spec_algorithm.is_none()
            else get_spec().speculative_num_draft_tokens or 0
        ),
    )
