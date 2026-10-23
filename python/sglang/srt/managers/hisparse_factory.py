"""Device dispatch for HiSparse without importing native GPU kernels on NPU."""

from sglang.srt.runtime_context import get_memory, get_parallel, get_platform


def uses_native_hisparse() -> bool:
    return get_memory().enable_hisparse and not get_platform().is_npu


def uses_hisparse_staging(coordinator) -> bool:
    """Native coordinators use a ready queue; NPU keeps ordinary batch merge."""
    return coordinator is not None and getattr(coordinator, "uses_staging", True)


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
        tp_group=(
            model_runner.attention_tp_group.cpu_group
            if get_parallel().enable_dp_attention
            else model_runner.tp_group.cpu_group
        ),
        host_to_device_ratio=config.host_to_device_ratio,
        swap_in_block_size=config.swap_in_block_size,
        shared_index_layers=resolve_shared_index_layers(
            hf_text_config=model_runner.model_config.hf_text_config,
            pp_size=model_runner.ps.pp_size,
            is_speculative=model_runner.spec_algorithm.is_speculative(),
        ),
    )
