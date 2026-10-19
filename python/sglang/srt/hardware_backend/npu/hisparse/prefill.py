"""Construct existing layerwise runtimes for the NPU HiSparse coordinator."""

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel, get_spec


def initialize_prefill_runtime(runner):
    if not envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.get():
        return
    if envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get():
        from sglang.srt.hardware_backend.npu.hisparse.prefill_overlap import (
            PipelinedIntegratedOffloadRuntime,
        )

        pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
        # Eager forwards and direct MTP graph replay both enter the runtime
        # context, which waits for the preceding Prefill Host publication.
        runtime = PipelinedIntegratedOffloadRuntime(pool, runner.req_to_token_pool)
        runner.prefill_offload_runtime = runtime
        runner.token_to_kv_pool.prefill_offload_runtime = runtime
        return
    if runner.device != "npu":
        raise RuntimeError("SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD requires NPU")
    if get_disagg().disaggregation_mode != "prefill":
        raise RuntimeError("layerwise Prefill offload must run on the Prefill service")
    from sglang.srt.hardware_backend.npu.hisparse.speculative_sparse import (
        sparse_mtp_tail,
    )

    sparse_mtp_tail(runner.spec_algorithm, get_spec(), runner.model_config)
    from sglang.srt.hardware_backend.npu.hisparse.prefill_offload_runtime import (
        validate_prefill_parallel_layout,
    )

    validate_prefill_parallel_layout(
        cp_size=runner.ps.attn_cp_size,
        dcp_size=runner.ps.attn_dcp_size,
        replicated_cp=get_parallel().enable_dsa_prefill_context_parallel,
        layer_split=get_parallel().enable_dsa_cache_layer_split,
        all_cp_transfer=envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get(),
    )
    if not get_memory().disable_radix_cache:
        raise ValueError("bounded Prefill runtime requires --disable-radix-cache")
    from sglang.srt.hardware_backend.npu.hisparse.prefill_offload import (
        PrefillKvMemoryBudget,
    )

    pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
    from sglang.srt.hardware_backend.npu.hisparse.prefill_offload_adapter import (
        NpuPrefillOffloadAdapter,
    )
    from sglang.srt.hardware_backend.npu.hisparse.prefill_offload_runtime import (
        NpuPrefillOffloadRuntime,
    )
    from sglang.srt.hardware_backend.npu.registered_host_kv import (
        NpuRegisteredHostKvPool,
    )
    from sglang.srt.hardware_backend.npu.shared_host_kv import (
        shared_host_group_from_runtime,
    )

    device_id = torch.npu.current_device()
    host_pool = NpuRegisteredHostKvPool(
        device_id, shared_group=shared_host_group_from_runtime(device_id)
    )
    pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
    budget = PrefillKvMemoryBudget(
        pool.layer_num,
        pool.prefill_layer_slots,
        (
            pool.kv_lora_rank
            + pool.qk_rope_head_dim
            + (
                0
                if getattr(pool, "_kpool_use_compress", False)
                else pool.index_head_dim
            )
        )
        * torch._utils._element_size(pool.store_dtype),
        envs.SGLANG_NPU_PREFILL_HOST_KV_BYTES.get(),
    )
    budget.validate_host_capacity(pool.size, runner.page_size)
    adapter = NpuPrefillOffloadAdapter(host_pool, page_size=runner.page_size)
    runtime = NpuPrefillOffloadRuntime(
        pool,
        runner.req_to_token_pool,
        adapter,
        publish=lambda sender, work, binding, event: sender.publish_prefill_layer(
            work,
            binding,
            event,
            layer_offset=getattr(
                runner.token_to_kv_pool, "fsa_transfer_layer_offset", 0
            ),
            layer_ids=getattr(pool, "fsa_wire_layer_ids", None),
        ),
        timeout_s=envs.SGLANG_NPU_PREFILL_LAYER_TIMEOUT_S.get(),
    )
    try:
        runtime.register_host_layers()
    except BaseException:
        host_pool.close(drained=True)
        raise
    runner.prefill_offload_adapter = adapter
    runner.prefill_offload_runtime = runtime
    runner.token_to_kv_pool.prefill_offload_runtime = runtime
