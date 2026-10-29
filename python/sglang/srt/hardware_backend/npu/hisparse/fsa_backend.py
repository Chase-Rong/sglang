"""Fused attention adapter for the baseline's NPU DSA and KPool metadata."""

from dataclasses import replace

import torch

from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
    FusedSparseAttentionState,
    paged_kv_view,
    run_fused_sparse_attention,
)


def sparse_pool_and_layer(pool, layer_id):
    if hasattr(pool, "full_kv_pool"):
        layer_id = pool._transfer_full_attention_id(layer_id)
        pool = pool.full_kv_pool
    return pool, layer_id - pool.start_layer


def sparse_layer_ids(pool):
    if hasattr(pool, "full_kv_pool"):
        return tuple(pool.full_attention_layer_id_mapping)
    return tuple(range(pool.start_layer, pool.start_layer + pool.layer_num))


def initialize_fsa_backend(backend, model_runner):
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg

    backend.fused_sparse_state = None
    if (
        get_disagg().disaggregation_mode == "prefill"
        and envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.get()
    ):
        return
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import (
        resolve_dsa_offload_backend,
    )

    offload_backend = resolve_dsa_offload_backend(
        envs.SGLANG_NPU_DSA_OFFLOAD_BACKEND.get()
    )
    if not envs.SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION.get():
        if (
            envs.SGLANG_NPU_DSA_OFFLOAD_BACKEND.is_set()
            and offload_backend != "external_plan"
        ):
            raise ValueError(
                "lim_copy_sfa requires fused sparse attention to be enabled"
            )
        return
    from sglang.srt.configs.model_config import dsa_layer_skips_topk
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.hisparse.speculative_sparse import (
        sparse_mtp_rows,
        sparse_mtp_tail,
    )
    from sglang.srt.runtime_context import get_exec, get_spec

    from .config import resolve_fsa_execution_abi

    execution_abi = (
        resolve_fsa_execution_abi() if offload_backend == "external_plan" else None
    )
    if offload_backend == "external_plan":
        native_ops()
        from .config import (
            resolve_shared_load_mode,
            resolve_shared_warmup_threshold,
            register_shared_execution,
        )

        load_mode = resolve_shared_load_mode()
        warmup_threshold = resolve_shared_warmup_threshold()
        if warmup_threshold:
            from sglang.srt.runtime_context import get_parallel

            if (
                get_parallel().pp_size != 1
                or not model_runner.spec_algorithm.is_none()
                or getattr(model_runner, "is_draft_worker", False)
            ):
                raise ValueError(
                    "automatic FSA warmup supports PP=1 target ordinary Decode without speculative/draft workers"
                )
        if warmup_threshold:
            from sglang.srt.hardware_backend.npu.fsa_native import (
                require_external_plan_warmup_capability,
            )

            require_external_plan_warmup_capability()
        register_shared_execution(
            backend.req_to_token_pool, load_mode, warmup_threshold
        )
    else:
        from sglang.srt.hardware_backend.npu.hisparse.config import validate_lim_dtypes

        validate_lim_dtypes(model_runner.kv_cache_dtype, model_runner.dtype)
    if backend.kv_lora_rank != 512 or backend.qk_rope_head_dim not in (0, 64):
        raise ValueError("FSA requires absorbed MLA latent=512 and RoPE=0 or 64")
    backend.fused_sparse_mode = envs.SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE.get()
    if backend.fused_sparse_mode not in ("fused", "reference"):
        raise ValueError("FSA mode must be fused or reference")
    if offload_backend == "external_plan":
        from sglang.srt.hardware_backend.npu.fsa_native import (
            require_shared_v3_capability,
        )

        if backend.fused_sparse_mode != "fused":
            raise ValueError("external_plan requires fused mode; reference execution was removed")
        if (
            backend.page_size != 128
            or backend.qk_rope_head_dim != 0
            or backend.kv_lora_rank != 512
            or str(model_runner.dtype) != "torch.bfloat16"
            or str(model_runner.kv_cache_dtype) != "torch.bfloat16"
            or getattr(backend, "tp_q_head_num", None) != 32
        ):
            raise ValueError("external_plan requires BF16/H32/NoPE512/RoPE0/page128")
        require_shared_v3_capability()
    config = model_runner.model_config.hf_text_config
    backend.fused_sparse_draft_reuse = (
        getattr(model_runner, "is_draft_worker", False)
        and getattr(config, "model_type", "") in ("glm5_next", "glm5_next_text")
        and getattr(config, "index_share_for_mtp_iteration", False)
    )
    backend.fused_sparse_topk = backend.dsa_index_topk + backend.dsa_index_kpool - 1
    configured_capacity = envs.SGLANG_NPU_FSA_RESIDENT_CAPACITY.get()
    capacity = configured_capacity
    if offload_backend == "external_plan":
        routes = sparse_mtp_rows(
            model_runner.spec_algorithm, get_spec(), model_runner.model_config
        )
        from sglang.srt.hardware_backend.npu.hisparse.external_sparse_planner import (
            expanded_resident_capacity,
        )

        capacity = expanded_resident_capacity(
            configured_capacity,
            routes,
            backend.fused_sparse_topk,
            backend.page_size,
        )
    if offload_backend == "external_plan":
        from .config import shared_union_width

        shared_union_width(routes, backend.fused_sparse_topk)
    if capacity < backend.fused_sparse_topk:
        raise ValueError(
            "FSA resident capacity must cover expanded KPool TopK and tail"
        )
    layer_ids = sparse_layer_ids(backend.token_to_kv_pool)
    anchors = {}
    anchor = None
    for layer_id in layer_ids:
        if anchor is None or not dsa_layer_skips_topk(config, layer_id):
            anchor = layer_id
        anchors[layer_id] = anchor
    owner = backend.token_to_kv_pool
    state = getattr(owner, "fsa_runtime_state", None)
    if state is None:
        graph = get_exec().graph.cuda_graph_config.decode
        graph_requests = (
            max([graph.max_bs or 0, *(graph.bs or ())])
            if graph.backend != "disabled"
            else 0
        )
        row_capacity = max(
            backend.req_to_token_pool.size, graph_requests
        ) * sparse_mtp_rows(
            model_runner.spec_algorithm, get_spec(), model_runner.model_config
        )
        if offload_backend == "external_plan":
            from .config import register_shared_geometry

            graph_reservation = (
                graph.backend,
                tuple(graph.bs or ()),
                graph.max_bs or 0,
            )
            register_shared_geometry(
                backend.req_to_token_pool,
                topk=backend.fused_sparse_topk,
                routes=routes,
                capacity=capacity,
                page_size=backend.page_size,
                dtype=model_runner.kv_cache_dtype,
                qcap=backend.req_to_token_pool.req_generation.numel(),
                bcap=max(backend.req_to_token_pool.size, graph_requests),
                tcap=row_capacity,
                graph_reservation=graph_reservation,
                abi=execution_abi,
            )
            backend.req_to_token_pool._shared_fsa_graph_reservation = graph_reservation
        if offload_backend == "lim_copy_sfa":
            from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_backend import (
                LimCopySfaState,
            )

            routes = sparse_mtp_rows(
                model_runner.spec_algorithm, get_spec(), model_runner.model_config
            )
            if (
                getattr(
                    getattr(owner, "full_kv_pool", owner), "decode_host_backing", None
                )
                is None
                or backend.qk_rope_head_dim != 0
                or backend.page_size != 128
                or backend.dsa_index_kpool != 4
                or backend.dsa_index_topk != 2048
                or backend.fused_sparse_mode != "fused"
            ):
                raise ValueError(
                    "lim_copy_sfa requires GLM5Next NoPE KPool4 Host offload"
                )
            source_tokens = (
                (model_runner.model_config.context_len + 128 + 511) // 512 * 512
            )
            capacity = max(capacity, (routes * 2051 + 127) // 128 * 128)
            state = LimCopySfaState(
                backend.device,
                backend.req_to_token_pool,
                source_tokens=source_tokens,
                routes=routes,
                capacity=capacity,
                batch_capacity=max(backend.req_to_token_pool.size, graph_requests),
                layer_anchors=anchors,
                mutable_tail=sparse_mtp_tail(
                    model_runner.spec_algorithm, get_spec(), model_runner.model_config
                ),
            )
            for layer_id in layer_ids:
                state.layer(layer_id)
        else:
            state = FusedSparseAttentionState(
                backend.device,
                backend.req_to_token_pool,
                external_planner=True,
                resident_capacity=capacity,
                layer_anchors=anchors,
                row_capacity=row_capacity,
                mutable_tail_tokens=sparse_mtp_tail(
                    model_runner.spec_algorithm, get_spec(), model_runner.model_config
                ),
                max_query_routes=routes,
                page_size=backend.page_size,
                execution_abi=execution_abi,
                batch_capacity=max(backend.req_to_token_pool.size, graph_requests),
                load_mode=load_mode,
                warmup_threshold=warmup_threshold,
            )
        if offload_backend == "external_plan":
            state.shared_topk = backend.fused_sparse_topk
        owner.fsa_runtime_state = state
    elif state.pool is not backend.req_to_token_pool:
        raise ValueError("FSA state belongs to a different request pool")
    if getattr(state, "backend_name", "external_plan") != offload_backend:
        raise ValueError("DSA offload backend cannot change after state allocation")
    if offload_backend == "external_plan" and state.execution_abi != execution_abi:
        raise ValueError("FSA execution ABI cannot change after state allocation")
    backend.fused_sparse_state = state


def reserve_fsa_graph(backend, max_num_tokens):
    state = backend.fused_sparse_state
    if state is None:
        return
    for layer_id in sparse_layer_ids(backend.token_to_kv_pool):
        key, rope = backend.token_to_kv_pool.get_kv_buffer(layer_id)
        state.reserve(
            layer_id,
            max_num_tokens,
            backend.fused_sparse_topk,
            paged_kv_view(rope, backend.page_size, backend.qk_rope_head_dim),
            paged_kv_view(key, backend.page_size, backend.kv_lora_rank),
        )


def forward_fsa(
    backend, q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope, topk_indices
):
    if save_kv_cache:
        backend.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, k_rope
        )
    full_kv, full_rope = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    metadata = backend.forward_metadata
    lengths = metadata.cache_seqlens_int32
    if topk_indices is None:
        raise ValueError("FSA requires the baseline Indexer's expanded token IDs")
    if topk_indices.shape[-1] != backend.fused_sparse_topk:
        raise ValueError("FSA TopK width must include every KPool tail column")
    pool, local_layer = sparse_pool_and_layer(backend.token_to_kv_pool, layer.layer_id)
    backing = getattr(pool, "decode_host_backing", None)
    full_table = metadata.real_page_table
    if backing is not None:
        full_table = metadata.host_page_table
        if full_table is None:
            raise RuntimeError("Integrated FSA requires the HiCache Host page table")
    if getattr(backend.fused_sparse_state, "backend_name", None) == "lim_copy_sfa":
        if backing is None or backing.page_binding is None:
            raise RuntimeError("LIM CopySFA requires mapped Host KV backing")
        return backend.fused_sparse_state.forward(
            backend,
            q,
            layer,
            forward_batch,
            topk_indices,
            full_kv,
            backing.regions[local_layer][0],
        )
    counts = backend._query_lens_device(forward_batch)
    query_lengths = counts.cumsum(0).to(dtype=lengths.dtype)
    if q_rope is None:
        q_rope = q.new_empty((*q.shape[:-1], 0))
    mapping_epoch = 0
    if backing is not None and backing.page_binding is not None:
        mapping_epoch = backing.page_binding.mapping_epoch
    inputs = backend.fused_sparse_state.build_inputs(
        layer_id=layer.layer_id,
        q_nope=q,
        q_rope=q_rope,
        topk_indices=topk_indices,
        full_kv=paged_kv_view(full_kv, backend.page_size, backend.kv_lora_rank),
        full_rope=paged_kv_view(full_rope, backend.page_size, backend.qk_rope_head_dim),
        block_table=full_table,
        kv_lengths=lengths,
        query_lengths=query_lengths,
        request_slots=forward_batch.req_pool_indices[: counts.numel()],
        scale=layer.scaling,
        query_positions=getattr(forward_batch, "positions", None),
        mapping_epoch=mapping_epoch,
    )
    if backing is not None:
        latent_region, rope_region = backing.regions[local_layer]
        inputs = replace(
            inputs,
            full_kv_region=latent_region,
            full_rope_region=rope_region,
            host_binding=backing.page_binding,
        )
    return run_fused_sparse_attention(inputs, mode=backend.fused_sparse_mode)
