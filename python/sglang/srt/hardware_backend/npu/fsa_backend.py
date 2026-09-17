"""Fused attention adapter for the baseline's NPU DSA and KPool metadata."""

from dataclasses import replace

from sglang.srt.hardware_backend.npu.fused_sparse_attention import (
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
    from sglang.srt.configs.model_config import dsa_layer_skips_topk
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.speculative_sparse import sparse_mtp_rows, sparse_mtp_tail
    from sglang.srt.runtime_context import get_exec, get_spec

    backend.fused_sparse_state = None
    if not envs.SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION.get():
        return
    native_ops()
    if backend.kv_lora_rank != 512 or backend.qk_rope_head_dim not in (0, 64):
        raise ValueError("FSA requires absorbed MLA latent=512 and RoPE=0 or 64")
    backend.fused_sparse_mode = envs.SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE.get()
    if backend.fused_sparse_mode not in ("fused", "reference"):
        raise ValueError("FSA mode must be fused or reference")
    config = model_runner.model_config.hf_text_config
    backend.fused_sparse_topk = (
        int(config.index_topk) + int(getattr(config, "index_kpool", 1)) - 1
    )
    capacity = envs.SGLANG_NPU_FSA_RESIDENT_CAPACITY.get()
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
        graph_requests = (max([graph.max_bs or 0, *(graph.bs or ())])
                          if graph.backend != "disabled" else 0)
        row_capacity = max(backend.req_to_token_pool.size, graph_requests) * sparse_mtp_rows(
            model_runner.spec_algorithm, get_spec(), model_runner.model_config
        )
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
        )
        owner.fsa_runtime_state = state
    elif state.pool is not backend.req_to_token_pool:
        raise ValueError("FSA state belongs to a different request pool")
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
    counts = backend._query_lens_device(forward_batch)
    query_lengths = counts.cumsum(0).to(dtype=lengths.dtype)
    if q_rope is None:
        q_rope = q.new_empty((*q.shape[:-1], 0))
    if topk_indices is None:
        raise ValueError("FSA requires the baseline Indexer's expanded token IDs")
    if topk_indices.shape[-1] != backend.fused_sparse_topk:
        raise ValueError("FSA TopK width must include every KPool tail column")
    inputs = backend.fused_sparse_state.build_inputs(
        layer_id=layer.layer_id,
        q_nope=q,
        q_rope=q_rope,
        topk_indices=topk_indices,
        full_kv=paged_kv_view(full_kv, backend.page_size, backend.kv_lora_rank),
        full_rope=paged_kv_view(full_rope, backend.page_size, backend.qk_rope_head_dim),
        block_table=metadata.real_page_table,
        kv_lengths=lengths,
        query_lengths=query_lengths,
        request_slots=forward_batch.req_pool_indices[: counts.numel()],
        scale=layer.scaling,
        query_positions=getattr(forward_batch, "positions", None),
    )
    pool, local_layer = sparse_pool_and_layer(backend.token_to_kv_pool, layer.layer_id)
    backing = getattr(pool, "decode_host_backing", None)
    if backing is not None:
        latent_region, rope_region = backing.regions[local_layer]
        inputs = replace(
            inputs, full_kv_region=latent_region, full_rope_region=rope_region
        )
    return run_fused_sparse_attention(inputs, mode=backend.fused_sparse_mode)
