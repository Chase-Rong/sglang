"""GLM5Next DSA backend sharing the existing HiCache/MemFabric offload lifecycle."""

from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_attention import (
    NativeCopySparseAttention,
)
from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import (
    LimCopySfaSpec,
)
from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_indexer import (
    NativePooledIndexer,
)
from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
    NativeIndexerManager,
)


class LimCopySfaState:
    backend_name = "lim_copy_sfa"

    def __init__(
        self,
        device,
        pool,
        *,
        source_tokens,
        routes,
        capacity,
        batch_capacity,
        layer_anchors,
        mutable_tail,
    ):
        if not 0 <= mutable_tail <= 32:
            raise ValueError("CopySFA mutable tail must be between zero and 32")
        self.device = torch.device(device)
        if self.device.type == "npu" and self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.pool = pool
        self.layers = {}
        self.layer_anchors = layer_anchors or {}
        self.mutable_tail_tokens = mutable_tail
        self.pending = {}
        self.generations = torch.empty_like(pool.req_generation, device=self.device)
        self._generation_snapshot = pool.req_generation.clone()
        self._pool_epoch = getattr(pool, "req_generation_epoch", 0)
        self._epoch = 0
        self._generations_initialized = False
        self._generation_version = 0
        self.refresh_generations()
        self.spec = LimCopySfaSpec(source_tokens, routes, capacity)
        self.batch_capacity = batch_capacity
        self.indexer = NativePooledIndexer(
            self.spec, batch_capacity * routes, self.device
        )
        self.copy_attention = NativeCopySparseAttention(
            self.device, batch_capacity * routes
        )
        self.neutral_scores = torch.zeros_like(
            self.indexer.raw_scores, dtype=torch.float32
        )

    def refresh_generations(self):
        current = self.pool.req_generation
        self.pending.clear()
        pool_epoch = getattr(self.pool, "req_generation_epoch", 0)
        if (
            self._generations_initialized
            and pool_epoch == self._pool_epoch
            and torch.equal(current, self._generation_snapshot)
        ):
            return
        self._generations_initialized = False
        if (
            pool_epoch != self._pool_epoch
            or torch.any(current < self._generation_snapshot).item()
        ):
            self._epoch += 1
        self._pool_epoch = pool_epoch
        self._generation_snapshot.copy_(current)
        self.generations.copy_(current + (self._epoch << 32))
        self._generation_version += 1
        self._generations_initialized = True

    def layer(self, layer_id):
        if layer_id not in self.layers:
            planner = NativeIndexerManager(
                self.spec,
                self.pool.req_generation.numel(),
                self.batch_capacity,
                self.device,
                indexer=self.indexer,
            )
            kv = torch.empty(
                (self.pool.req_generation.numel(), self.spec.resident_tokens, 512),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.layers[layer_id] = SimpleNamespace(planner=planner, kv=kv)
        return self.layers[layer_id]

    def reserve(self, layer_id, token_capacity, topk, full_rope, full_kv):
        if (
            token_capacity > self.batch_capacity * self.spec.max_query_routes
            or topk != 2051
        ):
            raise ValueError("LIM graph exceeds its reserved query/TopK geometry")
        self.layer(layer_id)

    def metadata(self, backend, batch, rows):
        counts = backend._query_lens_device(batch)
        if counts.numel() == 0 or not 0 < rows <= self.indexer.raw_scores.shape[0]:
            raise ValueError("LIM query rows exceed the reserved batch geometry")
        ends = counts.cumsum(0).to(torch.int32)
        lengths = (batch.positions[:rows] + 1).to(torch.int32)
        if lengths.numel() != rows:
            raise ValueError("LIM requires per-query causal positions")
        requests = batch.req_pool_indices[: counts.numel()]
        if rows % counts.numel():
            raise ValueError(
                "LIM serving requires uniform Decode/Verify/Draft query routes"
            )
        active = (
            (requests > 0) & (requests < self.generations.numel())
        ).repeat_interleave(rows // counts.numel())
        lengths = torch.where(active, lengths, 0)
        table = backend.forward_metadata.host_page_table
        if table is None:
            raise RuntimeError("LIM requires HiCache Host page mapping")
        return ends, lengths, requests, table

    def select(self, backend, batch, layer_id, query, key, weights, pool_table):
        from sglang.srt.hardware_backend.npu.hisparse.fsa_backend import (
            sparse_pool_and_layer,
        )

        ends, lengths, requests, table = self.metadata(backend, batch, query.shape[0])
        pool, local_layer = sparse_pool_and_layer(backend.token_to_kv_pool, layer_id)
        backing = pool.decode_host_backing
        region = backing.regions[local_layer][0]
        manager = self.layer(layer_id).planner
        topk, _, _ = manager.select_and_plan(
            query,
            key,
            weights,
            pool_table,
            ends,
            lengths,
            table,
            requests,
            self.generations,
            host_tokens=region.tensor.numel() // 512,
            mutable_tail=self.mutable_tail_tokens,
        )
        self.pending[layer_id] = (
            topk,
            batch,
            backend.forward_metadata,
            (ends, lengths, requests, table),
        )
        return topk

    def forward(self, backend, query, layer, batch, topk, full_kv, region):
        rows = query.shape[0]
        current = self.layer(layer.layer_id)
        selected = self.pending.pop(layer.layer_id, None)
        reuse_selected = (
            selected is not None
            and selected[0].data_ptr() == topk.data_ptr()
            and selected[0].shape == topk.shape
            and selected[1] is batch
            and selected[2] is backend.forward_metadata
        )
        if reuse_selected:
            ends, lengths, requests, table = selected[3]
        else:
            ends, lengths, requests, table = self.metadata(backend, batch, rows)

        cache = getattr(batch, "draft_sparse_plan_cache", None)
        if (
            not getattr(batch, "reuse_dsa_topk_indices", False)
            or not batch.forward_mode.is_decode()
            or rows != requests.numel()
        ):
            cache = None
        key = None
        identity = None
        if cache is not None:
            from sglang.srt.hardware_backend.npu.hisparse.fsa_backend import (
                sparse_pool_and_layer,
            )

            pool, _ = sparse_pool_and_layer(backend.token_to_kv_pool, layer.layer_id)
            backing = pool.decode_host_backing
            binding = getattr(backing, "page_binding", None)
            key = ("lim_copy_sfa", self, layer.layer_id)
            identity = (
                id(batch),
                rows,
                topk.data_ptr(),
                topk.stride(),
                None if torch.is_inference(topk) else topk._version,
                requests.data_ptr(),
                None if torch.is_inference(requests) else requests._version,
                table.data_ptr(),
                table.shape,
                None if torch.is_inference(table) else table._version,
                id(region),
                self._generation_version,
                getattr(binding, "mapping_epoch", 0),
            )
        reuse_plan = (
            not reuse_selected and cache is not None and cache.get(key) == identity
        )
        if not reuse_selected and not reuse_plan:
            current.planner.plan(
                topk.contiguous(),
                self.neutral_scores[:rows],
                ends,
                lengths,
                table,
                requests,
                self.generations,
                host_tokens=full_kv.numel() // 512,
                mutable_tail=self.mutable_tail_tokens,
                preserve_prefix=cache is not None,
            )
        output = self.copy_attention.forward(
            query.contiguous(),
            full_kv,
            region,
            current.kv,
            current.planner,
            ends,
            lengths,
            backend.forward_metadata.cache_seqlens_int32,
            table,
            requests,
            scale=layer.scaling,
            mutable_tail=self.mutable_tail_tokens,
            reuse_plan=reuse_plan,
        )
        if cache is not None and not reuse_plan:
            from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_reuse import (
                publish_draft_plan,
            )

            publish_draft_plan(
                current.planner,
                lengths,
                table,
                requests,
                host_tokens=full_kv.numel() // 512,
                mutable_tail=self.mutable_tail_tokens,
            )
            cache[key] = identity
        return output


def select_offloaded_topk(backend, indexer, batch, layer_id, query, weights, metadata):
    from sglang.kernels.ops.attention.kpool_bf16_index import (
        build_pooled_page_table_64,
    )

    backend = getattr(backend, "full_attn_backend", backend)
    state = getattr(backend, "fused_sparse_state", None)
    if (
        not isinstance(state, LimCopySfaState)
        or batch.forward_mode.is_extend_without_speculative()
    ):
        return None
    tables = metadata.get_page_table_64()
    if not batch.forward_mode.is_decode_or_idle():
        tables = tables[metadata.get_token_to_batch_idx()]
    pool_table = build_pooled_page_table_64(tables, indexer.index_kpool).contiguous()
    reserved_columns = state.indexer.raw_scores.shape[1] // 128
    if pool_table.shape[1] < reserved_columns:
        pool_table = torch.nn.functional.pad(
            pool_table, (0, reserved_columns - pool_table.shape[1]), value=0
        )
    rows = pool_table.shape[0]
    if rows != query.shape[0]:
        raise ValueError(
            "LIM requires explicit inactive request rows for Graph padding"
        )
    key = backend.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
    result = state.select(
        backend,
        batch,
        layer_id,
        query[:rows].contiguous(),
        key,
        weights[:rows].reshape(rows, indexer.n_heads).to(torch.bfloat16).contiguous(),
        pool_table,
    )
    return result
