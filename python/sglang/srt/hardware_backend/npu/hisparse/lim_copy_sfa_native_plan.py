"""Native resident-management qualification boundary for A5 LIM fusion."""

import torch
import triton

from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_device_plan import (
    _commit,
    _mutable_presence,
)
from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_indexer import (
    NativePooledIndexer,
    native_indexer_ops,
)


class NativeUnionPlanner:
    replacement_policy = "protected_hash"

    def __init__(self, spec, request_capacity, batch_capacity, device):
        if (
            not (0 < batch_capacity and 1 < request_capacity <= 32767)
            or spec.resident_tokens > 12416
        ):
            raise ValueError(
                "Native resident geometry exceeds the A5 qualification bounds"
            )
        self.spec = spec
        self.ops = native_indexer_ops()
        self.request_capacity = request_capacity
        self.batch_capacity = batch_capacity
        self.rows = batch_capacity * spec.max_query_routes
        self.union_width = triton.next_power_of_2(
            spec.max_query_routes * spec.token_topk
        )
        self.source_map = torch.full(
            (request_capacity, spec.source_capacity),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.sources = torch.full(
            (request_capacity, spec.resident_tokens),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.physical = torch.full_like(self.sources, -1, dtype=torch.int64)
        self.state = torch.full(
            (request_capacity, 16), -1, dtype=torch.int64, device=device
        )
        self.topk = torch.empty((self.rows, 2051), dtype=torch.int32, device=device)
        self.slots = torch.empty_like(self.topk)
        self.membership = torch.empty(
            (self.rows, 16400), dtype=torch.int16, device=device
        )
        self.union_sources = torch.empty(
            (batch_capacity, self.union_width), dtype=torch.int32, device=device
        )
        self.union_slots = torch.empty_like(self.union_sources)
        self.union_physical = torch.empty_like(self.union_sources, dtype=torch.int64)
        self.mutable_presence = torch.empty(
            (self.rows, 32), dtype=torch.int32, device=device
        )

    def plan(
        self,
        ids,
        scores,
        query_ends,
        lengths,
        table,
        request_slots,
        generations,
        *,
        host_tokens,
        mutable_tail=12,
        pooled=False,
        encoded=False,
        preserve_prefix=False,
    ):
        rows, batch = ids.shape[0], query_ends.numel()
        if not 0 < rows <= self.rows or not 0 < batch <= self.batch_capacity:
            raise ValueError("Native resident inputs exceed reserved rows")
        self.ops.manage_out(
            ids,
            scores,
            query_ends,
            lengths,
            table,
            request_slots,
            generations,
            self.source_map,
            self.sources,
            self.physical,
            self.state,
            self.topk[:rows],
            self.slots[:rows],
            self.membership[:rows],
            self.union_sources[:batch],
            self.union_slots[:batch],
            self.union_physical[:batch],
            host_tokens,
            mutable_tail,
            pooled,
            encoded,
            **({"preserve_prefix": True} if preserve_prefix else {}),
        )
        return self.topk[:rows], self.slots[:rows], self.membership[:rows]

    def commit(self, request_slots, query_ends, row_lengths, *, mutable_tail=12):
        if not 0 <= mutable_tail <= 32:
            raise ValueError("Mutable tail exceeds reserved storage")
        rows = row_lengths.numel()
        if mutable_tail:
            _mutable_presence[(rows, mutable_tail)](
                self.topk, row_lengths, self.mutable_presence, 2051, mutable_tail, 4096
            )
        _commit[(request_slots.numel(), triton.cdiv(self.union_width, 256))](
            self.union_slots,
            self.union_physical,
            self.union_sources,
            self.mutable_presence,
            query_ends,
            row_lengths,
            request_slots,
            self.physical,
            self.union_width,
            self.spec.resident_tokens,
            self.request_capacity,
            rows,
            self.spec.max_query_routes,
            mutable_tail,
            256,
            counts=self.state,
        )


class NativeIndexerManager(NativeUnionPlanner):
    """Single native launch for pooled selection and resident planning."""

    def __init__(self, spec, request_capacity, batch_capacity, device, *, indexer=None):
        super().__init__(spec, request_capacity, batch_capacity, device)
        self.indexer = indexer or NativePooledIndexer(spec, self.rows, device)

    def select_and_plan(
        self,
        query,
        key,
        weights,
        pool_table,
        query_ends,
        lengths,
        host_table,
        request_slots,
        generations,
        *,
        host_tokens,
        mutable_tail=12,
    ):
        rows, batch = query.shape[0], query_ends.numel()
        if not 0 < rows <= self.rows or not 0 < batch <= self.batch_capacity:
            raise ValueError("Fused LIM inputs exceed reserved rows")
        indexer = self.indexer
        self.ops.indexer_manage_out(
            query,
            key,
            weights,
            pool_table,
            query_ends,
            lengths,
            host_table,
            request_slots,
            generations,
            indexer.ends[:rows],
            indexer.lengths[:rows],
            indexer.indices[:rows],
            indexer.raw_scores[:rows],
            indexer.workspace,
            self.source_map,
            self.sources,
            self.physical,
            self.state,
            self.topk[:rows],
            self.slots[:rows],
            self.membership[:rows],
            self.union_sources[:batch],
            self.union_slots[:batch],
            self.union_physical[:batch],
            host_tokens,
            mutable_tail,
        )
        return self.topk[:rows], self.slots[:rows], self.membership[:rows]
