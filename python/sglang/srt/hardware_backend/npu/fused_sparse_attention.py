"""Runtime contract and state for the Ascend fused sparse attention op."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ReferenceSparsePlan:
    """Logical-to-physical plan shared by the reference gather+SFA path."""

    logical_topk: torch.Tensor
    physical_token_ids: torch.Tensor
    row_requests: torch.Tensor
    valid_rows: torch.Tensor
    miss_mask: torch.Tensor


class ReferenceSparsePlanner:
    """Bounded CPU oracle cache; not used in the device execution path."""

    def __init__(self):
        self._plans = {}

    def plan(
        self,
        group_id,
        topk_indices: torch.Tensor,
        block_table: torch.Tensor,
        kv_lengths: torch.Tensor,
        query_lengths: torch.Tensor,
        *,
        page_size: int,
        request_generation: Optional[torch.Tensor] = None,
        resident_token_ids: Optional[torch.Tensor] = None,
    ) -> ReferenceSparsePlan:
        tensors = (
            topk_indices,
            block_table,
            kv_lengths,
            query_lengths,
            request_generation,
            resident_token_ids,
        )
        if any(
            tensor is not None and tensor.device.type != "cpu" for tensor in tensors
        ):
            raise ValueError("cached reference oracle requires CPU inputs")
        key = (
            page_size,
            tuple(
                (
                    None
                    if tensor is None
                    else (
                        tuple(tensor.shape),
                        tensor.dtype,
                        tuple(tensor.reshape(-1).tolist()),
                    )
                )
                for tensor in tensors
            ),
        )
        cached = self._plans.get(group_id)
        if cached is None or cached[0] != key:
            plan = build_reference_sparse_plan(
                topk_indices,
                block_table,
                kv_lengths,
                query_lengths,
                page_size=page_size,
                resident_token_ids=resident_token_ids,
            )
            self._plans[group_id] = (key, plan)
        return self._plans[group_id][1]

    def invalidate(self, group_id=None) -> None:
        if group_id is None:
            self._plans.clear()
        else:
            self._plans.pop(group_id, None)


def build_reference_sparse_plan(
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    kv_lengths: torch.Tensor,
    query_lengths: torch.Tensor,
    *,
    page_size: int,
    resident_token_ids: Optional[torch.Tensor] = None,
) -> ReferenceSparsePlan:
    """Build a deterministic request-relative reference gather plan.

    TopK IDs remain logical token IDs.  Physical IDs are derived only through
    the request page table; this keeps the reference path from accidentally
    applying the fused operator's selection-layout translation twice.
    """
    if topk_indices.dim() == 2:
        logical = topk_indices.unsqueeze(1)
    elif topk_indices.dim() == 3 and topk_indices.shape[1] == 1:
        logical = topk_indices
    else:
        raise ValueError("reference TopK must have shape [T,K] or [T,1,K]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if block_table.dim() != 2 or block_table.shape[1] == 0:
        raise ValueError("block_table must have shape [B,pages]")
    kv_lengths = kv_lengths.to(dtype=torch.int64, device=logical.device).view(-1)
    query_lengths = query_lengths.to(dtype=torch.int64, device=logical.device).view(-1)
    if (
        kv_lengths.numel() != query_lengths.numel()
        or block_table.shape[0] != kv_lengths.numel()
        or kv_lengths.numel() == 0
    ):
        raise ValueError("KV and query length counts must match")
    rows = logical.shape[0]
    row_ids = torch.arange(rows, device=logical.device, dtype=torch.int64)
    row_requests = torch.bucketize(row_ids, query_lengths, right=True)
    valid_rows = row_requests < query_lengths.numel()
    row_requests = row_requests.clamp(max=max(query_lengths.numel() - 1, 0))
    logical_ids = logical[:, 0].to(dtype=torch.int64)
    valid = valid_rows[:, None] & (logical_ids >= 0)
    valid = valid & (logical_ids < kv_lengths[row_requests, None])
    pages = block_table.to(device=logical.device, dtype=torch.int64)[row_requests]
    page_ids = torch.div(logical_ids, page_size, rounding_mode="floor")
    offsets = logical_ids.remainder(page_size)
    selected_pages = pages.gather(
        1, page_ids.clamp(min=0, max=max(block_table.shape[1] - 1, 0))
    )
    valid = valid & (page_ids < block_table.shape[1]) & (selected_pages >= 0)
    physical = selected_pages * page_size + offsets
    physical = physical.masked_fill(~valid, -1).unsqueeze(1)
    if resident_token_ids is None:
        miss_mask = valid.unsqueeze(1)
    else:
        resident = resident_token_ids.to(device=logical.device, dtype=torch.int64)
        if resident.dim() == 1:
            if kv_lengths.numel() != 1:
                raise ValueError(
                    "resident IDs must be supplied per query row for multiple requests"
                )
            resident = resident.unsqueeze(0).expand(rows, -1)
        if resident.dim() != 2 or resident.shape[0] != rows or resident.shape[1] == 0:
            raise ValueError("resident IDs must have shape [T,capacity]")
        sorted_resident = resident.sort(dim=1).values.contiguous()
        positions = torch.searchsorted(sorted_resident, logical_ids.contiguous())
        positions = positions.clamp(max=resident.shape[1] - 1)
        hit = sorted_resident.gather(1, positions) == logical_ids
        miss_mask = (valid & ~hit).unsqueeze(1)
    return ReferenceSparsePlan(
        logical.clone(), physical, row_requests, valid_rows, miss_mask
    )


_STATUS_ALIGNMENT = 8
_MEMBERSHIP_MAP_INT16_COUNT = 16376
_MEMBERSHIP_ALIGNMENT_INT16_COUNT = 16
_MEMBERSHIP_CONTROL_INT16_COUNT = 8
_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT = (
    (_MEMBERSHIP_MAP_INT16_COUNT + _MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1)
    // _MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * _MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
_MEMBERSHIP_STORAGE_INT16_COUNT = (
    (
        _MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT
        + _MEMBERSHIP_CONTROL_INT16_COUNT
        + _MEMBERSHIP_ALIGNMENT_INT16_COUNT
        - 1
    )
    // _MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * _MEMBERSHIP_ALIGNMENT_INT16_COUNT
)


class FusedSparseAttentionUnavailable(RuntimeError):
    """Raised when the standalone fused op is not loaded in the process."""


@dataclass
class FusedSparseAttentionCache:
    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_kv_block_table: torch.Tensor
    selection_kv_block_status: torch.Tensor
    selection_membership_map: torch.Tensor
    row_owner: torch.Tensor
    row_seq_len: torch.Tensor
    max_tokens: int
    max_heads: int
    max_topk: int
    max_blocks_per_row: int
    selection_topk_block_size: int
    applied_keys: torch.Tensor
    applied_keys_with_padding: torch.Tensor


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _status_stride(topk: int) -> int:
    return _ceil_div(topk + 1, _STATUS_ALIGNMENT) * _STATUS_ALIGNMENT


class FusedSparseAttentionCacheManager:
    """Own persistent selection KV state for one DSA backend instance.

    The native side-effect kernel uses status and membership metadata as mutable
    state.  Keeping the buffers on the backend, instead of constructing them in
    a forward call, also makes eager replay and graph capture use stable storage.
    """

    def __init__(self, device: torch.device, resident_capacity: Optional[int] = None):
        self.device = torch.device(device)
        if self.device.type == "npu" and self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.cache: Optional[FusedSparseAttentionCache] = None
        self.capacity_locked = False
        self.resident_capacity = resident_capacity

    @staticmethod
    def normalize_topk(topk_indices: torch.Tensor) -> torch.Tensor:
        if topk_indices.dim() == 2:
            return topk_indices.unsqueeze(1).contiguous()
        if topk_indices.dim() == 3:
            return topk_indices.contiguous()
        raise ValueError(
            "fused sparse attention expects topk indices with shape [T,K] or "
            f"[T,H,K], got {tuple(topk_indices.shape)}"
        )

    def _allocate(
        self,
        token_count: int,
        head_count: int,
        topk: int,
        full_k_rope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        selection_topk_block_size: int,
    ) -> FusedSparseAttentionCache:
        selection_block_size = full_kv_cache.shape[1]
        blocks_per_row = max(
            _ceil_div(
                max(topk * selection_topk_block_size, self.resident_capacity or 0),
                selection_block_size,
            ),
            1,
        )
        row_capacity = token_count * head_count
        selection_block_count = row_capacity * blocks_per_row
        device = self.device
        applied_keys = torch.full(
            (token_count, blocks_per_row * selection_block_size + 1),
            -1,
            dtype=torch.int64,
            device=device,
        )
        cache = FusedSparseAttentionCache(
            selection_k_rope=torch.empty(
                (selection_block_count, full_k_rope.shape[1], full_k_rope.shape[2]),
                dtype=full_k_rope.dtype,
                device=device,
            ),
            selection_kv_cache=torch.empty(
                (selection_block_count, full_kv_cache.shape[1], full_kv_cache.shape[2]),
                dtype=full_kv_cache.dtype,
                device=device,
            ),
            selection_kv_block_table=torch.arange(
                selection_block_count, dtype=torch.int32, device=device
            ).view(row_capacity, blocks_per_row),
            selection_kv_block_status=torch.full(
                (token_count, head_count, _status_stride(topk)),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            selection_membership_map=torch.full(
                (row_capacity, _MEMBERSHIP_STORAGE_INT16_COUNT),
                -1,
                dtype=torch.int16,
                device=device,
            ),
            row_owner=torch.full((token_count,), -1, dtype=torch.int64, device=device),
            row_seq_len=torch.zeros(token_count, dtype=torch.int32, device=device),
            max_tokens=token_count,
            max_heads=head_count,
            max_topk=topk,
            max_blocks_per_row=blocks_per_row,
            selection_topk_block_size=selection_topk_block_size,
            applied_keys=applied_keys[:, :-1],
            applied_keys_with_padding=applied_keys,
        )
        self.cache = cache
        return cache

    def ensure(
        self,
        topk_indices: torch.Tensor,
        full_k_rope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        selection_topk_block_size: int = 1,
    ) -> tuple[FusedSparseAttentionCache, torch.Tensor]:
        normalized = self.normalize_topk(topk_indices)
        token_count, head_count, topk = normalized.shape
        if min(token_count, head_count, topk) <= 0:
            raise ValueError("selection dimensions must be positive")
        blocks_per_row = max(
            _ceil_div(
                max(topk * selection_topk_block_size, self.resident_capacity or 0),
                full_kv_cache.shape[1],
            ),
            1,
        )
        cache = self.cache
        if (
            cache is None
            or cache.selection_kv_cache.device != self.device
            or cache.max_tokens < token_count
            or cache.max_heads != head_count
            or cache.max_topk != topk
            or cache.max_blocks_per_row != blocks_per_row
            or cache.selection_topk_block_size != selection_topk_block_size
            or cache.selection_kv_cache.shape[1] != full_kv_cache.shape[1]
            or cache.selection_kv_cache.dtype != full_kv_cache.dtype
            or cache.selection_k_rope.dtype != full_k_rope.dtype
            or cache.selection_kv_cache.shape[-1] != full_kv_cache.shape[-1]
            or cache.selection_k_rope.shape[-1] != full_k_rope.shape[-1]
        ):
            if self.capacity_locked:
                raise RuntimeError(
                    "selection cache cannot grow or change layout after graph reservation"
                )
            cache = self._allocate(
                token_count,
                head_count,
                topk,
                full_k_rope,
                full_kv_cache,
                selection_topk_block_size,
            )
        return cache, normalized[:token_count, :head_count, :topk]

    def reset(self) -> None:
        """Drop captured selection state after a backend/request teardown."""

        self.cache = None
        self.capacity_locked = False

    def views(
        self,
        cache: FusedSparseAttentionCache,
        topk_indices: torch.Tensor,
        selection_topk_block_size: int = 1,
    ):
        token_count, head_count, topk = topk_indices.shape
        blocks_per_row = max(
            _ceil_div(
                max(topk * selection_topk_block_size, self.resident_capacity or 0),
                cache.selection_kv_cache.shape[1],
            ),
            1,
        )
        row_count = token_count * head_count
        block_count = row_count * blocks_per_row
        return (
            cache.selection_k_rope[:block_count],
            cache.selection_kv_cache[:block_count],
            cache.selection_kv_block_table[:row_count, :blocks_per_row],
            cache.selection_kv_block_status[
                :token_count, :head_count, : _status_stride(topk)
            ],
            cache.selection_membership_map[:row_count],
        )

    def invalidate_changed_rows(
        self,
        cache: FusedSparseAttentionCache,
        selection_kv_block_status: torch.Tensor,
        selection_membership_map: torch.Tensor,
        owners: torch.Tensor,
        token_count: int,
        head_count: int,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        owners = owners.to(device=cache.row_owner.device, dtype=torch.int64).view(-1)
        if owners.numel() != token_count:
            raise ValueError(
                f"owner count {owners.numel()} does not match token count {token_count}"
            )
        cached_owner = cache.row_owner[:token_count]
        changed = cached_owner != owners
        if seq_lens is not None:
            seq_lens = seq_lens.to(device=cache.row_seq_len.device, dtype=torch.int32)
            changed = changed | (seq_lens < cache.row_seq_len[:token_count])
            cache.row_seq_len[:token_count].copy_(seq_lens)
        selection_kv_block_status.masked_fill_(changed.view(token_count, 1, 1), -1)
        row_changed = (
            changed.view(token_count, 1).expand(token_count, head_count).reshape(-1)
        )
        control = selection_membership_map[
            :,
            _MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT : (
                _MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT + _MEMBERSHIP_CONTROL_INT16_COUNT
            ),
        ].view(torch.int32)
        control.masked_fill_(row_changed.view(-1, 1), -1)
        cache.applied_keys[:token_count].masked_fill_(changed[:, None], -1)
        cached_owner.copy_(owners)


def paged_kv_view(tensor: torch.Tensor, page_size: int, head_dim: int) -> torch.Tensor:
    """Expose split PA storage without copying the full context to HBM."""
    if tensor.dim() == 4 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    if (
        tensor.dim() != 3
        or tensor.shape[1:] != (page_size, head_dim)
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "fused KV requires contiguous split [pages,page_size,1,dim] storage"
        )
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused KV requires FP16 or BF16 storage")
    return tensor


def mask_causal_topk(topk, row_requests, query_lengths, kv_lengths, *, positions=None):
    rows = topk.shape[0]
    row_ids = torch.arange(rows, device=topk.device, dtype=torch.int32)
    starts = torch.cat((query_lengths.new_zeros(1), query_lengths[:-1]))
    counts = query_lengths - starts
    row_limit = (
        kv_lengths[row_requests]
        - counts[row_requests]
        + row_ids
        - starts[row_requests]
        + 1
    )
    if positions is not None:
        if positions.dim() != 1 or positions.numel() != rows:
            raise ValueError(
                "sparse query positions must have one logical position per query row"
            )
        row_limit = torch.minimum(row_limit, positions.to(device=topk.device) + 1)
    row_limit = torch.minimum(row_limit, kv_lengths[row_requests])
    return topk.masked_fill((topk < 0) | (topk >= row_limit[:, None, None]), -1)


class FusedSparseAttentionState:
    """Separate mutable KV state by layer and track request-slot lifetimes."""

    def __init__(
        self,
        device,
        req_to_token_pool,
        *,
        external_planner=False,
        resident_capacity=None,
        layer_anchors=None,
        row_capacity=None,
        mutable_tail_tokens=0,
        mapped_host_plan=False,
    ):
        self.device = torch.device(device)
        if self.device.type == "npu" and self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.pool = req_to_token_pool
        self.layers = {}
        self.external_planner = external_planner
        if mapped_host_plan and not external_planner:
            raise ValueError("mapped Host plans require the external planner")
        self.mapped_host_plan = mapped_host_plan
        self.resident_capacity = resident_capacity
        self.layer_anchors = layer_anchors or {}
        self.row_capacity = row_capacity
        if mutable_tail_tokens < 0 or (mutable_tail_tokens and not external_planner):
            raise ValueError(
                "speculative selection validity requires the external planner"
            )
        self.mutable_tail_tokens = mutable_tail_tokens
        self.planners = {}
        self.plan_versions = {}
        self.forward_serial = 0
        self.generations = torch.empty_like(
            self.pool.req_generation, device=self.device
        )
        self._generation_snapshot = self.pool.req_generation.clone()
        self._pool_epoch = getattr(self.pool, "req_generation_epoch", 0)
        self._epoch = 0
        self._generations_initialized = False
        self.refresh_generations()

    def refresh_generations(self):
        """Refresh outside capture; graph replay reads this stable device buffer."""
        current = self.pool.req_generation
        self.forward_serial += 1
        pool_epoch = getattr(self.pool, "req_generation_epoch", 0)
        # Request generations live on CPU and usually stay unchanged throughout
        # decode. Avoid a blocking H2D copy while still advancing plan versions.
        if (
            self._generations_initialized
            and pool_epoch == self._pool_epoch
            and torch.equal(current, self._generation_snapshot)
        ):
            return
        # A failed upload must be retried even after the CPU snapshot advances.
        self._generations_initialized = False
        if (
            pool_epoch != self._pool_epoch
            or torch.any(current < self._generation_snapshot).item()
        ):
            self._epoch += 1
        self._pool_epoch = pool_epoch
        self._generation_snapshot.copy_(current)
        identities = current + (self._epoch << 32)
        self.generations.copy_(identities)
        self._generations_initialized = True

    def layer(self, layer_id: int) -> FusedSparseAttentionCacheManager:
        if layer_id not in self.layers:
            self.layers[layer_id] = FusedSparseAttentionCacheManager(
                self.device, self.resident_capacity
            )
        return self.layers[layer_id]

    def reserve(self, layer_id, token_capacity, topk, full_rope, full_kv):
        if self.row_capacity is not None and token_capacity > self.row_capacity:
            raise ValueError("graph query rows exceed the budgeted FSA capacity")
        manager = self.layer(layer_id)
        token_capacity = self.row_capacity or token_capacity
        indices = torch.empty(
            (token_capacity, 1, topk), dtype=torch.int32, device=self.device
        )
        manager.ensure(indices, full_rope, full_kv)
        if self.external_planner:
            self._ensure_planner(layer_id, manager.cache, topk)
        manager.capacity_locked = True

    def _ensure_planner(self, layer_id, cache, topk):
        from sglang.srt.hardware_backend.npu.external_sparse_planner import (
            MappedSparsePlanner,
            NativeSparsePlanner,
        )

        anchor = self.layer_anchors.get(layer_id, layer_id)
        capacity = cache.max_blocks_per_row * cache.selection_kv_cache.shape[1]
        planner = self.planners.get(anchor)
        if planner is None or planner.rows < cache.max_tokens:
            if self.layer(layer_id).capacity_locked:
                raise RuntimeError("shared planner cannot grow after graph reservation")
            planner_type = MappedSparsePlanner if self.mapped_host_plan else NativeSparsePlanner
            planner = planner_type(self.device, cache.max_tokens, topk, capacity)
            self.planners[anchor] = planner
        if planner.topk != topk or planner.capacity != capacity:
            raise ValueError("shared index layers have incompatible selection geometry")
        if self.mapped_host_plan:
            planner.reserve_layer(layer_id)
        return anchor, planner

    def build_inputs(
        self,
        *,
        layer_id,
        q_nope,
        q_rope,
        topk_indices,
        full_kv,
        full_rope,
        block_table,
        kv_lengths,
        query_lengths,
        request_slots,
        scale,
        query_positions=None,
    ):
        if q_nope.dim() != 3 or q_rope.shape[:2] != q_nope.shape[:2]:
            raise ValueError("fused query must have matching [T,heads,dim] components")
        topk_indices = FusedSparseAttentionCacheManager.normalize_topk(topk_indices)
        if topk_indices.shape[1] != 1:
            raise ValueError("fused sparse attention supports one selection head")
        if (
            q_nope.shape[-1] != full_kv.shape[-1]
            or q_rope.shape[-1] != full_rope.shape[-1]
        ):
            raise ValueError(
                "fused sparse attention requires absorbed queries matching KV dimensions"
            )
        token_count = q_nope.shape[0]
        if self.row_capacity is not None and token_count > self.row_capacity:
            raise ValueError("query rows exceed the budgeted FSA capacity")
        if topk_indices.shape[0] > token_count:
            raise ValueError("TopK has more rows than the query")
        if topk_indices.shape[0] < token_count:
            padding = topk_indices.new_full(
                (token_count - topk_indices.shape[0], 1, topk_indices.shape[-1]), -1
            )
            topk_indices = torch.cat((topk_indices, padding))
        query_lengths = query_lengths.to(
            device=q_nope.device, dtype=torch.int32
        ).contiguous()
        kv_lengths = kv_lengths.to(device=q_nope.device, dtype=torch.int32).contiguous()
        request_slots = request_slots.to(device=q_nope.device, dtype=torch.int64)
        batch_size = query_lengths.numel()
        if (
            batch_size == 0
            or kv_lengths.numel() != batch_size
            or request_slots.numel() != batch_size
        ):
            raise ValueError("request, query-length and KV-length counts must match")
        if block_table.dim() != 2 or block_table.shape[0] != batch_size:
            raise ValueError("full block table must have one row per request")
        query_rows = torch.arange(token_count, device=q_nope.device, dtype=torch.int32)
        row_requests = torch.bucketize(query_rows, query_lengths, right=True)
        valid = row_requests < batch_size
        row_requests = row_requests.clamp(max=batch_size - 1)
        slots = request_slots[row_requests]
        valid = (
            valid
            & (slots > 0)
            & (slots < self.generations.numel())
            & (kv_lengths[row_requests] > 0)
        )
        safe_slots = slots.clamp(min=0, max=self.generations.numel() - 1)
        identities = (
            self.generations[safe_slots] * self.generations.numel() + safe_slots
        )
        identities = torch.where(valid, identities, -1)
        topk_indices = topk_indices.to(dtype=torch.int32).masked_fill(
            ~valid[:, None, None], -1
        )
        topk_indices = mask_causal_topk(
            topk_indices,
            row_requests,
            query_lengths,
            kv_lengths,
            positions=query_positions,
        )
        manager = self.layer(layer_id)
        if manager.cache is None and self.row_capacity is not None:
            self.reserve(
                layer_id, self.row_capacity, topk_indices.shape[-1], full_rope, full_kv
            )
        cache, topk_indices = manager.ensure(topk_indices, full_rope, full_kv)
        selection_rope, selection_kv, selection_table, status, membership = (
            manager.views(cache, topk_indices)
        )
        if not self.mapped_host_plan:
            manager.invalidate_changed_rows(
                cache, status, membership, identities, token_count, 1,
                seq_lens=kv_lengths[row_requests],
            )
        selection_plan = None
        membership_region = None
        if self.external_planner:
            from sglang.srt.hardware_backend.npu.external_sparse_planner import (
                materialize_layer_plan,
            )

            reference = build_reference_sparse_plan(
                topk_indices,
                block_table,
                kv_lengths,
                query_lengths,
                page_size=full_kv.shape[1],
            )
            physical_keys = reference.physical_token_ids[:, 0].contiguous()
            physical_keys = physical_keys.masked_fill(
                physical_keys >= full_kv.shape[0] * full_kv.shape[1], -1
            )
            topk_indices = topk_indices.masked_fill(physical_keys[:, None] < 0, -1)
            anchor, planner = self._ensure_planner(
                layer_id, cache, topk_indices.shape[-1]
            )
            cacheable = None
            if self.mutable_tail_tokens:
                stable_prefix = (
                    kv_lengths[row_requests] - self.mutable_tail_tokens
                ).clamp(min=0)
                cacheable = (topk_indices[:, 0] >= 0) & (
                    topk_indices[:, 0] < stable_prefix[:, None]
                )
            if layer_id == anchor or self.plan_versions.get(anchor) != (
                self.forward_serial,
                token_count,
            ):
                slots = planner.enqueue(
                    physical_keys, identities, kv_lengths[row_requests],
                    **({"cacheable": cacheable} if self.mapped_host_plan else {}),
                )
                self.plan_versions[anchor] = (self.forward_serial, token_count)
            else:
                slots = planner.slots[:token_count]
            if self.mapped_host_plan:
                selection_plan, membership_region = planner.materialize(
                    physical_keys, layer_id, token_count,
                )
                if token_count > 16:
                    planner.stage(membership_region, cache.selection_membership_map, token_count)
                    membership_region = None
                else:
                    membership = membership_region.tensor[:token_count]
            else:
                selection_plan = materialize_layer_plan(
                    cache, physical_keys, slots, membership, cacheable_mask=cacheable,
                )
        return FusedSparseAttentionInputs(
            query=(
                q_nope if q_rope.shape[-1] == 0 else torch.cat((q_nope, q_rope), dim=-1)
            ),
            selection_k_rope=selection_rope,
            selection_kv_cache=selection_kv,
            selection_kv_block_table=selection_table,
            selection_kv_block_status=status,
            selection_membership_map=membership,
            selection_topk_indices=topk_indices,
            full_k_rope=full_rope,
            full_kv_cache=full_kv,
            full_kv_block_table=block_table.to(
                device=q_nope.device, dtype=torch.int32
            ).contiguous(),
            full_kv_actual_seq=kv_lengths,
            full_q_actual_seq=query_lengths,
            scale_value=scale,
            selection_plan=selection_plan,
            selection_cache=cache if selection_plan is not None else None,
            membership_region=membership_region,
        )


@dataclass(frozen=True)
class FusedSparseAttentionInputs:
    """Complete runtime ABI passed by the offload-aware Decode path."""

    query: torch.Tensor
    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_kv_block_table: torch.Tensor
    selection_kv_block_status: torch.Tensor
    selection_membership_map: torch.Tensor
    selection_topk_indices: torch.Tensor
    full_k_rope: torch.Tensor
    full_kv_cache: torch.Tensor
    full_kv_block_table: torch.Tensor
    full_kv_actual_seq: torch.Tensor
    full_q_actual_seq: Optional[torch.Tensor]
    scale_value: float
    sparse_block_size: int = 1
    selection_topk_block_size: int = 1
    layout_query: str = "TND"
    layout_kv: str = "PA_BSND"
    sparse_mode: int = 3
    full_rope_region: Optional[object] = None
    full_kv_region: Optional[object] = None
    selection_plan: Optional[object] = None
    selection_cache: Optional[object] = None
    membership_region: Optional[object] = None


def _resolve_fused_op():
    from sglang.srt.hardware_backend.npu.fsa_native import load_native_extension

    if load_native_extension():
        return torch.ops.sgl_fsa.npu_fused_sparse_attention_overlap
    candidates = (
        ("npu", "npu_fused_sparse_attention_overlap"),
        ("sgl_kernel", "npu_fused_sparse_attention_overlap"),
        ("sgl_kernel", "fused_sparse_attention_overlap"),
    )
    for namespace, name in candidates:
        ops_namespace = getattr(torch.ops, namespace, None)
        if ops_namespace is not None and hasattr(ops_namespace, name):
            return getattr(ops_namespace, name)
    raise FusedSparseAttentionUnavailable(
        "standalone fused sparse attention op is not loaded; expected "
        "npu_fused_sparse_attention_overlap in the NPU custom-op package"
    )


def fused_sparse_attention_overlap(
    query: torch.Tensor,
    selection_k_rope: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_kv_block_table: torch.Tensor,
    selection_kv_block_status: torch.Tensor,
    selection_membership_map: torch.Tensor,
    selection_topk_indices: torch.Tensor,
    full_k_rope: torch.Tensor,
    full_kv_cache: torch.Tensor,
    full_kv_block_table: torch.Tensor,
    full_kv_actual_seq: torch.Tensor,
    full_q_actual_seq: Optional[torch.Tensor],
    *,
    scale_value: float,
    sparse_block_size: int = 1,
    selection_topk_block_size: int = 1,
    layout_query: str = "TND",
    layout_kv: str = "PA_BSND",
    sparse_mode: int = 3,
    full_rope_region=None,
    full_kv_region=None,
    membership_region=None,
):
    """Call the standalone side-effect fused op with the frozen ABI."""

    if selection_topk_block_size != 1:
        raise ValueError("selection_topk_block_size must be 1 for GLM5.2")
    if sparse_block_size != 1:
        raise ValueError("sparse_block_size must be 1 for GLM5.2")
    if query.device.type != "npu":
        raise ValueError(
            f"fused sparse attention requires NPU query, got {query.device}"
        )
    op = _resolve_fused_op()
    addresses = {}
    if membership_region is not None:
        from sglang.srt.hardware_backend.npu.fsa_native import native_ops

        membership_region._require_live()
        if (selection_membership_map.data_ptr() != membership_region.host_ptr
                or membership_region.device_id != query.device.index):
            raise ValueError("Host plan metadata does not match its registration")
        op = native_ops().npu_fused_sparse_attention_overlap_mapped
        addresses["membership_address"] = membership_region.device_ptr
    has_rope = full_k_rope.shape[-1] != 0
    if has_rope and ((full_rope_region is None) != (full_kv_region is None)):
        raise ValueError("both registered full-KV components must be supplied")
    if not has_rope and full_rope_region is not None:
        raise ValueError("NoPE does not accept a registered RoPE source")
    if full_kv_region is not None:
        for name, tensor, region in (
            ("full_rope_address", full_k_rope, full_rope_region),
            ("full_kv_address", full_kv_cache, full_kv_region),
        ):
            if name == "full_rope_address" and not has_rope:
                continue
            if (
                tensor.device.type != "cpu"
                or not tensor.is_contiguous()
                or tensor.data_ptr() != region.host_ptr
                or tensor.nbytes > region.nbytes
                or region.device_id != query.device.index
                or region.device_ptr <= 0
            ):
                raise ValueError(
                    "full-KV metadata does not match its Host registration"
                )
            addresses[name] = region.device_ptr
    return op(
        query,
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        selection_membership_map,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        full_q_actual_seq,
        scale_value,
        sparse_block_size,
        selection_topk_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        **addresses,
    )


def run_fused_sparse_attention(inputs: FusedSparseAttentionInputs, mode="fused"):
    """Run the frozen ABI from an immutable batch input object."""

    arguments = {
        name: value
        for name, value in inputs.__dict__.items()
        if name not in ("selection_plan", "selection_cache")
    }
    if mode == "reference":
        from sglang.srt.hardware_backend.npu.external_sparse_planner import MappedSelectionPlan

        if isinstance(inputs.selection_plan, MappedSelectionPlan):
            raise ValueError("reference attention requires device-resident plan metadata")
        if inputs.selection_plan is None:
            raise ValueError(
                "miss-only reference attention requires the production planner"
            )
        from sglang.srt.hardware_backend.npu.external_sparse_planner import (
            run_reference_sparse_attention,
        )

        output = run_reference_sparse_attention(inputs)
    elif mode == "fused":
        output = fused_sparse_attention_overlap(**arguments)
    else:
        raise ValueError(f"unknown FSA execution mode: {mode}")
    if inputs.selection_plan is not None:
        from sglang.srt.hardware_backend.npu.external_sparse_planner import (
            commit_layer_plan,
        )

        commit_layer_plan(inputs.selection_cache, inputs.selection_plan)
    return output
