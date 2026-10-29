"""Independent device planning contract for ``external_plan``.

Physical Host token keys, rather than logical TopK IDs, are deduplicated per
request.  All operations retain fixed shapes and stay on the input device;
there are no registered Host buffers or Host planner callbacks.
"""

from dataclasses import dataclass

import torch


@dataclass
class ExternalDeviceSelectionPlan:
    physical_keys: torch.Tensor
    slots: torch.Tensor
    miss_mask: torch.Tensor
    cacheable_mask: torch.Tensor | None
    writer_mask: torch.Tensor
    row_request_slots: torch.Tensor
    union_keys: torch.Tensor
    union_slots: torch.Tensor
    union_count: torch.Tensor
    layer_id: int
    row_to_request: torch.Tensor
    union_index: torch.Tensor
    overflow: torch.Tensor
    union_owner: torch.Tensor
    union_miss: torch.Tensor
    union_cacheable: torch.Tensor
    mapping_epoch: int = 0
    request_union: bool = True
    committed: bool = False


class ExternalPlanReservation:
    """Fixed planner buffers and validation before resident identity adoption."""

    request_union = True

    def __init__(
        self,
        resident,
        rows,
        topk,
        batch_capacity,
        *,
        minimum_union_width=16,
    ):
        if min(rows, topk, batch_capacity) <= 0 or (
            topk > 2051
        ):
            raise ValueError("invalid external device planner geometry")
        if rows > batch_capacity * resident.max_routes:
            raise ValueError("query rows exceed reserved request routes")
        if resident.capacity < resident.max_routes * topk:
            raise ValueError("external device capacity must cover request union")
        if resident.capacity > 32767 or resident.request_capacity > 32767:
            raise ValueError("external device planner exceeds the int16 ABI")
        self.resident = resident
        self.device = resident.device
        self.rows = rows
        self.topk = topk
        self.capacity = resident.capacity
        self.batch_capacity = batch_capacity
        self.max_routes = resident.max_routes
        self.union_width = max(minimum_union_width, 1 << (self.max_routes * topk - 1).bit_length())
        self.union_keys = torch.full(
            (batch_capacity, self.union_width),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.union_slots = torch.full_like(self.union_keys, -1, dtype=torch.int32)
        self.union_owner = torch.full_like(self.union_slots, -1)
        self.union_miss = torch.zeros_like(self.union_slots, dtype=torch.bool)
        self.union_cacheable = torch.zeros_like(self.union_miss)
        self.union_count = torch.zeros(
            batch_capacity, dtype=torch.int32, device=self.device
        )
        self.victims = torch.full(
            (batch_capacity, self.capacity), -1, dtype=torch.int64, device=self.device
        )
        self.slots = torch.full((rows, topk), -1, dtype=torch.int64, device=self.device)
        self.miss_mask = torch.zeros_like(self.slots, dtype=torch.bool)
        self.writer_mask = torch.zeros_like(self.miss_mask)

    def _validated_inputs(
        self, physical_keys, query_ends, request_slots, cacheable_mask
    ):
        if physical_keys.ndim != 2 or query_ends.ndim != 1 or request_slots.ndim != 1:
            raise ValueError("external device planner inputs have invalid ranks")
        rows, topk = physical_keys.shape
        batch = query_ends.numel()
        if (
            not 0 < rows <= self.rows
            or topk != self.topk
            or not 0 < batch <= self.batch_capacity
        ):
            raise ValueError("external device planner inputs exceed reservation")
        if request_slots.numel() != batch:
            raise ValueError("external device request counts must match")
        if cacheable_mask is not None and (
            cacheable_mask.shape != physical_keys.shape
            or cacheable_mask.dtype != torch.bool
            or cacheable_mask.device != self.device
        ):
            raise ValueError("cacheable selection mask must match physical keys")
        if any(
            t.device != self.device for t in (physical_keys, query_ends, request_slots)
        ):
            raise ValueError("external device planner tensors must share one device")
        if any(
            t.dtype not in (torch.int32, torch.int64)
            for t in (physical_keys, query_ends, request_slots)
        ):
            raise ValueError(
                "external device planner identities and boundaries must be integers"
            )
        keys = physical_keys.to(torch.int64)
        ends = query_ends.to(torch.long)
        owners = request_slots.to(torch.long)
        safe_owners = owners.clamp(min=0, max=self.resident.request_capacity - 1)
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        # Reject a malformed cumulative partition as a whole. In particular,
        # never repair a negative prefix by clamping it into another request.
        bad_boundaries = ((ends < 0) | (ends > rows) | (ends < starts)).any()
        row_ids = torch.arange(rows, dtype=torch.long, device=self.device)
        # Rows beyond the final cumulative end are legal Graph padding only
        # when every key is explicitly invalid. Do not silently drop work.
        orphan_rows = row_ids >= ends[-1]
        bad_boundaries |= (orphan_rows[:, None] & (keys != -1)).any()
        safe_ends = torch.cummax(ends.clamp(0, rows), dim=0).values
        safe_starts = torch.cat((safe_ends.new_zeros(1), safe_ends[:-1]))
        lengths = safe_ends - safe_starts
        valid_owner = (owners > 0) & (owners < self.resident.request_capacity)
        owner_counts = torch.zeros(
            self.resident.request_capacity, dtype=torch.int32, device=self.device
        )
        owner_counts.scatter_add_(
            0, safe_owners, (valid_owner & (lengths > 0)).to(torch.int32)
        )
        duplicate = valid_owner & (owner_counts[safe_owners] > 1)
        overflow = (
            (lengths > self.max_routes)
            | bad_boundaries
            | duplicate
            | (owners < 0)
            | (owners >= self.resident.request_capacity)
        )
        active_request = valid_owner & (lengths > 0) & ~overflow
        offsets = torch.arange(self.union_width, device=self.device, dtype=torch.long)
        source_rows = safe_starts[:, None] + offsets[None, :] // topk
        source_columns = offsets[None, :] % topk
        source_valid = (
            active_request[:, None]
            & (source_rows >= 0)
            & (source_rows < safe_ends[:, None])
            & (source_rows < rows)
            & (offsets[None, :] < self.max_routes * topk)
        )
        source_ids = source_rows.clamp(min=0, max=rows - 1) * topk + source_columns
        raw = keys.reshape(-1)[source_ids]
        sentinel = torch.iinfo(torch.int64).max
        invalid_key = (source_valid & ((raw < -1) | (raw == sentinel))).any(dim=1)
        overflow |= invalid_key
        active_request &= ~invalid_key
        self.resident._error(overflow, 2)
        source_valid &= active_request[:, None] & (raw >= 0) & (raw < sentinel)
        return (
            keys,
            safe_ends,
            owners,
            safe_owners,
            safe_starts,
            overflow,
            active_request,
            offsets,
            source_ids,
            raw,
            source_valid,
        )

    def validate_inputs(
        self, physical_keys, query_ends, request_slots, *, cacheable_mask=None
    ):
        """Return structurally safe owners before resident.prepare mutates identity.

        This has no persistent resident effects except the asynchronous error
        status. Generation/epoch adoption must use these owners, not raw input.
        """
        values = self._validated_inputs(
            physical_keys, query_ends, request_slots, cacheable_mask
        )
        return values[2].masked_fill(~values[6], 0)

