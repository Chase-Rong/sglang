"""Supplied-TopK A5 component with bounded tiles and NPU sorting.

This is a correctness bring-up boundary, not the completed fused Indexer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _collect(
    ids,
    ends,
    lengths,
    table,
    requests,
    raw,
    TOPK: tl.constexpr,
    SOURCE: tl.constexpr,
    COLS: tl.constexpr,
    UNION: tl.constexpr,
    REQUEST_CAP: tl.constexpr,
    ROWS: tl.constexpr,
    HOST_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    end = tl.load(ends + request)
    start = tl.load(ends + request - 1, mask=request > 0, other=0)
    owner = tl.load(requests + request)
    rows = start + offsets // TOPK
    valid = (rows < end) & (rows < ROWS) & (owner > 0) & (owner < REQUEST_CAP)
    keys = tl.load(ids + rows * TOPK + offsets % TOPK, mask=valid, other=-1)
    length = tl.load(lengths + rows, mask=valid, other=0)
    page = tl.load(
        table + request * COLS + keys // 128,
        mask=valid & (keys >= 0) & (keys // 128 < COLS),
        other=-1,
    )
    valid = valid & (keys >= 0) & (keys < length) & (keys < SOURCE)
    valid = valid & (page >= 0) & (page.to(tl.int64) * 128 + keys % 128 < HOST_TOKENS)
    tl.store(
        raw + request * UNION + offsets,
        tl.where(valid, keys, SOURCE),
        mask=offsets < UNION,
    )


@triton.jit
def _union(
    sorted_keys,
    ends,
    lengths,
    requests,
    generations,
    mapping,
    sources,
    physical,
    owners,
    previous_lengths,
    protected,
    union,
    counts,
    SOURCE: tl.constexpr,
    CAPACITY: tl.constexpr,
    UNION: tl.constexpr,
    REQUEST_CAP: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    owner = tl.load(requests + request)
    active = (owner > 0) & (owner < REQUEST_CAP)
    owner = tl.minimum(tl.maximum(owner, 0), REQUEST_CAP - 1)
    end = tl.load(ends + request)
    generation = tl.load(generations + owner)
    length = tl.load(
        lengths + end - 1, mask=active & (end > 0) & (end <= ROWS), other=0
    )
    reset = active & (
        (tl.load(owners + owner) != generation)
        | (length < tl.load(previous_lengths + owner))
    )
    local = tl.arange(0, BLOCK)
    for block in range(tl.cdiv(CAPACITY, BLOCK)):
        slots = block * BLOCK + local
        old = tl.load(
            sources + owner * CAPACITY + slots,
            mask=reset & (slots < CAPACITY),
            other=-1,
        )
        tl.store(
            mapping + owner * SOURCE + old,
            -1,
            mask=reset & (slots < CAPACITY) & (old >= 0),
        )
        tl.store(
            sources + owner * CAPACITY + slots, -1, mask=reset & (slots < CAPACITY)
        )
        tl.store(
            physical + owner * CAPACITY + slots, -1, mask=reset & (slots < CAPACITY)
        )
        tl.store(protected + request * CAPACITY + slots, 0, mask=slots < CAPACITY)
    tl.debug_barrier()
    total = 0
    for block in range(tl.cdiv(UNION, BLOCK)):
        offsets = block * BLOCK + local
        keys = tl.load(
            sorted_keys + request * UNION + offsets, mask=offsets < UNION, other=SOURCE
        )
        previous = tl.load(
            sorted_keys + request * UNION + offsets - 1,
            mask=(offsets > 0) & (offsets < UNION),
            other=SOURCE,
        )
        unique = active & (keys < SOURCE) & ((offsets == 0) | (keys != previous))
        rank = tl.cumsum(unique.to(tl.int32)) - 1
        tl.store(union + request * UNION + total + rank, keys, mask=unique)
        mapped = tl.load(mapping + owner * SOURCE + keys, mask=unique, other=-1)
        tl.store(
            protected + request * CAPACITY + mapped,
            1,
            mask=unique & (mapped >= 0) & (mapped < CAPACITY),
        )
        total += tl.sum(unique.to(tl.int32), 0)
    tl.store(counts + request, total)
    tl.store(owners + owner, generation, mask=active)
    tl.store(previous_lengths + owner, length, mask=active)


@triton.jit
def _priorities(
    scores,
    ends,
    requests,
    sources,
    protected,
    priorities,
    POOLS: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROUTES: tl.constexpr,
    REQUEST_CAP: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    owner = tl.minimum(tl.maximum(tl.load(requests + request), 0), REQUEST_CAP - 1)
    end = tl.load(ends + request)
    start = tl.load(ends + request - 1, mask=request > 0, other=0)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    old = tl.load(sources + owner * CAPACITY + slots, mask=slots < CAPACITY, other=-1)
    protect = tl.load(
        protected + request * CAPACITY + slots, mask=slots < CAPACITY, other=1
    )
    priority = tl.full((BLOCK,), 0, tl.float32)
    score_valid = tl.full((BLOCK,), False, tl.int1)
    for route in range(ROUTES):
        row = start + route
        score = tl.load(
            scores + row * POOLS + old // 4,
            mask=(row < end)
            & (row < ROWS)
            & (slots < CAPACITY)
            & (old >= 0)
            & (old // 4 < POOLS),
            other=-float("inf"),
        )
        finite = (score != -float("inf")) & (score != float("inf"))
        priority += tl.where(finite, score, 0.0)
        score_valid = score_valid | finite
    priority = tl.where(score_valid & (old >= 0), priority, -float("inf"))
    tl.store(
        priorities + request * CAPACITY + slots,
        tl.where(protect == 0, priority, float("inf")),
        mask=slots < CAPACITY,
    )


@triton.jit
def _assign(
    table,
    requests,
    mapping,
    sources,
    physical,
    victims,
    union,
    counts,
    union_slots,
    union_physical,
    SOURCE: tl.constexpr,
    CAPACITY: tl.constexpr,
    COLS: tl.constexpr,
    UNION: tl.constexpr,
    REQUEST_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    owner = tl.load(requests + request)
    active = (owner > 0) & (owner < REQUEST_CAP)
    owner = tl.minimum(tl.maximum(owner, 0), REQUEST_CAP - 1)
    count = tl.load(counts + request)
    local = tl.arange(0, BLOCK)
    misses_before = 0
    for block in range(tl.cdiv(UNION, BLOCK)):
        offsets = block * BLOCK + local
        valid = (offsets < count) & active
        selected = tl.load(union + request * UNION + offsets, mask=valid, other=SOURCE)
        mapped = tl.load(mapping + owner * SOURCE + selected, mask=valid, other=-1)
        hit = valid & (mapped >= 0) & (mapped < CAPACITY)
        missing = valid & ~hit
        misses = tl.sum(missing.to(tl.int32), 0)
        assigned = mapped
        if misses > 0:
            rank = misses_before + tl.cumsum(missing.to(tl.int32)) - 1
            victim = tl.load(
                victims + request * CAPACITY + rank,
                mask=missing & (rank < CAPACITY),
                other=-1,
            ).to(tl.int32)
            assigned = tl.where(hit, mapped, victim)
            valid = valid & (assigned >= 0) & (assigned < CAPACITY)
            evicted = tl.load(
                sources + owner * CAPACITY + assigned, mask=missing & valid, other=-1
            )
            tl.store(
                mapping + owner * SOURCE + evicted,
                -1,
                mask=missing & valid & (evicted >= 0),
            )
            tl.store(physical + owner * CAPACITY + assigned, -1, mask=missing & valid)
            tl.debug_barrier()
            tl.store(
                mapping + owner * SOURCE + selected, assigned, mask=missing & valid
            )
            tl.store(
                sources + owner * CAPACITY + assigned, selected, mask=missing & valid
            )
        valid = valid & (assigned >= 0) & (assigned < CAPACITY)
        page = tl.load(
            table + request * COLS + selected // 128,
            mask=valid & (selected // 128 < COLS),
            other=-1,
        )
        tl.store(
            union_slots + request * UNION + offsets,
            tl.where(valid, assigned, -1),
            mask=offsets < UNION,
        )
        tl.store(
            union_physical + request * UNION + offsets,
            tl.where(valid, page.to(tl.int64) * 128 + selected % 128, -1),
            mask=offsets < UNION,
        )
        misses_before += misses


@triton.jit
def _encode(
    ids,
    ends,
    lengths,
    table,
    requests,
    mapping,
    physical,
    output_ids,
    output_slots,
    membership,
    TOPK: tl.constexpr,
    SOURCE: tl.constexpr,
    CAPACITY: tl.constexpr,
    COLS: tl.constexpr,
    UNION: tl.constexpr,
    REQUEST_CAP: tl.constexpr,
    ROWS: tl.constexpr,
    MUTABLE: tl.constexpr,
    HOST_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    start = tl.load(ends + request - 1, mask=request > 0, other=0)
    end = tl.load(ends + request)
    owner = tl.load(requests + request)
    active = (owner > 0) & (owner < REQUEST_CAP)
    owner = tl.minimum(tl.maximum(owner, 0), REQUEST_CAP - 1)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    rows = start + offsets // TOPK
    output_mask = (rows < end) & (rows < ROWS) & (offsets < UNION)
    keys = tl.load(
        ids + rows * TOPK + offsets % TOPK, mask=output_mask & active, other=-1
    )
    visible = tl.load(lengths + rows, mask=output_mask & active, other=0)
    valid = output_mask & active & (keys >= 0) & (keys < SOURCE) & (keys < visible)
    slots = tl.load(mapping + owner * SOURCE + keys, mask=valid, other=-1)
    valid = valid & (slots >= 0) & (slots < CAPACITY)
    page = tl.load(
        table + request * COLS + keys // 128,
        mask=valid & (keys // 128 < COLS),
        other=-1,
    )
    address = page.to(tl.int64) * 128 + keys % 128
    applied = tl.load(physical + owner * CAPACITY + slots, mask=valid, other=-1)
    valid = valid & (page >= 0) & (address < HOST_TOKENS)
    hit = valid & (applied == address) & (keys < tl.maximum(visible - MUTABLE, 0))
    encoded = tl.where(valid, tl.where(hit, slots + 1, -(slots + 1)), 0)
    tl.store(
        output_ids + rows * TOPK + offsets % TOPK,
        tl.where(valid, keys, -1),
        mask=output_mask,
    )
    tl.store(
        output_slots + rows * TOPK + offsets % TOPK,
        tl.where(valid, slots, -1),
        mask=output_mask,
    )
    tl.store(
        membership + rows * 16400 + 16384 - TOPK + offsets % TOPK,
        encoded,
        mask=output_mask,
    )
    if tl.program_id(1) == 0:
        for row in range(start, end):
            control = tl.arange(0, 8)
            header = tl.where(
                control == 0,
                0x5A4D,
                tl.where(
                    control == 1,
                    0x5A45,
                    tl.where(
                        control == 2,
                        TOPK,
                        tl.where(
                            control == 3,
                            16384 - TOPK,
                            tl.where(
                                control == 4,
                                owner,
                                tl.where(
                                    control == 5,
                                    0x5A44,
                                    tl.where(control == 6, CAPACITY, 0),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            tl.store(
                membership + row * 16400 + 16384 + control, header, mask=row < ROWS
            )


@triton.jit
def _mutable_presence(
    ids,
    lengths,
    presence,
    TOPK: tl.constexpr,
    MUTABLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tail = tl.program_id(1)
    token = tl.load(lengths + row) - MUTABLE + tail
    columns = tl.arange(0, BLOCK)
    keys = tl.load(ids + row * TOPK + columns, mask=columns < TOPK, other=-1)
    selected = tl.max(
        ((keys == token) & (token >= 0) & (columns < TOPK)).to(tl.int32), 0
    )
    tl.store(presence + row * 32 + tail, selected)


@triton.jit
def _commit(
    slots,
    keys,
    logical,
    presence,
    ends,
    lengths,
    requests,
    physical,
    UNION: tl.constexpr,
    CAPACITY: tl.constexpr,
    REQUEST_CAP: tl.constexpr,
    ROWS: tl.constexpr,
    ROUTES: tl.constexpr,
    MUTABLE: tl.constexpr,
    BLOCK: tl.constexpr,
    counts=None,
):
    request = tl.program_id(0)
    owner = tl.load(requests + request)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid_count = UNION
    if counts is not None:
        valid_count = tl.load(
            counts + owner * 16 + 3, mask=(owner > 0) & (owner < REQUEST_CAP), other=0
        )
    selected = tl.load(
        slots + request * UNION + offsets,
        mask=(offsets < UNION) & (offsets < valid_count),
        other=-1,
    )
    value = tl.load(
        keys + request * UNION + offsets,
        mask=(offsets < UNION) & (offsets < valid_count),
        other=-1,
    )
    source = tl.load(
        logical + request * UNION + offsets,
        mask=(offsets < UNION) & (offsets < valid_count),
        other=-1,
    )
    start = tl.load(ends + request - 1, mask=request > 0, other=0)
    end = tl.load(ends + request)
    reusable = tl.full((BLOCK,), True, tl.int1)
    for route in range(ROUTES):
        row = start + route
        visible = tl.load(lengths + row, mask=(row < end) & (row < ROWS), other=0)
        tail = source - visible + MUTABLE
        mutable = tl.load(
            presence + row * 32 + tail,
            mask=(row < end)
            & (row < ROWS)
            & (source >= 0)
            & (tail >= 0)
            & (tail < MUTABLE),
            other=0,
        )
        reusable = reusable & (mutable == 0)
    tl.store(
        physical + owner * CAPACITY + selected,
        tl.where(reusable, value, -1),
        mask=(owner > 0)
        & (owner < REQUEST_CAP)
        & (selected >= 0)
        & (selected < CAPACITY),
    )


class DeviceUnionPlanner:
    """Component buffers; all planning executes on NPU without Host callbacks."""

    def __init__(self, spec, request_capacity, batch_capacity, device):
        if (
            not 1 < request_capacity <= 32767
            or not 0 < batch_capacity < request_capacity
        ):
            raise ValueError("Request storage must include the padding slot")
        self.spec, self.request_capacity, self.batch_capacity = (
            spec,
            request_capacity,
            batch_capacity,
        )
        self.rows = batch_capacity * spec.max_query_routes
        self.union_width = triton.next_power_of_2(
            spec.max_query_routes * spec.token_topk
        )
        options = dict(device=device)
        self.source_map = torch.full(
            (request_capacity, spec.source_capacity), -1, dtype=torch.int32, **options
        )
        self.sources = torch.full(
            (request_capacity, spec.resident_tokens), -1, dtype=torch.int32, **options
        )
        self.physical = torch.full_like(self.sources, -1, dtype=torch.int64)
        self.owners = torch.full((request_capacity,), -1, dtype=torch.int64, **options)
        self.lengths = torch.full_like(self.owners, -1)
        self.protected = torch.empty(
            (batch_capacity, spec.resident_tokens), dtype=torch.int32, **options
        )
        self.priorities = torch.empty_like(self.protected, dtype=torch.float32)
        self.sorted_priorities = torch.empty_like(self.priorities)
        self.victims = torch.empty_like(self.protected, dtype=torch.int64)
        self.counts = torch.empty(batch_capacity, dtype=torch.int32, **options)
        self.raw = torch.empty(
            (batch_capacity, self.union_width), dtype=torch.int32, **options
        )
        self.sorted_sources = torch.empty_like(self.raw)
        self.sort_indices = torch.empty_like(self.raw, dtype=torch.int64)
        self.union_sources = torch.empty_like(self.raw)
        self.union_slots = torch.empty_like(self.raw)
        self.union_physical = torch.empty_like(self.raw, dtype=torch.int64)
        self.topk = torch.empty(
            (self.rows, spec.token_topk), dtype=torch.int32, **options
        )
        self.slots = torch.empty_like(self.topk)
        self.membership = torch.empty((self.rows, 16400), dtype=torch.int16, **options)
        self.mutable_presence = torch.empty(
            (self.rows, 32), dtype=torch.int32, **options
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
    ):
        expected = (
            (ids, torch.int32, 2),
            (scores, torch.float32, 2),
            (query_ends, torch.int32, 1),
            (lengths, torch.int32, 1),
            (table, torch.int32, 2),
            (request_slots, torch.int64, 1),
            (generations, torch.int64, 1),
        )
        for tensor, dtype, dimensions in expected:
            if (
                tensor.dtype != dtype
                or tensor.ndim != dimensions
                or tensor.device != self.source_map.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "LIM inputs must match the reserved device, dtype and contiguous layout"
                )
        count, rows = query_ends.numel(), ids.shape[0]
        if rows < 1 or count < 1 or host_tokens < 1 or not 0 <= mutable_tail <= 32:
            raise ValueError("Invalid LIM active geometry or mutable tail")
        if rows > self.rows or ids.shape[1] != self.spec.token_topk:
            raise ValueError("LIM query rows exceed the reserved shape")
        if count > self.batch_capacity or table.shape[0] != count:
            raise ValueError("LIM request count exceeds reserved capacity")
        if generations.numel() != self.request_capacity:
            raise ValueError("LIM generations must match the request pool")
        if (
            scores.shape != (rows, self.spec.source_capacity // self.spec.pool_size)
            or lengths.numel() != rows
            or request_slots.numel() != count
            or table.shape[1] < 1
        ):
            raise ValueError("LIM score, length and request tensors disagree")
        spec = self.spec
        grid = (count, triton.cdiv(self.union_width, 256))
        _collect[grid](
            ids,
            query_ends,
            lengths,
            table,
            request_slots,
            self.raw,
            spec.token_topk,
            spec.source_capacity,
            table.shape[1],
            self.union_width,
            self.request_capacity,
            rows,
            host_tokens,
            256,
        )
        torch.sort(
            self.raw[:count],
            stable=True,
            dim=-1,
            out=(self.sorted_sources[:count], self.sort_indices[:count]),
        )
        _union[(count,)](
            self.sorted_sources,
            query_ends,
            lengths,
            request_slots,
            generations,
            self.source_map,
            self.sources,
            self.physical,
            self.owners,
            self.lengths,
            self.protected,
            self.union_sources,
            self.counts,
            spec.source_capacity,
            spec.resident_tokens,
            self.union_width,
            self.request_capacity,
            rows,
            2048 if spec.max_query_routes > 1 else 1024,
        )
        _priorities[(count, triton.cdiv(spec.resident_tokens, 256))](
            scores,
            query_ends,
            request_slots,
            self.sources,
            self.protected,
            self.priorities,
            scores.shape[1],
            spec.resident_tokens,
            spec.max_query_routes,
            self.request_capacity,
            rows,
            256,
        )
        torch.sort(
            self.priorities[:count],
            stable=True,
            dim=-1,
            out=(self.sorted_priorities[:count], self.victims[:count]),
        )
        _assign[(count,)](
            table,
            request_slots,
            self.source_map,
            self.sources,
            self.physical,
            self.victims,
            self.union_sources,
            self.counts,
            self.union_slots,
            self.union_physical,
            spec.source_capacity,
            spec.resident_tokens,
            table.shape[1],
            self.union_width,
            self.request_capacity,
            2048 if spec.max_query_routes > 1 else 1024,
        )
        _encode[grid](
            ids,
            query_ends,
            lengths,
            table,
            request_slots,
            self.source_map,
            self.physical,
            self.topk,
            self.slots,
            self.membership,
            spec.token_topk,
            spec.source_capacity,
            spec.resident_tokens,
            table.shape[1],
            self.union_width,
            self.request_capacity,
            rows,
            mutable_tail,
            host_tokens,
            256,
        )
        return self.topk[:rows], self.slots[:rows], self.membership[:rows]

    def commit(self, request_slots, query_ends, row_lengths, *, mutable_tail=12):
        if not 0 <= mutable_tail <= 32:
            raise ValueError("Mutable tail must fit the reserved 32-token table")
        rows = row_lengths.numel()
        if mutable_tail:
            _mutable_presence[(rows, mutable_tail)](
                self.topk,
                row_lengths,
                self.mutable_presence,
                self.spec.token_topk,
                mutable_tail,
                4096,
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
        )
