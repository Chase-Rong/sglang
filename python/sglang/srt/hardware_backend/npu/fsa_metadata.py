"""Device metadata for the stream-ordered external sparse planner."""

import torch
import triton
import triton.language as tl


@triton.jit
def _external_metadata_kernel(
    TopK,
    Table,
    Lengths,
    Cumulative,
    Slots,
    Generations,
    Positions,
    Logical,
    Physical,
    Owners,
    RowLengths,
    Cacheable,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    PAGES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOTAL: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    TAIL: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.arange(0, BLOCK_BATCH)
    ends = tl.load(Cumulative + b, b < BATCH, other=2147483647)
    request = tl.sum(((b < BATCH) & (ends <= row)).to(tl.int32), 0)
    valid_row = request < BATCH
    request = tl.minimum(request, BATCH - 1)
    length = tl.load(Lengths + request)
    end = tl.load(Cumulative + request)
    start = tl.load(Cumulative + request - 1, request > 0, other=0)
    slot = tl.load(Slots + request)
    valid_row = valid_row & (slot > 0) & (slot < NUM_SLOTS) & (length > 0)
    safe_slot = tl.minimum(tl.maximum(slot, 0), NUM_SLOTS - 1)
    generation = tl.load(Generations + safe_slot).to(tl.int64)
    owner = tl.where(valid_row, generation * NUM_SLOTS + safe_slot, -1)
    tl.store(Owners + row, owner)
    tl.store(RowLengths + row, length)
    limit = tl.minimum(length - (end - start) + row - start + 1, length)
    if HAS_POSITIONS:
        position = tl.load(Positions + row)
        limit = tl.minimum(limit, position + 1)
    k = tl.arange(0, BLOCK_K)
    logical = tl.load(TopK + row * K + k, k < K, other=-1).to(tl.int32)
    valid = valid_row & (logical >= 0) & (logical < limit)
    safe_logical = tl.maximum(logical, 0).to(tl.int64)
    page = safe_logical // PAGE_SIZE
    physical_page = tl.load(
        Table + request.to(tl.int64) * TABLE_STRIDE + page,
        (k < K) & valid & (page < PAGES),
        other=-1,
    ).to(tl.int64)
    physical = physical_page * PAGE_SIZE + safe_logical % PAGE_SIZE
    valid = valid & (page < PAGES) & (physical_page >= 0) & (physical < TOTAL)
    # The reference translation retains a wrapped physical ID, but the caller
    # masks its logical ID and cacheability if int64 page arithmetic is negative.
    logical = tl.where(valid & (physical >= 0), logical, -1)
    tl.store(Logical + row * K + k, logical, k < K)
    tl.store(Physical + row * K + k, tl.where(valid, physical, -1), k < K)
    if TAIL:
        stable = valid & (logical >= 0) & (logical < tl.maximum(length - TAIL, 0))
        tl.store(Cacheable + row * K + k, stable, k < K)


def build_external_metadata_plan(
    topk,
    block_table,
    kv_lengths,
    query_lengths,
    request_slots,
    generations,
    *,
    positions=None,
    page_size,
    full_token_count,
    mutable_tail_tokens=0,
):
    """Return logical IDs, physical IDs, owners, row lengths and tail validity.

    Called with normalized contiguous TopK and per-request vectors. All dynamic
    values are read by the captured kernel on replay; no cross-forward cache is
    used. Table row stride may include unused graph-capacity columns.
    """
    if topk.dim() == 3 and topk.shape[1] == 1:
        topk = topk[:, 0]
    if topk.dim() != 2 or not topk.is_contiguous():
        raise ValueError("external metadata TopK must be contiguous [T,K] or [T,1,K]")
    rows, width = topk.shape
    batch = query_lengths.numel()
    if min(rows, width, batch, generations.numel()) <= 0 or page_size <= 0:
        raise ValueError("external metadata requires positive row and page geometry")
    if (
        block_table.dim() != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
        or block_table.stride(1) != 1
    ):
        raise ValueError(
            "external metadata requires a page table with contiguous columns"
        )
    tensors = (topk, block_table, kv_lengths, query_lengths, request_slots, generations)
    if any(t.device != topk.device for t in tensors) or topk.device.type != "npu":
        raise ValueError("external metadata tensors must share one NPU device")
    if any(t.dim() != 1 or not t.is_contiguous() for t in tensors[2:]):
        raise ValueError("external metadata request vectors must be contiguous")
    if kv_lengths.numel() != batch or request_slots.numel() != batch:
        raise ValueError("external metadata request counts must match")
    if positions is not None and (
        positions.dim() != 1
        or positions.numel() != rows
        or positions.device != topk.device
        or not positions.is_contiguous()
    ):
        raise ValueError(
            "sparse query positions must have one logical position per query row"
        )
    logical = torch.empty((rows, 1, width), dtype=torch.int32, device=topk.device)
    physical = torch.empty((rows, width), dtype=torch.int64, device=topk.device)
    owners = torch.empty(rows, dtype=torch.int64, device=topk.device)
    lengths = torch.empty(rows, dtype=torch.int32, device=topk.device)
    cacheable = (
        torch.empty((rows, width), dtype=torch.bool, device=topk.device)
        if mutable_tail_tokens
        else None
    )
    _external_metadata_kernel[(rows,)](
        topk,
        block_table,
        kv_lengths,
        query_lengths,
        request_slots,
        generations,
        positions,
        logical,
        physical,
        owners,
        lengths,
        cacheable,
        K=width,
        BATCH=batch,
        PAGES=block_table.shape[1],
        TABLE_STRIDE=block_table.stride(0),
        PAGE_SIZE=page_size,
        TOTAL=full_token_count,
        NUM_SLOTS=generations.numel(),
        TAIL=mutable_tail_tokens,
        HAS_POSITIONS=positions is not None,
        BLOCK_K=triton.next_power_of_2(width),
        BLOCK_BATCH=triton.next_power_of_2(batch),
    )
    return logical, physical, owners, lengths, cacheable
