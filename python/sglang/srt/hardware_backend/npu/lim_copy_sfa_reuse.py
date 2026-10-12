"""Post-copy plans reused only within one fixed seeded draft chain."""

import triton
import triton.language as tl


@triton.jit
def _publish_draft_plan(
    tokens,
    slots,
    membership,
    physical,
    state,
    table,
    requests,
    lengths,
    union_sources,
    union_slots,
    union_physical,
    capacity: tl.constexpr,
    request_capacity: tl.constexpr,
    columns: tl.constexpr,
    union_width: tl.constexpr,
    host_tokens: tl.constexpr,
    tail: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, block)
    owner = tl.load(requests + row)
    active = (owner > 0) & (owner < request_capacity)
    safe_owner = tl.minimum(tl.maximum(owner, 0), request_capacity - 1)
    length = tl.load(lengths + row)
    count = tl.full((), 0, tl.int32)
    body_count = tl.full((), 0, tl.int32)
    for chunk in range(tl.cdiv(2051, block)):
        offset = chunk * block + lane
        token = tl.load(tokens + row * 2051 + offset, offset < 2051, other=-1)
        slot = tl.load(slots + row * 2051 + offset, offset < 2051, other=-1)
        valid = active & (offset < 2051) & (token >= 0) & (token < length)
        valid &= (slot >= 0) & (slot < capacity) & (token // 128 < columns)
        safe_token = tl.where(valid, token, 0)
        page = tl.load(table + row * columns + safe_token // 128)
        address = page.to(tl.int64) * 128 + safe_token % 128
        valid &= (page >= 0) & (address >= 0) & (address < host_tokens)
        safe_slot = tl.minimum(tl.maximum(slot, 0), capacity - 1)
        previous = tl.load(physical + safe_owner * capacity + safe_slot)
        stable = valid & (token < length - tail) & (previous == address)
        misses = valid & ~stable
        encoded = tl.where(valid, tl.where(stable, slot + 1, -slot - 1), 0)
        tl.store(membership + row * 16400 + 14333 + offset, encoded, offset < 2051)
        ranks = tl.cumsum(misses.to(tl.int32), 0)
        destination = row * union_width + tl.maximum(count + ranks - 1, 0)
        tl.store(union_sources + destination, token, misses)
        tl.store(union_slots + destination, slot, misses)
        tl.store(union_physical + destination, address, misses)
        count += tl.sum(misses.to(tl.int32), 0)
        body_count += tl.sum((misses & (offset < 2048)).to(tl.int32), 0)
    previous_misses = tl.full((), 0, tl.int32)
    for chunk in range(tl.cdiv(2048, block)):
        offset = chunk * block + lane
        encoded = tl.load(
            membership + row * 16400 + 14333 + offset, offset < 2048, other=0
        )
        misses = (offset < 2048) & (encoded < 0)
        ranks = previous_misses + tl.cumsum(misses.to(tl.int32), 0)
        order = tl.where(misses, ranks - 1, body_count + offset - ranks)
        tl.store(
            membership + row * 16400 + 4096 + tl.maximum(order, 0),
            offset,
            offset < 2048,
        )
        previous_misses += tl.sum(misses.to(tl.int32), 0)
    tl.store(membership + row * 16400 + 16391, count)
    tl.store(membership + row * 16400 + 16394, body_count)
    tl.store(state + safe_owner * 16 + 2, 0, active)
    tl.store(state + safe_owner * 16 + 3, count, active)


def publish_draft_plan(planner, lengths, table, requests, *, host_tokens, mutable_tail):
    rows = lengths.numel()
    if rows != requests.numel() or table.shape[0] != rows:
        raise ValueError("Draft reuse requires exactly one query per request")
    _publish_draft_plan[(rows,)](
        planner.topk,
        planner.slots,
        planner.membership,
        planner.physical,
        planner.state,
        table,
        requests,
        lengths,
        planner.union_sources,
        planner.union_slots,
        planner.union_physical,
        planner.spec.resident_tokens,
        planner.request_capacity,
        table.shape[1],
        planner.union_width,
        host_tokens,
        mutable_tail,
        256,
    )
