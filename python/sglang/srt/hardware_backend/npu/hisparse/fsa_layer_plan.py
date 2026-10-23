"""Triton materialization of per-layer sparse selection plans on NPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_plan_kernel(
    keys,
    keys_row_stride,
    keys_col_stride,
    slots,
    slots_row_stride,
    slots_col_stride,
    applied,
    applied_row_stride,
    applied_col_stride,
    membership,
    membership_row_stride,
    membership_col_stride,
    cacheable,
    cacheable_row_stride,
    cacheable_col_stride,
    safe_out,
    miss_out,
    rows,
    topk,
    capacity,
    control_offset,
    HAS_CACHEABLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    active = (row < rows) & (col < topk)

    key = tl.load(keys + row * keys_row_stride + col * keys_col_stride, active, other=0).to(tl.int64)
    slot = tl.load(slots + row * slots_row_stride + col * slots_col_stride, active, other=0).to(tl.int64)
    safe = tl.minimum(tl.maximum(slot, 0), capacity - 1)
    valid = active & (key >= 0) & (slot >= 0) & (slot < capacity)
    previous = tl.load(
        applied + row * applied_row_stride + safe * applied_col_stride,
        active,
        other=-1,
    ).to(tl.int64)
    hit = valid & (previous == key)
    if HAS_CACHEABLE:
        stable = tl.load(
            cacheable + row * cacheable_row_stride + col * cacheable_col_stride,
            active,
            other=False,
        ).to(tl.int1)
        hit = hit & stable
    miss = valid & ~hit
    encoded = tl.where(hit, safe + 1, -(safe + 1))
    encoded = tl.where(valid, encoded, 0).to(tl.int16)
    tl.store(
        membership
        + row * membership_row_stride
        + (control_offset - topk + col) * membership_col_stride,
        encoded,
        active,
    )
    tl.store(safe_out + row * topk + col, safe, active)
    tl.store(miss_out + row * topk + col, miss, active)

    # The first tile of each row owns its ABI control header.
    control = control_offset + col
    control_mask = (tile == 0) & (col < 8) & (row < rows)
    values = tl.where(
        col == 0,
        0x5A4D,
        tl.where(
            col == 1,
            0x5A45,
            tl.where(
                col == 2,
                topk,
                tl.where(
                    col == 3,
                    control_offset - topk,
                    tl.where(
                        col == 4,
                        row,
                        tl.where(col == 5, 0x5A44, tl.where(col == 6, capacity, 0)),
                    ),
                ),
            ),
        ),
    ).to(tl.int16)
    tl.store(
        membership + row * membership_row_stride + control * membership_col_stride,
        values,
        control_mask,
    )


def build_layer_plan(
    physical_keys,
    slots,
    applied_keys,
    membership,
    *,
    capacity,
    cacheable_mask=None,
    control_offset=16384,
):
    """Materialize one layer plan, returning ``(safe_slots, miss_mask)``."""
    if physical_keys.device.type != "npu":
        raise ValueError("fused layer plan requires NPU tensors")
    if physical_keys.dim() != 2 or slots.shape != physical_keys.shape:
        raise ValueError("layer plan keys and slots must have the same rank-2 shape")
    rows, topk = physical_keys.shape
    if topk <= 0 or topk > 2051 or capacity <= 0 or capacity > 32767:
        raise ValueError("layer plan geometry exceeds the int16 metadata ABI")
    if (
        membership.dim() != 2
        or membership.shape[0] < rows
        or control_offset < topk
        or control_offset + 8 > membership.shape[1]
    ):
        raise ValueError("layer plan membership does not contain the control ABI")
    if applied_keys.dim() != 2 or applied_keys.shape[0] < rows or applied_keys.shape[1] < capacity:
        raise ValueError("layer plan applied-key table is too small")
    if cacheable_mask is not None and (
        cacheable_mask.shape != physical_keys.shape
        or cacheable_mask.dtype is not torch.bool
        or cacheable_mask.device != physical_keys.device
    ):
        raise ValueError("cacheable selection mask must match the physical keys")
    if any(t.device != physical_keys.device for t in (slots, applied_keys, membership)):
        raise ValueError("fused layer plan tensors must share one NPU device")

    safe = torch.empty_like(slots, memory_format=torch.contiguous_format)
    miss = torch.empty_like(
        physical_keys, dtype=torch.bool, memory_format=torch.contiguous_format
    )
    if rows == 0:
        return safe, miss
    # Bound temporary vectors to fit Ascend UB even for the maximum TopK.
    block = min(256, max(8, triton.next_power_of_2(topk)))
    cacheable = cacheable_mask if cacheable_mask is not None else physical_keys
    _layer_plan_kernel[(rows, triton.cdiv(topk, block))](
        physical_keys,
        physical_keys.stride(0),
        physical_keys.stride(1),
        slots,
        slots.stride(0),
        slots.stride(1),
        applied_keys,
        applied_keys.stride(0),
        applied_keys.stride(1),
        membership,
        membership.stride(0),
        membership.stride(1),
        cacheable,
        cacheable.stride(0),
        cacheable.stride(1),
        safe,
        miss,
        rows,
        topk,
        capacity,
        control_offset,
        HAS_CACHEABLE=cacheable_mask is not None,
        BLOCK=block,
    )
    return safe, miss
