"""CPU row plans for same-offset Prefill and independent HiCache allocations."""

import operator

import numpy as np
import torch


def mapped_row_runs(
    source_rows, destination_rows, source_capacity, destination_capacity
):
    """Return contiguous CPU int64 [source row, destination row, count] runs.

    For native ``prefill_dma_copy(..., to_host=True)``, source rows/capacity
    describe device_tensor and destination rows/capacity describe host_metadata.
    For ``to_host=False``, source describes Host and destination describes device.
    Indices address rows of those 2-D tensors, not bytes or implicit page IDs;
    callers must expand page IDs if each tensor row represents only one token.

    Both index tensors must be 1-D CPU int64 with matching lengths. Row zero is
    valid, all out-of-range rows and duplicate destinations are errors, and
    repeated sources are allowed. Only pairs consecutive at both ends coalesce.
    Empty input produces [0, 3]; the returned plan owns its metadata. Callers
    retain the registered Host allocation until the DMA stream has drained.
    """
    for rows in (source_rows, destination_rows):
        if rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 1:
            raise ValueError(
                "DMA row planning requires one-dimensional CPU int64 indices"
            )
    if source_rows.numel() != destination_rows.numel():
        raise ValueError("DMA source and destination row counts must match")
    capacities = []
    for capacity in (source_capacity, destination_capacity):
        try:
            capacity = operator.index(capacity)
        except TypeError as exc:
            raise ValueError(
                "DMA row capacities must be positive int64 integers"
            ) from exc
        if not 0 < capacity <= np.iinfo(np.int64).max:
            raise ValueError("DMA row capacities must be positive int64 integers")
        capacities.append(capacity)
    source = source_rows.numpy()
    destination = destination_rows.numpy()
    for name, values, capacity in zip(
        ("source", "destination"), (source, destination), capacities
    ):
        if np.any(values < 0) or np.any(values >= capacity):
            raise ValueError(f"DMA {name} row is outside the KV allocation")
    if np.unique(destination).size != destination.size:
        raise ValueError("DMA plan contains duplicate destination rows")
    if not source.size:
        return torch.empty((0, 3), dtype=torch.int64)
    boundaries = (
        np.flatnonzero(
            (source[1:] != source[:-1] + 1) | (destination[1:] != destination[:-1] + 1)
        )
        + 1
    )
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [source.size]))
    return torch.from_numpy(
        np.column_stack((source[starts], destination[starts], ends - starts)).astype(
            np.int64, copy=False
        )
    )


def row_runs(rows, capacity):
    """Return legacy [row, count] runs, filtering padding and repeated rows.

    Legacy row zero remains padding. Repeated same-offset copies are idempotent
    and are removed in first-occurrence order before native overlap validation.
    """
    if rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 1:
        raise ValueError("DMA row planning requires one-dimensional CPU int64 indices")
    if capacity <= 0:
        raise ValueError("DMA row capacity must be positive")
    values = rows.numpy()
    selected = values[(values > 0) & (values < capacity)]
    if not selected.size:
        return torch.empty((0, 2), dtype=torch.int64)
    _, first_occurrences = np.unique(selected, return_index=True)
    selected = selected[np.sort(first_occurrences)]
    boundaries = np.flatnonzero(selected[1:] != selected[:-1] + 1) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [selected.size]))
    return torch.from_numpy(
        np.column_stack((selected[starts], ends - starts)).astype(np.int64, copy=False)
    )


def prefill_row_plans(prefix_rows, new_rows, capacity):
    combined = torch.cat((prefix_rows, new_rows)).to(device="cpu", dtype=torch.int64)
    prefix, current = combined.split((prefix_rows.numel(), new_rows.numel()))
    return row_runs(prefix, capacity), row_runs(current, capacity)
