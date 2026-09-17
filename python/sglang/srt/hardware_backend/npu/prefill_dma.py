"""Host row plans shared by all layers of one integrated Prefill forward."""

import numpy as np
import torch


def row_runs(rows, capacity):
    """Coalesce same-address KV row copies without inspecting any KV payload."""
    if rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 1:
        raise ValueError("DMA row planning requires one-dimensional CPU int64 indices")
    if capacity <= 0:
        raise ValueError("DMA row capacity must be positive")
    values = rows.numpy()
    selected = values[(values > 0) & (values < capacity)]
    if not selected.size:
        return torch.empty((0, 2), dtype=torch.int64)
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
