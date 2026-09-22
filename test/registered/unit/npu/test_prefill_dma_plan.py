"""Production DMA descriptor tests independent of NPU availability."""

import importlib.util
from pathlib import Path

import pytest
import torch

PATH = (
    Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu/prefill_dma.py"
)
SPEC = importlib.util.spec_from_file_location("prefill_dma_plan_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("rows", [[], [0, -1, 32], [5, 6, 7, 1, 2, 9], [3, 3, 4, 2]])
def test_runs_preserve_fragmented_copy_rows_and_padding(rows):
    indices = torch.tensor(rows, dtype=torch.int64)
    plan = MODULE.row_runs(indices, 32)
    source = torch.arange(32 * 8).reshape(32, 8)
    target = torch.full_like(source, -1)
    for start, count in plan.tolist():
        target[start : start + count] = source[start : start + count]
    expected = torch.full_like(source, -1)
    valid = indices[(indices > 0) & (indices < 32)]
    expected[valid] = source[valid]
    assert torch.equal(target, expected)
    assert plan.shape[1] == 2 and plan.dtype == torch.int64


def test_byte_offsets_above_four_gib_are_not_truncated():
    first = (1 << 32) // 1024 + 128
    plan = MODULE.row_runs(torch.arange(first, first + 256), first + 512)
    assert plan.tolist() == [[first, 256]]
    assert plan[0, 0].item() * 1024 > 1 << 32


def test_forward_plans_are_independent_of_mutating_input_indices():
    prefix = torch.tensor([8, 9, 3, 4])
    current = torch.tensor([12, 13, 0, 64])
    prefix_plan, current_plan = MODULE.prefill_row_plans(prefix, current, 64)
    prefix.fill_(0)
    current.fill_(0)
    assert prefix_plan.tolist() == [[8, 2], [3, 2]]
    assert current_plan.tolist() == [[12, 2]]


@pytest.mark.parametrize("rows", [torch.ones(2), torch.ones((2, 2), dtype=torch.int64)])
def test_invalid_metadata_rejected(rows):
    with pytest.raises(ValueError, match="CPU int64"):
        MODULE.row_runs(rows, 16)
