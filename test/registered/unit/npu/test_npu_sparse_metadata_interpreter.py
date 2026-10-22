from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from pytest import main as run_pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="requires the CPU Triton interpreter",
)
NPU = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/hardware_backend/npu/hisparse"
)


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"p4_interpreter_{name}", NPU / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("length", [2047, 2048, 2049, 2051])
def test_lim_expansion_preserves_all_2051_columns(length):
    module = load("lim_copy_sfa_indexer")
    pools = torch.arange(512, dtype=torch.int32).flip(0).repeat(6, 1)
    lengths = torch.tensor(
        [length, length, length, length, length, 0], dtype=torch.int32
    )
    output = torch.empty(6, 2051, dtype=torch.int32)
    module._expand[(6, 9)](pools, lengths, output, 256)
    expected = pools.repeat_interleave(4, dim=1) * 4 + torch.arange(2048).remainder(4)
    tail = (length // 4 * 4 + torch.arange(3)).repeat(6, 1)
    expected = torch.cat((expected, tail), dim=1)
    if length <= 2048:
        expected[:] = torch.arange(2051)
    expected[expected >= length] = -1
    expected[-1] = -1
    assert torch.equal(output, expected)


def test_draft_membership_reorders_columns_without_moving_causal_coordinates():
    module = load("lim_copy_sfa_reuse")
    width, capacity, routes = 2051, 4096, 6
    order = torch.cat((torch.arange(2048).roll(37), torch.arange(2048, 2051)))
    tokens = order.int().repeat(routes, 1)
    slots = torch.arange(width, dtype=torch.int32).repeat(routes, 1)
    membership = torch.full((routes, 16400), -7, dtype=torch.int16)
    table = torch.arange(1, 18, dtype=torch.int32).repeat(routes, 1)
    requests = torch.arange(1, routes + 1, dtype=torch.int32)
    lengths = torch.full((routes,), 2051, dtype=torch.int32)
    physical = torch.full((routes + 1, capacity), -1, dtype=torch.int64)
    physical[1:, :width] = tokens.long() + 128
    physical[1:, 7] = -1
    state = torch.full((routes + 1, 16), -1, dtype=torch.int64)
    union_sources = torch.empty(routes, capacity, dtype=torch.int32)
    union_slots = torch.empty_like(union_sources)
    union_physical = torch.empty(routes, capacity, dtype=torch.int64)
    module._publish_draft_plan[(routes,)](
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
        capacity,
        routes + 1,
        table.shape[1],
        capacity,
        4096,
        12,
        256,
    )
    misses = tokens >= 2039
    misses[:, 7] = True
    assert torch.equal(membership[:, 14333:16384] < 0, misses)
    for row in range(routes):
        columns = torch.arange(2048)
        expected = torch.cat(
            (columns[misses[row, :2048]], columns[~misses[row, :2048]])
        )
        remap = membership[row, 4096:6144].long()
        assert torch.equal(remap, expected)
        assert torch.equal(tokens[row, remap] < 2047, (tokens[row] < 2047)[expected])
        assert membership[row, 16391].item() == misses[row].sum().item()
        assert membership[row, 16394].item() == misses[row, :2048].sum().item()


def test_external_metadata_reads_six_routes_and_inactive_padding():
    module = load("fsa_metadata")
    rows, width = 18, 2051
    topk = torch.arange(width, dtype=torch.int32).repeat(rows, 1)
    table = torch.stack(
        (torch.arange(1, 18), torch.arange(18, 35), torch.zeros(17))
    ).int()
    lengths = torch.tensor([2049, 2051, 0], dtype=torch.int32)
    ends = torch.tensor([6, 12, 18], dtype=torch.int32)
    requests = torch.tensor([2, 1, 0], dtype=torch.int32)
    generations = torch.tensor([0, 3, 7], dtype=torch.int64)
    positions = torch.cat(
        (torch.arange(2043, 2049), torch.arange(2045, 2051), torch.zeros(6))
    ).int()
    logical = torch.empty_like(topk)
    physical = torch.empty_like(topk, dtype=torch.int64)
    owners = torch.empty(rows, dtype=torch.int64)
    row_lengths = torch.empty(rows, dtype=torch.int32)
    cacheable = torch.empty_like(topk, dtype=torch.bool)
    module._external_metadata_kernel[(rows,)](
        topk,
        table,
        lengths,
        ends,
        requests,
        generations,
        positions,
        logical,
        physical,
        owners,
        row_lengths,
        cacheable,
        K=width,
        BATCH=3,
        PAGES=17,
        TABLE_STRIDE=17,
        PAGE_SIZE=128,
        TOTAL=35 * 128,
        NUM_SLOTS=3,
        TAIL=12,
        HAS_POSITIONS=True,
        BLOCK_K=4096,
        BLOCK_BATCH=4,
    )
    for row in range(12):
        request = row // 6
        limit = positions[row].item() + 1
        assert torch.equal(logical[row, :limit], torch.arange(limit).int())
        assert (logical[row, limit:] == -1).all()
        offset = (1 if request == 0 else 18) * 128
        assert torch.equal(physical[row, :limit], torch.arange(limit) + offset)
        assert owners[row].item() == (23 if request == 0 else 10)
        assert cacheable[row].sum().item() == lengths[request].item() - 12
    assert (logical[12:] == -1).all()
    assert (physical[12:] == -1).all()
    assert (owners[12:] == -1).all()
    assert not cacheable[12:].any()


if __name__ == "__main__":
    sys.exit(run_pytest([__file__, *sys.argv[1:]]))
