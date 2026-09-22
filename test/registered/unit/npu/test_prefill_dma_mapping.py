"""CPU mapping contracts; optional native checks use the existing NPU test gate."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

PATH = (
    Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu/prefill_dma.py"
)
SPEC = importlib.util.spec_from_file_location("prefill_dma_mapping_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plan_rows(source, destination, source_capacity=32, destination_capacity=64):
    return MODULE.mapped_row_runs(
        torch.tensor(source, dtype=torch.int64),
        torch.tensor(destination, dtype=torch.int64),
        source_capacity,
        destination_capacity,
    )


@pytest.mark.parametrize(
    "source,destination,expected",
    [
        ([], [], []),
        ([0], [0], [[0, 0, 1]]),
        ([1, 2, 3], [20, 21, 22], [[1, 20, 3]]),
        ([1, 2, 3], [20, 23, 24], [[1, 20, 1], [2, 23, 2]]),
        ([1, 4, 5], [20, 21, 22], [[1, 20, 1], [4, 21, 2]]),
        ([5, 6, 1, 2], [30, 31, 0, 1], [[5, 30, 2], [1, 0, 2]]),
        ([3, 3, 4], [0, 1, 2], [[3, 0, 1], [3, 1, 2]]),
        ([31], [63], [[31, 63, 1]]),
    ],
)
def test_coalescing_requires_both_ends_contiguous(source, destination, expected):
    plan = plan_rows(source, destination)
    assert plan.tolist() == expected
    assert plan.shape == (len(expected), 3)
    assert plan.dtype == torch.int64 and plan.device.type == "cpu"
    assert plan.is_contiguous()


@pytest.mark.parametrize("to_host", [False, True])
def test_fragmented_page_mapping_matches_indexed_copy(to_host):
    page_size = 4
    host_rows = (
        torch.tensor([0, 5, 2])[:, None] * page_size + torch.arange(page_size)
    ).flatten()
    device_rows = (
        torch.tensor([3, 1, 6])[:, None] * page_size + torch.arange(page_size)
    ).flatten()
    host = torch.arange(24 * 8).reshape(24, 8) + 10000
    device = torch.arange(32 * 8).reshape(32, 8)
    source, destination = (device, host) if to_host else (host, device)
    source_rows, destination_rows = (
        (device_rows, host_rows) if to_host else (host_rows, device_rows)
    )
    expected = destination.clone()
    expected[destination_rows] = source[source_rows]
    plan = MODULE.mapped_row_runs(
        source_rows, destination_rows, source.size(0), destination.size(0)
    )
    assert plan.size(0) == 3
    for source_row, destination_row, count in plan.tolist():
        destination[destination_row : destination_row + count] = source[
            source_row : source_row + count
        ]
    assert torch.equal(destination, expected)


def test_random_mapping_preserves_order_and_untouched_rows():
    generator = torch.Generator().manual_seed(20260921)
    source = torch.arange(17 * 5).reshape(17, 5)
    for count in range(34):
        source_rows = torch.randint(17, (count,), generator=generator)
        destination_rows = torch.randperm(41, generator=generator)[:count]
        plan = MODULE.mapped_row_runs(source_rows, destination_rows, 17, 41)
        actual = torch.full((41, 5), -1)
        expected = actual.clone()
        expected[destination_rows] = source[source_rows]
        expanded = []
        for source_row, destination_row, length in plan.tolist():
            actual[destination_row : destination_row + length] = source[
                source_row : source_row + length
            ]
            expanded.extend(
                (source_row + offset, destination_row + offset)
                for offset in range(length)
            )
        assert expanded == list(zip(source_rows.tolist(), destination_rows.tolist()))
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "source,destination,match",
    [
        ([1], [], "counts must match"),
        ([-1], [1], "source row is outside"),
        ([32], [1], "source row is outside"),
        ([1], [-1], "destination row is outside"),
        ([1], [64], "destination row is outside"),
        ([1, 2], [3, 3], "duplicate destination"),
        ([1, 1], [3, 3], "duplicate destination"),
        ([1, 2, 3], [3, 4, 3], "duplicate destination"),
    ],
)
def test_invalid_mapping_is_rejected_without_filtering(source, destination, match):
    with pytest.raises(ValueError, match=match):
        plan_rows(source, destination)


@pytest.mark.parametrize("capacity", [0, -1, 1.5, 1 << 63])
@pytest.mark.parametrize("source_side", [False, True])
def test_invalid_capacity_even_for_empty_work(capacity, source_side):
    capacities = (capacity, 64) if source_side else (32, capacity)
    with pytest.raises(ValueError, match="positive int64"):
        plan_rows([], [], *capacities)


@pytest.mark.parametrize("source_side", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        torch.tensor([1.0]),
        torch.tensor([1], dtype=torch.int32),
        torch.ones((1, 1), dtype=torch.int64),
        torch.tensor(1),
        torch.empty(1, dtype=torch.int64, device="meta"),
    ],
)
def test_metadata_contract(invalid, source_side):
    valid = torch.tensor([1])
    rows = (invalid, valid) if source_side else (valid, invalid)
    with pytest.raises(ValueError, match="CPU int64"):
        MODULE.mapped_row_runs(*rows, 32, 64)


def test_strided_input_plan_owns_metadata():
    source = torch.tensor([2, 99, 3, 99, 9, 99])[::2]
    destination = torch.tensor([0, 99, 1, 99, 6, 99])[::2]
    plan = MODULE.mapped_row_runs(source, destination, 16, 8)
    source.fill_(0)
    destination.fill_(0)
    assert plan.tolist() == [[2, 0, 2], [9, 6, 1]]


def test_independent_offsets_are_int64_without_overflow():
    maximum = torch.iinfo(torch.int64).max
    plan = plan_rows(
        [maximum - 3, maximum - 2, maximum - 1],
        [(1 << 33), (1 << 33) + 1, (1 << 33) + 2],
        maximum,
        (1 << 34),
    )
    assert plan.tolist() == [[maximum - 3, 1 << 33, 3]]


def test_legacy_plan_keeps_padding_semantics_and_removes_duplicate_writes():
    rows = torch.tensor([0, -1, 32, 3, 3, 4, 2, 4])
    assert MODULE.row_runs(rows, 32).tolist() == [[3, 2], [2, 1]]
    prefix, current = MODULE.prefill_row_plans(rows, torch.tensor([7, 8]), 32)
    assert prefix.tolist() == [[3, 2], [2, 1]]
    assert current.tolist() == [[7, 2]]


@pytest.fixture
def native_mapping():
    if os.environ.get("SGLANG_UNIFIED_HOST_TEST") != "1":
        pytest.skip("requires explicitly selected idle NPU and MemFabric A5 build")
    import torch_npu

    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.memfabric_host_kv import MemFabricHostKvPool

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    pool = MemFabricHostKvPool(device_id, required_bytes=24 * 8 * 4)
    try:
        region = pool.allocate_component(
            "mapping", (24, 8), torch.float32, page_size=1
        ).region
        device = torch.empty((32, 8), dtype=torch.float32, device=f"npu:{device_id}")
        yield native_ops(), region, device
    finally:
        torch_npu.npu.synchronize()
        pool.close(drained=True)


@pytest.mark.parametrize("to_host", [False, True])
def test_native_direction_and_legacy_compatibility(native_mapping, to_host):
    ops, region, device = native_mapping
    device_pattern = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    host_pattern = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8) + 1000
    device.copy_(device_pattern)
    region.tensor.copy_(host_pattern)
    source_rows, destination_rows = ([3, 4, 9], [0, 1, 6])
    source_capacity, destination_capacity = (32, 24) if to_host else (24, 32)
    plan = plan_rows(
        source_rows, destination_rows, source_capacity, destination_capacity
    )
    ops.prefill_dma_copy(device, region.tensor, region.device_ptr, plan, to_host)
    ops.prefill_dma_copy(
        device, region.tensor, region.device_ptr, torch.tensor([[12, 2]]), to_host
    )
    for columns in (2, 3):
        ops.prefill_dma_copy(
            device,
            region.tensor,
            region.device_ptr,
            torch.empty((0, columns), dtype=torch.int64),
            to_host,
        )
    torch.npu.synchronize()
    expected = host_pattern.clone() if to_host else device_pattern.clone()
    source = device_pattern if to_host else host_pattern
    expected[destination_rows] = source[source_rows]
    expected[12:14] = source[12:14]
    actual = region.tensor if to_host else device.cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("to_host", [False, True])
@pytest.mark.parametrize(
    "entries,match",
    [
        ([[1, 2, 0]], "outside"),
        ([[-1, 2, 1]], "outside"),
        ([[1, -1, 1]], "outside"),
        ([[32, 1, 1]], "outside"),
        ([[1, 32, 1]], "outside"),
        ([[1, 2, (1 << 63) - 1]], "outside"),
        ([[1, 4, 3], [9, 6, 2]], "duplicate destination"),
        ([[9, 6, 2], [1, 4, 3]], "duplicate destination"),
        ([[1, 2], [2, 1]], "duplicate destination"),
        ([[0, 1]], "outside"),
    ],
)
def test_native_rejects_invalid_runs_before_submission(
    native_mapping, to_host, entries, match
):
    ops, region, device = native_mapping
    with pytest.raises(RuntimeError, match=match):
        ops.prefill_dma_copy(
            device, region.tensor, region.device_ptr, torch.tensor(entries), to_host
        )
