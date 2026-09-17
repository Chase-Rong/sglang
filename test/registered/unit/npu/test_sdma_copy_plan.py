"""CPU coverage for the validation-only SDMA address plan, no hardware imports."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name("sdma_copy_lab.py")
SPEC = importlib.util.spec_from_file_location("sdma_copy_lab_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coalesced_copies_preserve_noncontiguous_row_mapping():
    sources = [8, 9, 17, 2, 3, 31]
    destinations = [1, 2, 9, 10, 11, 0]
    valid = [True, True, True, True, True, False]
    runs = MODULE.row_copy_runs(sources, destinations, valid, 32, 32, 4)
    source = bytes(range(128))
    destination = bytearray([255] * 128)
    expected = bytearray(destination)
    for start, target, length in runs:
        destination[target : target + length] = source[start : start + length]
    for source_row, destination_row, enabled in zip(sources, destinations, valid):
        if enabled:
            expected[destination_row * 4 : (destination_row + 1) * 4] = source[
                source_row * 4 : (source_row + 1) * 4
            ]
    assert destination == expected
    assert len(runs) == 3


def test_offsets_above_four_gib_do_not_truncate():
    start = (1 << 32) // 1024 + 17
    assert MODULE.row_copy_runs(
        [start, start + 1],
        [start + 3, start + 4],
        [True, True],
        start + 8,
        start + 8,
        1024,
    ) == ((start * 1024, (start + 3) * 1024, 2048),)


@pytest.mark.parametrize(
    "sources,destinations,valid",
    [
        ([], [], []),
        ([999], [-1], [False]),
    ],
)
def test_empty_or_invalid_rows_submit_no_copy(sources, destinations, valid):
    assert not MODULE.row_copy_runs(sources, destinations, valid, 8, 8, 2)


@pytest.mark.parametrize(
    "sources,destinations,valid",
    [
        ([1], [1, 2], [True]),
        ([-1], [1], [True]),
        ([8], [1], [True]),
        ([1], [8], [True]),
        ([1, 2], [3, 3], [True, True]),
    ],
)
def test_unsafe_address_plans_are_rejected(sources, destinations, valid):
    with pytest.raises(ValueError):
        MODULE.row_copy_runs(sources, destinations, valid, 8, 8, 2)


def test_native_bridge_refuses_queued_torch_mode(monkeypatch):
    monkeypatch.delenv("TASK_QUEUE_ENABLE", raising=False)
    with pytest.raises(RuntimeError, match="TASK_QUEUE_ENABLE"):
        MODULE.AclCopyBridge("unused")


@pytest.mark.parametrize(
    "sources,destinations,valid",
    [
        ([2, 3], [4, 5], [True, True]),
        ([1, 2], [3, 4], [True, True]),
        ([1, 2], [4, 5], [True, False]),
    ],
)
def test_same_length_but_different_runtime_metadata_is_rejected(
    sources, destinations, valid
):
    import torch

    fixture = MODULE.FixedPrefillSdmaFixture.__new__(MODULE.FixedPrefillSdmaFixture)
    fixture.verify_metadata = True
    with pytest.raises(ValueError):
        fixture._verify(
            torch.tensor(sources),
            torch.tensor(destinations),
            torch.tensor(valid),
            [1, 2],
            [4, 5],
        )


def test_inflight_copy_owners_cannot_be_released():
    from types import SimpleNamespace

    bridge = MODULE.AclCopyBridge.__new__(MODULE.AclCopyBridge)
    owner = object()
    bridge.pending = [owner]
    bridge.streams = {1: SimpleNamespace(query=lambda: False)}
    with pytest.raises(RuntimeError, match="completion"):
        bridge.release_after_drain()
    assert bridge.pending == [owner]
