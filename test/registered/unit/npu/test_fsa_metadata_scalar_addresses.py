"""CPU interpreter regression for Ascend scalar loads at request boundaries."""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter only"
)
def test_first_request_predecessor_is_in_bounds_and_preserves_causal_limits():
    from triton.runtime.interpreter import interpreter_builder

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/hardware_backend/npu/fsa_metadata.py"
    )
    spec = importlib.util.spec_from_file_location("fsa_metadata_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    topk = torch.tensor([[0, 3, 4], [0, 3, 4], [0, 5, 6]], dtype=torch.int32)
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    lengths = torch.tensor([5, 7], dtype=torch.int32)
    cumulative = torch.tensor([2, 3], dtype=torch.int32)
    slots = torch.tensor([1, 2], dtype=torch.int32)
    generations = torch.tensor([0, 7, 9], dtype=torch.int64)
    logical = torch.empty_like(topk)
    physical = torch.empty_like(topk, dtype=torch.int64)
    owners = torch.empty(3, dtype=torch.int64)
    row_lengths = torch.empty(3, dtype=torch.int32)
    inputs = (topk, table, lengths, cumulative, slots, generations)
    ranges = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in inputs
    ]
    original_load = interpreter_builder.create_masked_load

    def checked_load(ptrs, mask, other, *args):
        addresses = np.asarray(ptrs.data).reshape(-1)
        # Ascend can issue scalar loads even when the logical mask is false.
        active = (
            np.ones(addresses.size, dtype=bool)
            if addresses.size == 1
            else np.asarray(mask.data).reshape(-1)
        )
        width = ptrs.get_element_ty().primitive_bitwidth // 8
        for address in addresses[active]:
            assert any(
                lo <= int(address) and int(address) + width <= hi for lo, hi in ranges
            ), "read outside tensor storage"
        return original_load(ptrs, mask, other, *args)

    with patch.object(interpreter_builder, "create_masked_load", checked_load):
        module._external_metadata_kernel[(3,)](
            *inputs,
            None,
            logical,
            physical,
            owners,
            row_lengths,
            None,
            K=3,
            BATCH=2,
            PAGES=2,
            TABLE_STRIDE=2,
            PAGE_SIZE=4,
            TOTAL=20,
            NUM_SLOTS=3,
            TAIL=0,
            HAS_POSITIONS=False,
            BLOCK_K=4,
            BLOCK_BATCH=2,
        )
    assert logical.tolist() == [[0, 3, -1], [0, 3, 4], [0, 5, 6]]
    assert physical.tolist() == [[4, 7, -1], [4, 7, 8], [12, 17, 18]]
    assert owners.tolist() == [22, 22, 29]
    assert row_lengths.tolist() == [5, 5, 7]
