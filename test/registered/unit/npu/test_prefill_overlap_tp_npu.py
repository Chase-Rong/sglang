"""Shared Host visibility across pipelined Prefill generations under torchrun."""

import os

import pytest
import torch

from test_fsa_shared_tp_npu import shared_group
from test_fsa_shared_tp_npu import tp as tp_fixture
from test_prefill_overlap_npu import (
    exercise_pipelined_overlap,
    exercise_pipelined_settled_restore,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_SHARED_TP") != "1",
    reason="requires torchrun on explicitly selected idle NPUs",
)

tp = tp_fixture


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("h2d,d2h", [(True, False), (False, True), (True, True)])
def test_shared_prefill_pipeline(tp, dtype, h2d, d2h):
    exercise_pipelined_overlap(
        dtype,
        tp.device_id,
        h2d_prefetch=h2d,
        d2h_async=d2h,
        shared_group=shared_group(tp),
    )


def test_shared_empty_write_generation(tp):
    exercise_pipelined_settled_restore(
        torch.bfloat16,
        tp.device_id,
        h2d_prefetch=True,
        d2h_async=True,
        shared_group=shared_group(tp),
    )
