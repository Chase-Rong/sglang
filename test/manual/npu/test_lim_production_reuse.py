"""Repeat the allocation transition preceding a production-geometry failure."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1"
    or not os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK"),
    reason="Requires an authorized NPU and benchmark directory",
)


def test_large_batch_to_mtp_reallocation():
    from test_lim_copy_sfa_component import run_request_shared_attention

    for _ in range(3):
        for batch, routes in ((16, 1), (1, 6)):
            run_request_shared_attention(
                batch,
                routes,
                include_indexer=True,
                native_fused=True,
                native_copy=True,
                three_way=True,
                context=131072,
                kv_length=131072,
                reserved_context=131072,
                indexer_heads=32,
                changing_topk=True,
            )
