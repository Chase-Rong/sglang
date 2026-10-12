"""Synthetic production geometry; not a captured request or serving TPOT."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1"
    or not os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK"),
    reason="Requires an authorized NPU and benchmark directory",
)


@pytest.mark.parametrize("batch", [1, 8, 16])
@pytest.mark.parametrize("routes", [1, 6])
@pytest.mark.parametrize("changing_topk", [False, True])
def test_production_geometry(batch, routes, changing_topk):
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        batch,
        routes,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        three_way=True,
        context=131072,
        reserved_context=131072,
        kv_length=131072,
        indexer_heads=32,
        changing_topk=changing_topk,
    )


@pytest.mark.parametrize(
    "batch,routes,shift,length",
    [
        (1, 1, 0, 131072),
        (1, 1, 5, 131072),
        (1, 1, 512, 131072),
        (1, 1, 128, 131068),
        (1, 1, 128, 131069),
        (8, 6, 5, 131072),
        (16, 6, 5, 131072),
    ],
)
def test_production_miss_boundaries(batch, routes, shift, length):
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        batch,
        routes,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        three_way=True,
        context=131072,
        reserved_context=131072,
        kv_length=length,
        indexer_heads=32,
        changing_topk=True,
        pooled_shift=shift,
    )
