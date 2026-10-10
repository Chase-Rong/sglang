"""Warm Graph comparison of the current path, native pair and LIM/existing-FSA."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1"
    or not os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK"),
    reason="Requires an authorized NPU environment and benchmark output directory",
)


@pytest.mark.parametrize("batch,routes", [(1, 1), (1, 6), (10, 1), (10, 6)])
def test_three_way_warm_graph(batch, routes):
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        batch,
        routes,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        three_way=True,
    )
