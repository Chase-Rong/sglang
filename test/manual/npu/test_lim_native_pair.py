"""Actual native LIM + CopySFA pair, including mutable Host backing and replay."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an explicitly authorized NPU component environment",
)


@pytest.mark.parametrize("batch,routes", [(1, 1), (1, 6), (10, 1), (10, 6)])
def test_native_pair(batch, routes):
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        batch, routes, include_indexer=True, native_fused=True, native_copy=True
    )


@pytest.mark.parametrize("batch,routes", [(1, 1), (2, 6)])
def test_framework_native_pair(batch, routes):
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        batch,
        routes,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        framework=True,
    )


def test_framework_graph_padding():
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        2,
        6,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        framework=True,
        inactive_last=True,
    )


@pytest.mark.parametrize("kv_length", [24, 32, 128, 512, 2048])
@pytest.mark.parametrize("batch,routes,framework", [(1, 1, False), (1, 6, False), (2, 6, True)])
def test_short_context_reordered_mask(batch, routes, framework, kv_length):
    """Mask the gathered KV order, including warm-cache reuse and MTP routes."""
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        batch,
        routes,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        framework=framework,
        kv_length=kv_length,
    )


def test_framework_short_prefix_in_larger_reservation():
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        1,
        1,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        framework=True,
        reserved_context=32768,
    )


@pytest.mark.parametrize("heads", [1, 32, 64])
def test_native_pair_attention_head_boundaries(heads):
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(
        2,
        6,
        include_indexer=True,
        native_fused=True,
        native_copy=True,
        attention_heads=heads,
    )
