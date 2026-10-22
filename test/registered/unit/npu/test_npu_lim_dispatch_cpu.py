from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytest import main as run_pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NPU = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/hardware_backend/npu/hisparse"
)
PREFIX = "sglang.srt.hardware_backend.npu.hisparse."


def load(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(PREFIX + name, NPU / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    written_locations = []
    load(monkeypatch, "lim_copy_sfa_contract")
    load(monkeypatch, "fused_sparse_attention")
    adapter = load(monkeypatch, "fsa_backend")

    class Indexer:
        def __init__(self, spec, rows, device):
            self.raw_scores = torch.empty(
                rows, spec.source_capacity // 4, dtype=torch.int32
            )

    class Planner:
        def __init__(self, spec, request_capacity, batch_capacity, device, **kwargs):
            self.topk = torch.full(
                (batch_capacity * spec.max_query_routes, 2051), -1, dtype=torch.int32
            )

        def select_and_plan(self, query, *args, **kwargs):
            calls.append("select")
            return self.topk[: query.shape[0]], None, None

        def plan(self, topk, *args, **kwargs):
            self.topk[: topk.shape[0]].copy_(topk)
            calls.append("plan")

    class Copy:
        def __init__(self, *args):
            pass

        def forward(self, query, *args, **kwargs):
            assert args[0] is host
            assert args[7] is metadata.host_page_table
            calls.append(("copy", kwargs["reuse_plan"]))
            return query + 1

    for name, symbols in {
        "lim_copy_sfa_indexer": {"NativePooledIndexer": Indexer},
        "lim_copy_sfa_native_plan": {"NativeIndexerManager": Planner},
        "lim_copy_sfa_attention": {"NativeCopySparseAttention": Copy},
        "lim_copy_sfa_reuse": {
            "publish_draft_plan": lambda *args, **kwargs: calls.append("publish")
        },
    }.items():
        monkeypatch.setitem(sys.modules, PREFIX + name, SimpleNamespace(**symbols))
    module = load(monkeypatch, "lim_copy_sfa_backend")
    pool = SimpleNamespace(req_generation=torch.tensor([0, 1, 2]))
    state = module.LimCopySfaState(
        "cpu",
        pool,
        source_tokens=8192,
        routes=6,
        capacity=12416,
        batch_capacity=2,
        layer_anchors={},
        mutable_tail=12,
    )
    host = torch.empty(8192, 512, dtype=torch.bfloat16)
    region = SimpleNamespace(tensor=host)
    binding = SimpleNamespace(mapping_epoch=0)
    backing = SimpleNamespace(regions=[(region, None)], page_binding=binding)

    def write_current(layer, locations, key, rope):
        written_locations.append(locations)
        calls.append("write")

    kv_pool = SimpleNamespace(
        start_layer=0,
        decode_host_backing=backing,
        get_index_k_with_scale_buffer=lambda layer: host,
        get_kv_buffer=lambda layer: (host, torch.empty(8192, 0)),
        set_kv_buffer=write_current,
    )
    metadata = SimpleNamespace(
        host_page_table=torch.arange(64).int().repeat(2, 1),
        real_page_table=torch.full((2, 64), -1, dtype=torch.int32),
        cache_seqlens_int32=torch.tensor([4096, 4096], dtype=torch.int32),
    )
    backend = SimpleNamespace(
        fused_sparse_state=state,
        fused_sparse_topk=2051,
        token_to_kv_pool=kv_pool,
        forward_metadata=metadata,
        _query_lens_device=lambda batch: torch.ones(2, dtype=torch.int32),
    )
    batch = SimpleNamespace(
        positions=torch.tensor([4095, 4095]),
        req_pool_indices=torch.tensor([1, 2]),
        out_cache_loc=torch.tensor([128, 256]),
        forward_mode=SimpleNamespace(
            is_decode=lambda: True,
            is_decode_or_idle=lambda: True,
            is_extend_without_speculative=lambda: False,
        ),
        reuse_dsa_topk_indices=True,
        draft_sparse_plan_cache={},
    )
    return SimpleNamespace(
        module=module,
        adapter=adapter,
        state=state,
        backend=backend,
        batch=batch,
        host=host,
        region=region,
        binding=binding,
        calls=calls,
        written_locations=written_locations,
    )


def test_lim_selection_and_attention_share_one_plan(runtime):
    query = torch.zeros(2, 1, 512)
    selected = runtime.state.select(
        runtime.backend, runtime.batch, 0, query, None, None, None
    )
    runtime.batch.draft_sparse_plan_cache = None
    output = runtime.adapter.forward_fsa(
        runtime.backend,
        query,
        query,
        None,
        SimpleNamespace(layer_id=0, scaling=0.5),
        runtime.batch,
        True,
        None,
        None,
        selected,
    )
    torch.testing.assert_close(output, query + 1)
    assert runtime.calls == ["select", "write", ("copy", False)]
    assert len(runtime.written_locations) == 1
    assert runtime.written_locations[0] is runtime.batch.out_cache_loc
    assert not runtime.state.pending


@pytest.mark.parametrize("routes", [1, 6])
def test_lim_metadata_keeps_causal_lengths_and_inactive_requests(runtime, routes):
    runtime.backend._query_lens_device = lambda batch: torch.full(
        (2,), routes, dtype=torch.int32
    )
    runtime.batch.positions = torch.arange(routes).repeat(2) + 2046
    runtime.batch.req_pool_indices[1] = 0
    ends, lengths, requests, table = runtime.state.metadata(
        runtime.backend, runtime.batch, routes * 2
    )
    assert ends.tolist() == [routes, routes * 2]
    assert lengths[:routes].tolist() == list(range(2047, 2047 + routes))
    assert not lengths[routes:].any()
    assert requests.tolist() == [1, 0]
    assert table is runtime.backend.forward_metadata.host_page_table


@pytest.mark.parametrize(
    "change", [None, "mapping", "generation", "seed", "requests", "table", "batch"]
)
@pytest.mark.parametrize("inference", [False, True])
def test_lim_seed_reuse_invalidates_identity_changes(runtime, change, inference):
    with torch.inference_mode(inference):
        query = torch.zeros(2, 1, 512)
        seed = torch.arange(2051, dtype=torch.int32).repeat(2, 1)
        layer = SimpleNamespace(layer_id=0, scaling=0.5)
        for _ in range(2):
            runtime.state.forward(
                runtime.backend,
                query,
                layer,
                runtime.batch,
                seed,
                runtime.host,
                runtime.region,
            )
            runtime.batch.positions.add_(1)
        first_two = ["plan", ("copy", False), "publish", ("copy", True)]
        assert runtime.calls == first_two
        if change == "mapping":
            runtime.binding.mapping_epoch += 1
        elif change == "generation":
            runtime.state.pool.req_generation.add_(1)
            runtime.state.refresh_generations()
        elif change == "seed":
            seed = seed.clone()
        elif change == "requests":
            runtime.batch.req_pool_indices.copy_(torch.tensor([2, 1]))
        elif change == "table":
            runtime.backend.forward_metadata.host_page_table.add_(1)
        elif change == "batch":
            runtime.batch = SimpleNamespace(**vars(runtime.batch))
        runtime.state.forward(
            runtime.backend,
            query,
            layer,
            runtime.batch,
            seed,
            runtime.host,
            runtime.region,
        )
        final_step = (
            [("copy", True)] if change is None else ["plan", ("copy", False), "publish"]
        )
        assert runtime.calls == first_two + final_step
        copies = [call for call in runtime.calls if isinstance(call, tuple)]
        assert copies == [("copy", False), ("copy", True), ("copy", change is None)]
        assert runtime.calls.count("plan") == (1 if change is None else 2)
        assert runtime.calls.count("publish") == (1 if change is None else 2)


def test_lim_generations_follow_monotonic_pool_clear(runtime):
    pointer = runtime.state.generations.data_ptr()
    runtime.state.pool.req_generation.add_(3)
    runtime.state.refresh_generations()
    assert runtime.state.generations.tolist() == [3, 4, 5]
    assert runtime.state.generations.data_ptr() == pointer


def test_hybrid_sparse_layers_keep_model_and_pool_ids_separate(runtime):
    sparse = SimpleNamespace(start_layer=0, layer_num=2)
    owner = SimpleNamespace(
        full_kv_pool=sparse,
        full_attention_layer_id_mapping={3: 0, 7: 1},
        _transfer_full_attention_id={3: 0, 7: 1}.__getitem__,
    )
    assert runtime.adapter.sparse_layer_ids(owner) == (3, 7)
    assert runtime.adapter.sparse_pool_and_layer(owner, 7) == (sparse, 1)


@pytest.mark.parametrize("explicit", [False, True])
def test_disabled_fusion_does_not_load_native_operators(runtime, monkeypatch, explicit):
    flags = SimpleNamespace(
        SGLANG_NPU_DSA_OFFLOAD_BACKEND=SimpleNamespace(
            get=lambda: "lim_copy_sfa", is_set=lambda: explicit
        ),
        SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=SimpleNamespace(get=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", SimpleNamespace(envs=flags))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(get_disagg=lambda: SimpleNamespace(disaggregation_mode="null")),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.hardware_backend.npu.fsa_native", None)
    backend = SimpleNamespace()
    if explicit:
        with pytest.raises(ValueError, match="requires fused sparse attention"):
            runtime.adapter.initialize_fsa_backend(backend, None)
    else:
        runtime.adapter.initialize_fsa_backend(backend, None)
        assert backend.fused_sparse_state is None


@pytest.mark.parametrize("routes", [1, 6])
def test_lim_selector_uses_current_kernel_path_and_expands_query_tables(
    runtime, monkeypatch, routes
):
    built = []

    def pooled_table(table, kpool):
        built.append((table.clone(), kpool))
        return table[:, :1]

    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.attention.kpool_bf16_index",
        SimpleNamespace(build_pooled_page_table_64=pooled_table),
    )
    runtime.backend._query_lens_device = lambda batch: torch.full(
        (2,), routes, dtype=torch.int32
    )
    runtime.batch.positions = torch.arange(routes).repeat(2) + 4095
    runtime.batch.forward_mode.is_decode_or_idle = lambda: routes == 1
    tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    metadata = SimpleNamespace(
        get_page_table_64=lambda: tables,
        get_token_to_batch_idx=lambda: torch.arange(2).repeat_interleave(routes),
    )
    result = runtime.module.select_offloaded_topk(
        runtime.backend,
        SimpleNamespace(index_kpool=4, n_heads=1),
        runtime.batch,
        0,
        torch.zeros(routes * 2, 1, 128),
        torch.ones(routes * 2, 1),
        metadata,
    )
    assert result.shape == (routes * 2, 2051)
    assert torch.equal(built[0][0], tables.repeat_interleave(routes, dim=0))
    assert built[0][1] == 4
    assert runtime.calls == ["select"]


if __name__ == "__main__":
    sys.exit(run_pytest([__file__, *sys.argv[1:]]))
