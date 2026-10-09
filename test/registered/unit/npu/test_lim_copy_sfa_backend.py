"""Serving dispatch, seed semantics and memory accounting without NPU allocation."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[4]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu"
PREFIX = "sglang.srt.hardware_backend.npu."


def load(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(PREFIX + name, NPU / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, PREFIX + name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(monkeypatch):
    calls = []
    load("fused_sparse_attention", monkeypatch)
    load("lim_copy_sfa_contract", monkeypatch)

    class Indexer:
        def __init__(self, spec, rows, device):
            self.raw_scores = torch.empty(
                rows, spec.source_capacity // 4, dtype=torch.int32
            )

    class Planner:
        def __init__(self, *args, **kwargs):
            self.topk = torch.full((6, 2051), -1, dtype=torch.int32)

        def select_and_plan(self, query, *args, **kwargs):
            calls.append("indexer_manage")
            return self.topk[: query.shape[0]], None, None

        def plan(self, topk, *args, **kwargs):
            calls.append(("given_topk", topk.clone()))

    class Copy:
        def __init__(self, *args):
            pass

        def forward(self, query, *args, **kwargs):
            calls.append("copy_sfa")
            return query + 1

    for name, members in {
        "lim_copy_sfa_indexer": {"NativePooledIndexer": Indexer},
        "lim_copy_sfa_native_plan": {"NativeIndexerManager": Planner},
        "lim_copy_sfa_attention": {"NativeCopySparseAttention": Copy},
    }.items():
        monkeypatch.setitem(sys.modules, PREFIX + name, SimpleNamespace(**members))
    module = load("lim_copy_sfa_backend", monkeypatch)
    adapter = load("fsa_backend", monkeypatch)
    pool = SimpleNamespace(
        req_generation=torch.tensor([0, 1, 2]), req_generation_epoch=0
    )
    state = module.LimCopySfaState(
        "cpu",
        pool,
        source_tokens=4096,
        routes=1,
        capacity=4096,
        batch_capacity=2,
        layer_anchors={3: 3, 7: 3},
        mutable_tail=12,
    )
    host = torch.empty(4096, 512, dtype=torch.bfloat16)
    region = SimpleNamespace(tensor=host)
    backing = SimpleNamespace(integrated=True, regions=[(region, None), (region, None)])
    sparse_pool = SimpleNamespace(start_layer=0, decode_host_backing=backing)
    kv_pool = SimpleNamespace(
        full_kv_pool=sparse_pool,
        _transfer_full_attention_id={3: 0, 7: 1}.__getitem__,
        get_kv_buffer=lambda _: (host, torch.empty(4096, 0)),
        set_kv_buffer=lambda *args: calls.append("write_current"),
    )
    backend = SimpleNamespace(
        fused_sparse_state=state,
        fused_sparse_topk=2051,
        token_to_kv_pool=kv_pool,
        forward_metadata=SimpleNamespace(
            host_page_table=torch.arange(32, dtype=torch.int32).repeat(2, 1),
            real_page_table=torch.full((2, 32), -1, dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([4096, 4096], dtype=torch.int32),
        ),
        _query_lens_device=lambda _: torch.ones(2, dtype=torch.int32),
    )
    batch = SimpleNamespace(
        positions=torch.tensor([4095, 4095]),
        req_pool_indices=torch.tensor([1, 2]),
        out_cache_loc=torch.tensor([1, 2]),
    )
    return module, adapter, state, backend, batch, host, region, calls


def test_selected_rows_call_exactly_one_manager_and_one_copy(fixture):
    _, adapter, state, backend, batch, host, region, calls = fixture
    query = torch.zeros(2, 8, 512)
    selected = state.select(backend, batch, 3, query, None, None, None)
    result = adapter.forward_fsa(
        backend,
        query,
        query,
        None,
        SimpleNamespace(layer_id=3, scaling=0.0625),
        batch,
        True,
        None,
        None,
        selected,
    )
    assert calls == ["indexer_manage", "write_current", "copy_sfa"]
    assert torch.all(result == 1)
    assert not state.pending


def test_shared_layer_and_draft_seed_keep_exact_supplied_topk(fixture):
    _, _, state, backend, batch, host, region, calls = fixture
    query = torch.zeros(2, 8, 512)
    seed = torch.arange(2051, dtype=torch.int32).repeat(2, 1).flip(-1)
    for layer in (3, 7, 3):
        state.forward(
            backend,
            query,
            SimpleNamespace(layer_id=layer, scaling=0.0625),
            batch,
            seed,
            host,
            region,
        )
    assert len(calls) == 6
    for index in (0, 2, 4):
        assert calls[index][0] == "given_topk"
        torch.testing.assert_close(calls[index][1], seed)
        assert calls[index + 1] == "copy_sfa"
    assert state.layer(3).kv.data_ptr() != state.layer(7).kv.data_ptr()
    assert state.layer(3).planner is not state.layer(7).planner


def test_request_generations_and_causal_positions_are_preserved(fixture):
    _, _, state, backend, batch, _, _, _ = fixture
    batch.positions = torch.tensor([1023, 3071])
    ends, lengths, requests, table = state.metadata(backend, batch, 2)
    assert lengths.tolist() == [1024, 3072]
    assert requests.tolist() == [1, 2]
    assert table is backend.forward_metadata.host_page_table
    state.pool.req_generation[1] += 1
    state.refresh_generations()
    assert state.generations.tolist() == [0, 2, 2]
    state.pool.req_generation.zero_()
    state.pool.req_generation_epoch += 1
    state.refresh_generations()
    assert state.generations[1] > 2


def test_lifecycle_rejects_missing_host_table(fixture):
    _, _, state, backend, batch, _, _, _ = fixture
    backend.forward_metadata.host_page_table = None
    with pytest.raises(RuntimeError, match="HiCache"):
        state.metadata(backend, batch, 2)


def test_graph_padding_cannot_update_a_real_request(fixture):
    _, _, state, backend, batch, _, _, _ = fixture
    batch.req_pool_indices = torch.tensor([0, 99])
    assert state.metadata(backend, batch, 2)[1].tolist() == [0, 0]


def test_lim_budget_counts_device_map_and_request_shared_payload(monkeypatch):
    module = load("decode_memory_budget", monkeypatch)
    budget = module.DecodeKvMemoryBudget(
        13,
        13,
        512,
        0,
        128,
        2,
        2051,
        4096,
        1 << 40,
        rows_per_request=6,
        backend="lim_copy_sfa",
        lim_source_tokens=(1 << 20) + 128,
    )
    host, device = budget.fixed_bytes(10, 128)
    assert host == 0
    assert device >= 13 * 11 * (12416 * (1024 + 12) + ((1 << 20) + 512) * 4)
    assert replace(budget, lim_runtime_instances=2).fixed_bytes(10, 128)[1] > device
    assert replace(budget, graph_rows=16).fixed_bytes(10, 128)[1] > device
    assert budget.fixed_bytes(11, 128)[1] > device
    old = replace(budget, backend="external_plan")
    assert (
        budget.required_bytes(8192, 10, 128)[0] < old.required_bytes(8192, 10, 128)[0]
    )
