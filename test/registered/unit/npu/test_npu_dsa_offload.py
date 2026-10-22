from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from pytest import main as run_pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"
ADAPTER = "sglang.srt.hardware_backend.npu.hisparse.fsa_backend"
BINDING = "sglang.srt.hardware_backend.npu.hisparse.hicache_offload"
LIM = "sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_backend"
DSA = ROOT / "hardware_backend/npu/attention/ascend_dsa_backend.py"
INDEXER = ROOT / "layers/attention/dsa/dsa_indexer_kpool.py"


class Mode:
    def __init__(self, name):
        self.name = name

    def is_idle(self):
        return self.name == "idle"

    def is_decode(self):
        return self.name == "decode"

    def is_decode_or_idle(self):
        return self.name in ("decode", "idle")

    def is_extend(self):
        return self.name in ("prefill", "draft_extend")

    def is_extend_without_speculative(self):
        return self.name == "prefill"

    def is_target_verify(self):
        return self.name == "verify"

    def is_draft_extend_v2(self):
        return self.name == "draft_extend"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if (
            not isinstance(node, (ast.ClassDef, ast.FunctionDef))
            or node.name not in names
        ):
            continue
        methods = names[node.name]
        if isinstance(node, ast.ClassDef) and methods is not None:
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in methods
            ]
        nodes.append(node)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.fixture
def metadata_backend(monkeypatch):
    namespace = definitions(
        DSA,
        {
            "AscendDSAForwardMetadata": None,
            "AscendDSAAttnBackend": {
                "_cache_seqlens",
                "_cache_seqlens_cpu_max",
                "_query_lens_device",
                "_query_layout_sizes",
                "_page_table_max_seq_len",
                "_populate_dsa_metadata",
                "_replay_dsa_metadata",
            },
        },
        {
            "torch": torch,
            "dataclass": dataclass,
            "ForwardMetadata": object,
            "AscendAttnBackend": object,
        },
    )
    backend = namespace["AscendDSAAttnBackend"]()
    backend.forward_metadata = namespace["AscendDSAForwardMetadata"]()
    backend.page_size = 128
    backend.dsa_index_kpool = 1
    backend.speculative_step_id = 0
    backend.speculative_num_draft_tokens = 6
    mapping = torch.arange(3 * 1024).reshape(3, 1024)
    backend.req_to_token_pool = SimpleNamespace(req_to_token=mapping)
    fills = []

    def fill(source, output):
        fills.append(source.clone())
        output.zero_()
        output[: source.shape[0], : source.shape[1]].copy_(source + 20)
        return output

    binding = SimpleNamespace(
        host_table=lambda source: source + 20,
        fill_host_table=fill,
        mapping_epoch=0,
    )
    backend.token_to_kv_pool = SimpleNamespace(
        decode_host_backing=SimpleNamespace(page_binding=binding)
    )
    monkeypatch.setitem(
        sys.modules,
        BINDING,
        SimpleNamespace(get_offload_backing=lambda pool: pool.decode_host_backing),
    )
    batch = SimpleNamespace(
        batch_size=2,
        seq_lens=torch.tensor([130, 257]),
        seq_lens_cpu=torch.tensor([130, 257]),
        req_pool_indices=torch.tensor([1, 2]),
        spec_info=None,
        forward_mode=Mode("decode"),
    )
    return backend, batch, binding, fills


def test_host_graph_metadata_keeps_addresses_and_clears_shrinking_columns(
    metadata_backend,
):
    backend, batch, binding, fills = metadata_backend
    backend._populate_dsa_metadata(batch, _graph_capture=True)
    metadata = backend.forward_metadata
    names = (
        "host_page_table",
        "real_page_table",
        "cache_seqlens_int32",
        "token_to_batch_idx",
    )
    pointers = {name: getattr(metadata, name).data_ptr() for name in names}
    assert metadata.real_page_table.shape == metadata.host_page_table.shape == (2, 8)
    batch.req_pool_indices.copy_(torch.tensor([2, 1]))
    batch.seq_lens.fill_(127)
    batch.seq_lens_cpu.fill_(127)
    backend._replay_dsa_metadata(batch)
    assert metadata.host_page_table[:, 0].tolist() == [36, 28]
    assert not metadata.host_page_table[:, 1:].any()
    assert pointers == {name: getattr(metadata, name).data_ptr() for name in names}
    binding.mapping_epoch += 1
    backend._replay_dsa_metadata(batch)
    assert len(fills) == 2


def test_host_graph_rejects_lost_binding(metadata_backend):
    backend, batch, _, _ = metadata_backend
    backend._populate_dsa_metadata(batch, _graph_capture=True)
    backend.token_to_kv_pool.decode_host_backing = None
    with pytest.raises(RuntimeError, match="Host page binding"):
        backend._replay_dsa_metadata(batch)


@pytest.mark.parametrize("length", [124, 125, 127, 128, 131071])
def test_draft_page_table_covers_last_step(metadata_backend, length):
    backend, batch, _, _ = metadata_backend
    backend._draft_host_table_scope = SimpleNamespace(last_step=4)
    batch.spec_info = object()
    batch.seq_lens_cpu.fill_(length)
    for step in range(5):
        backend.speculative_step_id = step
        assert backend._page_table_max_seq_len(batch) == length + 5


@pytest.mark.parametrize("fail", [False, True])
def test_draft_host_scope_restores_previous_scope(monkeypatch, fail):
    scopes = []

    class Parent:
        def init_forward_metadata_out_graph(self, batch, in_capture=False):
            scope = self.attn_backends[0]._draft_host_table_scope
            assert all(
                item._draft_host_table_scope is scope for item in self.attn_backends[:5]
            )
            assert scope.last_step == 4
            scopes.append(scope)
            if fail:
                raise RuntimeError("prepare failed")

    namespace = definitions(
        DSA,
        {"AscendDSAAttnMultiStepDraftBackend": None},
        {"AscendAttnMultiStepDraftBackend": Parent, "AscendDSAAttnBackend": object},
    )
    monkeypatch.setitem(
        sys.modules,
        BINDING,
        SimpleNamespace(
            DraftHostTableScope=lambda last_step: SimpleNamespace(last_step=last_step),
            get_offload_backing=lambda pool: pool,
        ),
    )
    backend = namespace["AscendDSAAttnMultiStepDraftBackend"]()
    previous = object()
    backend.speculative_num_steps = 6
    backend.attn_backends = [
        SimpleNamespace(token_to_kv_pool=object(), _draft_host_table_scope=previous)
        for _ in range(6)
    ]
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="prepare failed"):
                backend.init_forward_metadata_out_graph(None)
        else:
            backend.init_forward_metadata_out_graph(None)
        assert all(
            item._draft_host_table_scope is previous for item in backend.attn_backends
        )
    assert scopes[0] is not scopes[1]


@pytest.mark.parametrize("backend_name", [None, "external_plan", "lim_copy_sfa"])
@pytest.mark.parametrize(
    "mode", ["prefill", "decode", "verify", "draft_extend", "idle"]
)
def test_indexer_dispatch_follows_write_phase(monkeypatch, backend_name, mode):
    calls = []
    metadata = object()
    backend = SimpleNamespace(
        fused_sparse_state=SimpleNamespace(backend_name=backend_name),
        get_indexer_metadata=lambda *args: metadata,
    )
    namespace = definitions(
        INDEXER,
        {"IndexerKPool": {"forward_npu"}},
        {
            "torch": torch,
            "F": functional,
            "MultiPlatformOp": object,
            "get_attn_backend": lambda: backend,
            "rearrange": lambda tensor, *args, **kwargs: tensor.reshape(-1, 1, 4),
        },
    )
    indexer = namespace["IndexerKPool"]()
    indexer.head_dim = 4
    indexer.rope_head_dim = 0
    indexer.index_topk = 2048
    indexer.index_kpool = 4
    indexer.index_kpool_compress_gate = torch.eye(4)
    indexer.wq_b = lambda tensor: (tensor, None)
    indexer.wk = lambda tensor: (tensor, None)
    indexer.k_norm = lambda tensor: tensor
    indexer._compress_write_mtp_npu = lambda *args: calls.append("write_mtp")
    indexer._compress_write_decode_npu = lambda *args: calls.append("write_decode")
    indexer._compress_write_extend_npu = lambda *args: calls.append("write_extend")
    indexer._get_logits_head_gate_npu = lambda tensor: tensor
    indexer._prepare_decode_indexer_inputs_npu = lambda *args: (None, None, None)
    vendor = torch.full((1, 2051), 9, dtype=torch.int32)
    selected = torch.full_like(vendor, 11)
    indexer._get_topk_paged_npu = lambda *args, **kwargs: (
        calls.append("vendor") or vendor
    )
    indexer._get_topk_ragged_npu = indexer._get_topk_paged_npu
    indexer._fix_short_sequence_topk_a5 = lambda result, *args: result
    monkeypatch.setitem(
        sys.modules,
        LIM,
        SimpleNamespace(
            select_offloaded_topk=lambda *args: calls.append("lim") or selected
        ),
    )
    batch = SimpleNamespace(forward_mode=Mode(mode), seq_lens_cpu=torch.tensor([4096]))
    result = indexer.forward_npu(
        torch.ones(1, 4), torch.ones(1, 4), torch.tensor([4095]), batch, 3
    )
    if mode == "idle":
        assert not calls
        assert (result == -1).all()
    else:
        write = (
            "write_decode"
            if mode == "decode"
            else "write_extend"
            if mode == "prefill"
            else "write_mtp"
        )
        use_lim = backend_name == "lim_copy_sfa" and mode != "prefill"
        assert calls == [write, "lim" if use_lim else "vendor"]
        assert result is (selected if use_lim else vendor)


@pytest.mark.parametrize("length", [2047, 2048, 2049])
def test_vendor_short_sequence_boundary_preserves_2051_columns(length):
    namespace = definitions(
        INDEXER, {"_a5_full_topk_for_short_sequences": None}, {"torch": torch}
    )
    supplied = torch.arange(2051, dtype=torch.int32).flip(0).repeat(2, 1)
    result = namespace["_a5_full_topk_for_short_sequences"](
        supplied, torch.tensor([length])
    )
    expected = torch.arange(2051, dtype=torch.int32)
    expected[expected >= length] = -1
    assert torch.equal(result[0], expected if length < 2048 else supplied[0])
    assert (result[1] == -1).all()


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_prefill_callback_order_and_no_duplicate_write(monkeypatch, save, fail):
    calls = []

    class Parent:
        def forward_sparse(self, *args, **kwargs):
            assert args[5] is False
            calls.append("vendor")
            if fail:
                raise RuntimeError("attention failed")
            return args[0]

    namespace = definitions(
        DSA, {"AscendDSAAttnBackend": {"forward_sparse"}}, {"AscendAttnBackend": Parent}
    )
    runtime = SimpleNamespace(
        on_compute_start=lambda layer: calls.append(("start", layer)),
        finish_layer=lambda layer: calls.append(("finish", layer)),
    )
    pool = SimpleNamespace(start_layer=5, prefill_offload_runtime=runtime)
    backend = namespace["AscendDSAAttnBackend"]()
    backend.fused_sparse_state = None
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 0
    backend.token_to_kv_pool = SimpleNamespace(
        set_kv_buffer=lambda *args: calls.append("write"),
        get_kv_buffer=lambda *args: calls.append("prepare"),
    )
    monkeypatch.setitem(
        sys.modules,
        ADAPTER,
        SimpleNamespace(sparse_pool_and_layer=lambda *args: (pool, 2)),
    )
    query = torch.zeros(1, 1, 512)
    batch = SimpleNamespace(
        forward_mode=Mode("prefill"), out_cache_loc=torch.tensor([128])
    )
    layer = SimpleNamespace(layer_id=13, tp_k_head_num=1)
    if fail:
        with pytest.raises(RuntimeError, match="attention failed"):
            backend.forward_sparse(query, query, None, layer, batch, save_kv_cache=save)
    else:
        assert (
            backend.forward_sparse(query, query, None, layer, batch, save_kv_cache=save)
            is query
        )
    assert calls == (["write"] if save else []) + [
        "prepare",
        ("start", 7),
        "vendor",
    ] + ([] if fail else [("finish", 7)])


@pytest.mark.parametrize("name", ["external_plan", "lim_copy_sfa"])
def test_attention_dispatch_uses_initialized_state(monkeypatch, name):
    namespace = definitions(
        DSA, {"AscendDSAAttnBackend": {"forward_sparse"}}, {"AscendAttnBackend": object}
    )
    backend = namespace["AscendDSAAttnBackend"]()
    backend.fused_sparse_state = SimpleNamespace(backend_name=name)
    output = object()
    monkeypatch.setitem(
        sys.modules, ADAPTER, SimpleNamespace(forward_fsa=lambda *args: output)
    )
    assert (
        backend.forward_sparse(
            None, None, None, None, SimpleNamespace(forward_mode=Mode("decode"))
        )
        is output
    )


if __name__ == "__main__":
    sys.exit(run_pytest([__file__, *sys.argv[1:]]))
