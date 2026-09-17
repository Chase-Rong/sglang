"""GLM-5.3 NoPE contracts at the fused runtime boundary, without NPU allocation."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[4]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu"


def load_module(name):
    qualified = f"sglang.srt.hardware_backend.npu.{name}"
    spec = importlib.util.spec_from_file_location(qualified, NPU / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tail", range(4))
def test_expanded_kpool_width_preserves_tail_and_padding(tail):
    fsa = load_module("fused_sparse_attention")
    pool = SimpleNamespace(req_generation=torch.tensor([0, 3, 9]))
    state = fsa.FusedSparseAttentionState("cpu", pool, resident_capacity=4096)
    indices = torch.full((2, 2051), -1, dtype=torch.int32)
    indices[:, :2048] = torch.arange(2048)
    indices[:, 2048 : 2048 + tail] = torch.arange(4096, 4096 + tail)
    query = torch.randn(2, 16, 512)
    keys = torch.empty(70, 128, 512)
    rope = torch.empty(70, 128, 0)
    tables = torch.stack((torch.arange(1, 35), torch.arange(35, 69))).int()
    inputs = state.build_inputs(
        layer_id=5,
        q_nope=query,
        q_rope=torch.empty(2, 16, 0),
        topk_indices=indices,
        full_kv=keys,
        full_rope=rope,
        block_table=tables,
        kv_lengths=torch.tensor([4096 + tail] * 2),
        query_lengths=torch.tensor([1, 2]),
        request_slots=torch.tensor([1, 2]),
        scale=0.0625,
    )
    assert inputs.query.data_ptr() == query.data_ptr()
    assert inputs.selection_topk_indices.shape == (2, 1, 2051)
    assert torch.equal(inputs.selection_topk_indices[:, 0], indices)
    assert inputs.selection_k_rope.shape[-1] == 0
    assert inputs.scale_value == 0.0625
    assert state.layers[5].cache.max_blocks_per_row == 32


def test_external_plan_2051_rows_preserves_reshuffled_hits_and_layer_isolation():
    fsa = load_module("fused_sparse_attention")
    planner = load_module("external_sparse_planner")
    managers = [fsa.FusedSparseAttentionCacheManager("cpu", 4096) for _ in range(2)]
    ids = torch.arange(2051).reshape(1, 1, -1).int()
    caches = [
        manager.ensure(ids, torch.empty(40, 128, 0), torch.empty(40, 128, 512))[0]
        for manager in managers
    ]
    keys = torch.arange(2051).reshape(1, -1).long() + 128
    slots = torch.arange(2051).reshape(1, -1)
    first = planner.materialize_layer_plan(
        caches[0], keys, slots, caches[0].selection_membership_map
    )
    assert first.miss_mask.all()
    planner.commit_layer_plan(caches[0], first)
    permutation = torch.randperm(2051)
    second = planner.materialize_layer_plan(
        caches[0],
        keys[:, permutation],
        slots[:, permutation],
        caches[0].selection_membership_map,
    )
    assert not second.miss_mask.any()
    control = caches[0].selection_membership_map[0, 16384:16392]
    assert control[:4].tolist() == [0x5A4D, 0x5A45, 2051, 16384 - 2051]
    other = planner.materialize_layer_plan(
        caches[1], keys, slots, caches[1].selection_membership_map
    )
    assert other.miss_mask.all()


def test_host_nope_allocates_only_latent_and_overwrites_current_token(monkeypatch):
    registered = load_module("registered_host_kv")
    backing_module = load_module("decode_host_kv")

    class HostPool:
        def __init__(self, device_id):
            self.components = {}

        def allocate_component(self, name, shape, dtype, **kwargs):
            tensor = torch.full(shape, -99, dtype=dtype)
            region = SimpleNamespace(
                tensor=tensor, nbytes=tensor.nbytes, view=tensor.view
            )
            self.components[name] = region
            return SimpleNamespace(region=region)

        def copy_to_host(self, name, src, src_index, dst_index, valid):
            self.components[name].tensor[dst_index[valid]] = src[src_index[valid]]

    monkeypatch.setattr(backing_module, "NpuRegisteredHostKvPool", HostPool)
    backing = backing_module.DecodeHostKvBacking(
        layers=2,
        pages=3,
        page_size=128,
        latent_dim=512,
        rope_dim=0,
        dtype=torch.bfloat16,
        device_id=1,
        host_budget=2 * 384 * 512 * 2,
    )
    assert len(backing.pool.components) == 2
    assert backing.regions[0][1] is None
    assert backing.layer_views(0)[1].shape == (3, 128, 1, 0)
    locations = torch.tensor([0, 128, 383, 384, -1])
    values = torch.randn(5, 1, 512).bfloat16()
    backing.write_current(1, locations, values, torch.empty(5, 1, 0))
    torch.testing.assert_close(backing.regions[1][0].tensor[128], values[1, 0])
    assert torch.all(backing.regions[0][0].tensor == -99)
    backing.write_current(1, locations, values + 1, torch.empty(5, 1, 0))
    torch.testing.assert_close(backing.regions[1][0].tensor[128], values[1, 0] + 1)


def test_nope_budget_includes_2051_tail_columns():
    budget_module = load_module("decode_memory_budget")
    budget = budget_module.DecodeKvMemoryBudget(
        layers=13,
        planner_groups=13,
        latent_dim=512,
        rope_dim=0,
        index_dim=128,
        element_bytes=2,
        topk=2051,
        resident_capacity=4096,
        host_bytes=1 << 38,
    )
    assert budget.host_bytes_per_token == 13 * 512 * 2
    assert budget.selection_capacity(128) == 4096
    host, device = budget.required_bytes(1024 * 1024, 4, 128)
    assert host > 13 * 1024 * 1024 * 512 * 2
    assert device > 13 * 1024 * 1024 * 128 * 2


def test_hybrid_global_layer_mapping_is_not_assumed_contiguous():
    load_module("fused_sparse_attention")
    adapter = load_module("fsa_backend")
    full = SimpleNamespace(start_layer=0, layer_num=3)
    mapping = {3: 0, 7: 1, 11: 2}
    wrapper = SimpleNamespace(
        full_kv_pool=full,
        full_attention_layer_id_mapping=mapping,
        _transfer_full_attention_id=mapping.__getitem__,
    )
    assert adapter.sparse_layer_ids(wrapper) == (3, 7, 11)
    assert adapter.sparse_pool_and_layer(wrapper, 7) == (full, 1)


def test_extension_loading_is_optional_and_does_not_import_replacement_wheel(
    monkeypatch,
):
    native = load_module("fsa_native")
    monkeypatch.delenv("SGLANG_NPU_FSA_EXTENSION", raising=False)
    monkeypatch.setattr(
        torch.ops, "load_library", lambda _: pytest.fail("unexpected load")
    )
    assert native.load_native_extension() is False
    with pytest.raises(RuntimeError, match="installed sgl-kernel-npu wheel"):
        native.native_ops()


def test_actual_backend_adapter_preserves_hybrid_host_mapping_and_mtp_causality(monkeypatch):
    fsa = load_module("fused_sparse_attention")
    adapter = load_module("fsa_backend")
    request_pool = SimpleNamespace(req_generation=torch.tensor([0, 2, 3]))
    state = fsa.FusedSparseAttentionState("cpu", request_pool, resident_capacity=8)
    keys = torch.empty(8, 2, 1, 512, dtype=torch.float16)
    rope = torch.empty(8, 2, 1, 0, dtype=torch.float16)
    selected_region = object()
    full = SimpleNamespace(
        start_layer=0,
        decode_host_backing=SimpleNamespace(regions=[(object(), None), (selected_region, None)]),
    )
    writes = []
    pool = SimpleNamespace(
        full_kv_pool=full,
        _transfer_full_attention_id={3: 0, 7: 1}.__getitem__,
        get_kv_buffer=lambda layer: (keys, rope),
        set_kv_buffer=lambda *args: writes.append(args),
    )
    table = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    backend = SimpleNamespace(
        token_to_kv_pool=pool,
        fused_sparse_state=state,
        fused_sparse_topk=3,
        fused_sparse_mode="fused",
        page_size=2,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        forward_metadata=SimpleNamespace(
            real_page_table=table, cache_seqlens_int32=torch.tensor([6, 6], dtype=torch.int32)
        ),
        _query_lens_device=lambda batch: torch.tensor([2, 2], dtype=torch.int32),
    )
    query = torch.zeros(4, 16, 512, dtype=torch.float16)
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor([6, 7, 12, 13]),
        req_pool_indices=torch.tensor([1, 2]),
        positions=torch.tensor([4, 5, 4, 5]),
    )
    layer = SimpleNamespace(layer_id=7, scaling=0.0625)
    monkeypatch.setattr(adapter, "run_fused_sparse_attention", lambda inputs, mode: inputs)
    inputs = adapter.forward_fsa(
        backend, query, query[:, :1], None, layer, batch, True, None,
        query.new_empty(4, 1, 0), torch.tensor([[0, 4, 5]] * 4, dtype=torch.int32),
    )
    assert len(writes) == 1 and writes[0][0] is layer
    assert inputs.full_kv_region is selected_region and inputs.full_rope_region is None
    assert inputs.query is query and inputs.full_kv_cache.data_ptr() == keys.data_ptr()
    assert inputs.selection_k_rope.shape[-1] == 0
    assert inputs.scale_value == 0.0625
    torch.testing.assert_close(inputs.full_q_actual_seq, torch.tensor([2, 4], dtype=torch.int32))
    torch.testing.assert_close(inputs.full_kv_block_table, table)
    torch.testing.assert_close(
        inputs.selection_topk_indices[:, 0],
        torch.tensor([[0, 4, -1], [0, 4, 5], [0, 4, -1], [0, 4, 5]], dtype=torch.int32),
    )
def test_prefill_kpool_budget_keeps_indexer_resident_across_layers():
    module = load_module("prefill_offload")
    budget = module.PrefillKvMemoryBudget(
        13, 2, 512 * 2, 1 << 38, resident_index_bytes_per_token=13 * 128 * 2
    )
    assert budget.device_bytes_per_token == 2 * 512 * 2 + 13 * 128 * 2
    assert budget.host_bytes_per_token == 13 * 512 * 2


@pytest.mark.parametrize("last_chunk", [False, True])
def test_kpool_final_send_preserves_baseline_indexer_and_kda_state(last_chunk):
    import ast
    import numpy as np

    path = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MooncakeKVSender")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "send")
    method.returns = None
    method.decorator_list = []
    for argument in method.args.args + method.args.kwonlyargs:
        argument.annotation = None
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    indices = np.array([2, 3], dtype=np.int32)
    state_indices = [np.array([7], dtype=np.int32)]
    sender = SimpleNamespace(
        _prepare_send_indices=lambda *args: (indices, slice(0, 2), last_chunk, False),
        _record_transfer_indices=lambda *args: None,
        _layer_ready_operations={1: SimpleNamespace(state="acknowledged")},
        kv_mgr=SimpleNamespace(
            kv_args=SimpleNamespace(prefill_layerwise_offload=True, prefill_layerwise_aux_kv=True),
            add_transfer_request=lambda *args, **kwargs: calls.append((args, kwargs)),
        ),
        bootstrap_room=1, aux_index=9,
        trace_ctx=SimpleNamespace(copy_for_thread=lambda: None),
    )
    namespace["send"](sender, indices, state_indices=state_indices, num_kv_tokens=129)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert np.array_equal(args[1], indices)
    assert kwargs["num_kv_tokens"] == 129
    if last_chunk:
        assert kwargs["state_indices"] is state_indices
