"""Run with torchrun: real TP shared Host allocation, DMA and fused Graph replay."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_SHARED_TP") != "1",
    reason="requires torchrun and explicitly selected idle NPUs",
)


@pytest.fixture(scope="module")
def tp():
    import torch_npu
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel, get_attn_tp_group,
        init_distributed_environment, initialize_model_parallel, set_custom_all_reduce,
    )
    from sglang.srt.runtime_context import get_parallel

    rank = int(os.environ["RANK"])
    devices = [int(value) for value in os.environ["SGLANG_FSA_TEST_DEVICES"].split(",")]
    device_id = devices[rank]
    torch.npu.set_device(device_id)
    set_custom_all_reduce(False)
    init_distributed_environment(world_size=len(devices), rank=rank, local_rank=device_id,
                                 distributed_init_method="env://", backend="hccl", timeout=90)
    initialize_model_parallel(tensor_model_parallel_size=len(devices), backend="hccl")
    group = get_attn_tp_group()
    with get_parallel().override(attn_dcp_size=1):
        try:
            yield SimpleNamespace(group=group, rank=rank, device_id=device_id,
                                  device=torch.device(f"npu:{device_id}"))
        finally:
            torch.npu.synchronize()
            destroy_model_parallel()
            destroy_distributed_environment()


def shared_group(tp):
    from sglang.srt.hardware_backend.npu.shared_host_kv import shared_host_group_from_runtime

    group = shared_host_group_from_runtime(tp.device_id)
    assert group is not None
    return group


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_integrated_shared_prefill_decode(tp, dtype):
    from test_fsa_pool_lifecycle_npu import exercise_integrated_slots

    exercise_integrated_slots(dtype, tp.device_id, shared_group=shared_group(tp))


def test_default_optout_and_dcp_guard(tp, monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.shared_host_kv import shared_host_group_from_runtime
    from sglang.srt.runtime_context import get_parallel

    monkeypatch.delenv("SGLANG_NPU_SHARED_HOST_KV", raising=False)
    assert envs.SGLANG_NPU_SHARED_HOST_KV.get()
    with get_parallel().override(attn_dcp_size=2):
        with pytest.raises(ValueError, match="DCP=1"):
            shared_host_group_from_runtime(tp.device_id)
    monkeypatch.setenv("SGLANG_NPU_SHARED_HOST_KV", "0")
    assert shared_host_group_from_runtime(tp.device_id) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shared_prefill_four_layers_two_slots(tp, dtype):
    from test_fsa_pool_lifecycle_npu import exercise_prefill_slots

    exercise_prefill_slots(dtype, tp.device_id, shared_group=shared_group(tp))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shared_allocation_one_writer_graph_and_independent_detach(tp, dtype):
    from sglang.srt.hardware_backend.npu.registered_host_kv import NpuRegisteredHostKvPool

    shared = shared_group(tp)
    host = NpuRegisteredHostKvPool(tp.device_id, shared_group=shared)
    entry = host.allocate_component("tp-latent", (128, 512), dtype, page_size=128)
    handle = entry.region.export_handle()
    identities = tp.group.all_gather_object((handle.segment, handle.creator, handle.created, handle.nbytes))
    assert len(set(identities)) == 1
    assert sum(tp.group.all_gather_object(shared.is_writer)) == 1
    source = torch.full((128, 512), tp.rank + 2, dtype=dtype, device=tp.device)
    output = torch.empty_like(source)
    indices = torch.arange(128, dtype=torch.int64, device=tp.device)
    valid = torch.ones(128, dtype=torch.bool, device=tp.device)
    graph = None

    def copy():
        host.copy_to_host("tp-latent", source, indices, indices, valid)
        host.copy_from_host("tp-latent", output, indices, indices, valid)

    try:
        copy()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), torch.full((128, 512), 2, dtype=dtype), atol=0, rtol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            copy()
        for value in (4, 8, 16, 32):
            source.fill_(value + tp.rank * 100)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), torch.full((128, 512), value, dtype=dtype), atol=0, rtol=0)
        graph.reset()
        graph = None
        if shared.is_writer:
            host.close(drained=True)
        tp.group.all_gather_object("writer detached")
        if not shared.is_writer:
            host.copy_from_host("tp-latent", output, indices, indices, valid)
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), torch.full((128, 512), 32, dtype=dtype), atol=0, rtol=0)
        tp.group.all_gather_object("readers drained")
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        host.close(drained=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_shared_decode_fused_replay_with_tp_output_collective(tp, dtype, integrated, mapped):
    from sglang.srt.hardware_backend.npu.decode_host_kv import DecodeHostKvBacking
    from sglang.srt.hardware_backend.npu.fused_sparse_attention import FusedSparseAttentionState, run_fused_sparse_attention

    backing = DecodeHostKvBacking(layers=1, pages=18, page_size=128, latent_dim=512,
                                  rope_dim=0, dtype=dtype, device_id=tp.device_id,
                                  host_budget=1 << 24, shared_group=shared_group(tp), integrated=integrated)
    region = backing.regions[0][0]
    source_cpu = (torch.arange(2176 * 512, device="cpu").view(2176, 512) % 97).to(dtype) / 128
    source = source_cpu.to(tp.device) + tp.rank * 10
    ids = torch.arange(2176, dtype=torch.int64, device=tp.device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    torch.manual_seed(621 + tp.rank)
    query_cpu = torch.randn(1, 16, 512, device="cpu").to(dtype) * 0.25
    query = query_cpu.to(tp.device)
    topk = torch.arange(2051, dtype=torch.int32, device=tp.device)[None]
    table = torch.arange(17, dtype=torch.int32, device=tp.device)[None]
    lengths = torch.tensor([2176], dtype=torch.int32, device=tp.device)
    qlengths = torch.tensor([1], dtype=torch.int32, device=tp.device)
    requests = torch.tensor([1], dtype=torch.int64, device=tp.device)
    pool = SimpleNamespace(req_generation=torch.tensor([0, 1], device="cpu"))
    state = FusedSparseAttentionState(tp.device, pool, external_planner=True,
                                    resident_capacity=4096, mapped_host_plan=mapped)
    key, rope = backing.layer_views(0)
    key, rope = key.squeeze(2), rope.squeeze(2)
    graph = None

    def invoke():
        backing.pool.copy_to_host("decode-0-latent", source, ids, ids, valid)
        inputs = state.build_inputs(layer_id=0, q_nope=query, q_rope=query.new_empty(1, 16, 0),
                                    topk_indices=topk, full_kv=key, full_rope=rope,
                                    block_table=table, kv_lengths=lengths, query_lengths=qlengths,
                                    request_slots=requests, scale=0.0625)
        result = run_fused_sparse_attention(replace(inputs, full_kv_region=region)).float()
        torch.distributed.all_reduce(result, group=tp.group.device_group)
        return result

    def check(output, shift):
        selected = (source_cpu[:2051] + shift).float()
        local = (query_cpu.float() @ selected.T * 0.0625).softmax(-1) @ selected
        expected = sum(tp.group.all_gather_object(local))
        torch.testing.assert_close(output.cpu(), expected, atol=0.006, rtol=0.025)

    try:
        state.reserve(0, 1, 2051, rope, key)
        result = invoke()
        torch.npu.synchronize()
        check(result, 0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = invoke()
        for shift in (0.125, 0.25, 0.5):
            source.copy_((source_cpu + shift).to(tp.device) + tp.rank * 10)
            pool.req_generation.add_(1)
            state.refresh_generations()
            topk.copy_(topk.flip(-1))
            graph.replay()
            torch.npu.synchronize()
            check(result, shift)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        for planner in state.planners.values():
            planner.close(drained=True)
        backing.pool.close(drained=True)
