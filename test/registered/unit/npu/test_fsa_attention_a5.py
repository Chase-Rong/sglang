"""A5 NoPE attention, real registered Host KV and external-plan integration."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires qualified idle A5 device, standalone extension and matching OPP",
)


@pytest.fixture
def device():
    import torch_npu

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def reference_attention(query, keys, indices, batch):
    result = []
    for row in range(batch):
        selected = keys[row, indices[row][indices[row] >= 0].long()].float()
        if selected.shape[0] == 0:
            result.append(torch.zeros_like(query[row], dtype=torch.float32))
            continue
        logits = query[row].float() @ selected.T * 0.0625
        result.append(torch.softmax(logits, dim=-1) @ selected)
    return torch.stack(result)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,heads,tail", [(1, 16, 0), (1, 64, 3), (4, 16, 1), (1, 128, 2)])
@pytest.mark.parametrize("seed,valid_prefix", [(137, 2048), (271, 31), (911, 0)])
@pytest.mark.parametrize("mode", ["fused", "reference"])
def test_host_attention_cold_warm_reshuffle_and_graph(device, dtype, batch, heads, tail, seed, valid_prefix, mode):
    from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion
    from sglang.srt.hardware_backend.npu.fused_sparse_attention import (
        FusedSparseAttentionState,
        run_fused_sparse_attention,
    )

    torch.manual_seed(seed)
    page_size, sequence, topk = 128, 4096, 2051
    pages = sequence // page_size
    source = (torch.randn(batch, sequence, 512, device="cpu") * 0.5).to(dtype)
    query_cpu = (torch.randn(batch, heads, 512, device="cpu") * 0.5).to(dtype)
    query = query_cpu.to(device)
    ids = torch.full((batch, topk), -1, dtype=torch.int32, device="cpu")
    ids[:, :valid_prefix] = torch.arange(valid_prefix, device="cpu")
    if tail:
        ids[:, 2048 : 2048 + tail] = torch.arange(sequence - tail, sequence, device="cpu")
    indices = ids.to(device)
    expected = reference_attention(query_cpu, source, ids, batch)
    region = RegisteredHostRegion.allocate(
        (batch * pages, page_size, 512), dtype, device_id=device.index
    )
    region.tensor.copy_(source.view_as(region.tensor))
    rope = torch.empty(batch * pages, page_size, 0, dtype=dtype, device="cpu")
    query_rope = query.new_empty(batch, heads, 0)
    table = torch.arange(batch * pages, dtype=torch.int32, device=device).view(batch, pages)
    lengths = torch.full((batch,), sequence, dtype=torch.int32, device=device)
    query_lengths = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    requests = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    pool = SimpleNamespace(req_generation=torch.ones(batch + 1, dtype=torch.int64, device="cpu"))
    state = FusedSparseAttentionState(device, pool, external_planner=True, resident_capacity=4096)
    graph = None

    def invoke():
        inputs = state.build_inputs(
            layer_id=0, q_nope=query, q_rope=query_rope, topk_indices=indices,
            full_kv=region.tensor, full_rope=rope, block_table=table,
            kv_lengths=lengths, query_lengths=query_lengths, request_slots=requests,
            scale=0.0625,
        )
        inputs = replace(inputs, full_kv_region=region)
        return run_fused_sparse_attention(inputs, mode=mode), inputs

    def check(actual):
        torch.npu.synchronize()
        actual_cpu = actual.cpu().float()
        try:
            torch.testing.assert_close(actual_cpu, expected, atol=0.002, rtol=0.025)
        except AssertionError:
            from pathlib import Path

            torch.save({"actual": actual_cpu, "expected": expected},
                       Path(os.environ["SGLANG_FSA_WORK_DIR"]) / "tmp/attention_failure.pt")
            raise

    try:
        output, inputs = invoke()
        torch.npu.synchronize()
        if os.environ.get("SGLANG_FSA_DIAG_REFERENCE") == "1":
            import torch_npu
            from pathlib import Path

            device_keys = source.reshape(batch * pages, page_size, 1, 512).to(device)
            baseline, _, _ = torch_npu.npu_sparse_flash_attention(
                query=query, key=device_keys, value=device_keys,
                query_rope=query.new_zeros(batch, heads, 64),
                key_rope=query.new_zeros(batch * pages, page_size, 1, 64),
                sparse_indices=indices[:, None], block_table=table,
                actual_seq_lengths_query=query_lengths,
                actual_seq_lengths_kv=lengths, scale_value=0.0625,
                sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
                sparse_mode=3, attention_mode=2, return_softmax_lse=False,
            )
            torch.npu.synchronize()
            print("max_error fused/cpu", (output.cpu().float() - expected).abs().max().item(),
                  "sfa/cpu", (baseline.cpu().float() - expected).abs().max().item(),
                  "fused/sfa", (output.float() - baseline.float()).abs().max().item(), flush=True)
            torch.save({
                "query": query_cpu, "source": source, "indices": ids,
                "fused": output.cpu(), "sfa": baseline.cpu(), "expected": expected,
                "slots": inputs.selection_plan.slots.cpu(),
                "selection": inputs.selection_kv_cache.cpu(),
            }, Path(os.environ["SGLANG_FSA_WORK_DIR"]) / "tmp/attention_precision.pt")
        check(output)
        assert inputs.selection_plan.miss_mask.sum().item() == batch * (valid_prefix + tail)
        torch.npu.synchronize()
        region.tensor.zero_()
        indices.copy_(indices.flip(-1))
        output, inputs = invoke()
        check(output)
        assert not inputs.selection_plan.miss_mask.any().item()
        state.reserve(0, batch, topk, rope, region.tensor)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output, _ = invoke()
        indices.copy_(indices.flip(-1))
        graph.replay()
        check(output)
        region.tensor.copy_((source + 0.25).view_as(region.tensor))
        pool.req_generation.add_(1)
        state.refresh_generations()
        expected = reference_attention(query_cpu, source + 0.25, ids, batch)
        graph.replay()
        check(output)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        for planner in state.planners.values():
            planner.close(drained=True)
        region.close(drained=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mtp_graph_rewrites_mutable_tail_without_owner_change(device, dtype):
    from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion
    from sglang.srt.hardware_backend.npu.fused_sparse_attention import FusedSparseAttentionState, run_fused_sparse_attention

    torch.manual_seed(721)
    requests, draft_rows, heads, sequence, page_size = 2, 3, 16, 256, 128
    rows = requests * draft_rows
    source = torch.randn(requests, sequence, 512, dtype=dtype, device="cpu") * 0.25
    query_cpu = torch.randn(rows, heads, 512, dtype=dtype, device="cpu") * 0.25
    region = RegisteredHostRegion.allocate((4, page_size, 512), dtype, device_id=device.index)
    region.tensor.copy_(source.view_as(region.tensor))
    rope = torch.empty(4, page_size, 0, dtype=dtype, device="cpu")
    query = query_cpu.to(device)
    ids = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    ids[:, :32] = torch.arange(32, device=device)
    ids[:, 2048:] = torch.arange(253, 256, device=device)
    lengths = torch.full((requests,), sequence, dtype=torch.int32, device=device)
    cumulative = torch.tensor([3, 6], dtype=torch.int32, device=device)
    positions = torch.tensor([253, 254, 255] * 2, device=device)
    table = torch.arange(4, dtype=torch.int32, device=device).view(2, 2)
    slots = torch.tensor([1, 2], dtype=torch.int64, device=device)
    pool = SimpleNamespace(req_generation=torch.tensor([0, 1, 1], device="cpu"))
    state = FusedSparseAttentionState(device, pool, external_planner=True,
                                     resident_capacity=4096, mutable_tail_tokens=3)
    graph = None

    def invoke():
        inputs = state.build_inputs(layer_id=0, q_nope=query, q_rope=query.new_empty(rows, heads, 0),
                                    topk_indices=ids, full_kv=region.tensor, full_rope=rope,
                                    block_table=table, kv_lengths=lengths, query_lengths=cumulative,
                                    request_slots=slots, query_positions=positions, scale=0.0625)
        return run_fused_sparse_attention(replace(inputs, full_kv_region=region))

    def expected():
        logical = ids.cpu().masked_fill(ids.cpu() > positions.cpu()[:, None], -1)
        return reference_attention(query_cpu, source.repeat_interleave(draft_rows, dim=0), logical, rows)

    try:
        state.reserve(0, rows, 2051, rope, region.tensor)
        output = invoke()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu().float(), expected(), atol=0.002, rtol=0.025)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = invoke()
        source[:, 253:] += 0.5
        region.tensor.copy_(source.view_as(region.tensor))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu().float(), expected(), atol=0.002, rtol=0.025)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        for planner in state.planners.values():
            planner.close(drained=True)
        region.close(drained=True)
