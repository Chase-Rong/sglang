"""Qualify the A5 native Indexer and exported scores before management fusion."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an explicitly authorized NPU component environment",
)


@pytest.mark.parametrize("rows,heads", [(1, 32), (6, 64), (10, 32)])
def test_native_indexer_matches_installed_op_and_scores(rows, heads):
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    torch.ops.load_library(os.environ["SGLANG_LIM_NATIVE_EXTENSION"])
    generator = torch.Generator().manual_seed(771)
    query_cpu = (torch.randn(rows, heads, 128, generator=generator) * 0.1).bfloat16()
    key_cpu = (torch.randn(33, 128, 1, 128, generator=generator) * 0.1).bfloat16()
    weights_cpu = (torch.rand(rows, heads, generator=generator) * 0.2 - 0.05).bfloat16()
    query, key, weights = [
        value.to("npu") for value in (query_cpu, key_cpu, weights_cpu)
    ]
    table_cpu = torch.arange(1, 33, dtype=torch.int32).flip(0).repeat(rows, 1)
    table = table_cpu.to("npu")
    lengths_cpu = torch.tensor(
        [4096 - row * 137 for row in range(rows)], dtype=torch.int32
    )
    lengths = lengths_cpu.to("npu")
    query_ends = torch.arange(1, rows + 1, dtype=torch.int32, device="npu")
    indices = torch.empty((rows, 512), dtype=torch.int32, device="npu")
    scores = torch.empty((rows, 4096), dtype=torch.int32, device="npu")
    cores = 8
    workspace = torch.empty(cores * 4 * 4096 * 4, dtype=torch.uint8, device="npu")

    def invoke():
        torch.ops.sgl_lim_native.indexer_out(
            query,
            key,
            weights,
            query_ends,
            lengths,
            table,
            indices,
            scores,
            workspace,
            cores,
        )

    def check():
        reference, _ = torch_npu.npu_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_key=lengths,
            block_table=table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=3,
        )
        torch.npu.synchronize()
        actual = indices.cpu().long()
        codes = scores.cpu().long() & 0xFFFFFFFF
        bits = torch.where(codes & 0x80000000 != 0, codes ^ 0x80000000, ~codes)
        decoded = bits.to(torch.int32).view(torch.float32)
        visible_cpu = lengths.cpu()
        current_query = query.cpu().float()
        current_weights = weights.cpu().float()
        for row in range(rows):
            visible = int(visible_cpu[row])
            keys = key_cpu[table_cpu[row].long()].reshape(-1, 128).float()[:visible]
            golden = (
                (current_query[row] @ keys.T).relu() * current_weights[row, :, None]
            ).sum(0)
            torch.testing.assert_close(
                decoded[row, :visible], golden, atol=0.0002, rtol=0.003
            )
            selected = actual[row][actual[row] >= 0]
            assert selected.numel() == min(512, visible)
            assert torch.unique(selected).numel() == selected.numel()
            assert torch.all(selected < visible)
            native_set = selected.sort().values
            baseline_set = reference[row].cpu().flatten().long()
            baseline_set = (
                baseline_set[(baseline_set >= 0) & (baseline_set < visible)]
                .sort()
                .values
            )
            torch.testing.assert_close(native_set, baseline_set, rtol=0, atol=0)

    invoke()
    check()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            invoke()
        for visible in (511, 0, 3077):
            lengths.fill_(visible)
            query.mul_(0.875)
            graph.replay()
            check()
    finally:
        torch.npu.synchronize()
        del graph


def test_native_pooled_adapter_causal_expansion_and_replay():
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.lim_copy_sfa_contract import LimCopySfaSpec
    from sglang.srt.hardware_backend.npu.lim_copy_sfa_indexer import NativePooledIndexer

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    row_count = 7
    spec = LimCopySfaSpec(16384, 6, 12416)
    indexer = NativePooledIndexer(spec, row_count, "npu:0")
    generator = torch.Generator().manual_seed(99)
    query = torch.randn(row_count, 32, 128, generator=generator).bfloat16().to("npu")
    keys = torch.randn(33, 128, 1, 128, generator=generator).bfloat16().to("npu")
    weights = torch.rand(row_count, 32, generator=generator).bfloat16().to("npu")
    table = torch.arange(1, 33, dtype=torch.int32, device="npu").repeat(row_count, 1)
    initial = torch.tensor([0, 1, 3, 2048, 2051, 4099, 16384], dtype=torch.int32)
    lengths = initial.to("npu")

    def check(topk, scores):
        torch.npu.synchronize()
        lengths_cpu = lengths.cpu()
        selected = topk.cpu()
        pools = indexer.indices.cpu()
        score_cpu = scores.cpu()
        for row, visible in enumerate(lengths_cpu.tolist()):
            if visible <= 2048:
                expected = torch.arange(visible, dtype=torch.int32)
            else:
                expected = torch.cat(
                    (
                        (pools[row, :, None] * 4 + torch.arange(4)).flatten(),
                        torch.arange(visible // 4 * 4, visible),
                    )
                ).to(torch.int32)
            valid = selected[row][selected[row] >= 0]
            torch.testing.assert_close(valid, expected, rtol=0, atol=0)
            assert torch.isfinite(score_cpu[row, : visible // 4]).all()
            assert torch.isneginf(score_cpu[row, visible // 4 :]).all()

    check(*indexer.select(query, keys, weights, lengths, table))
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            topk, scores = indexer.select(query, keys, weights, lengths, table)
        for shift in (1, 2, 3):
            lengths.copy_(torch.roll(initial, shifts=shift))
            graph.replay()
            check(topk, scores)
    finally:
        torch.npu.synchronize()
        del graph


def test_native_indexer_keeps_high_pooled_ids_at_one_million_tokens():
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.lim_copy_sfa_contract import LimCopySfaSpec
    from sglang.srt.hardware_backend.npu.lim_copy_sfa_indexer import NativePooledIndexer

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    tokens = 1 << 20
    pools = tokens // 4
    indexer = NativePooledIndexer(LimCopySfaSpec(tokens, 1, 4096), 1, "npu:0")
    query = torch.zeros((1, 32, 128), dtype=torch.bfloat16, device="npu")
    weights = torch.zeros((1, 32), dtype=torch.bfloat16, device="npu")
    key = torch.zeros(
        (pools // 128 + 1, 128, 1, 128), dtype=torch.bfloat16, device="npu"
    )
    query[0, 0, 0] = 1
    weights[0, 0] = 1
    key[-4:, :, 0, 0] = 1
    table = torch.arange(1, pools // 128 + 1, dtype=torch.int32, device="npu")[None]
    lengths = torch.tensor([tokens], dtype=torch.int32, device="npu")
    selected, scores = indexer.select(query, key, weights, lengths, table)
    torch.testing.assert_close(
        selected[0, :2048].cpu().sort().values,
        torch.arange(tokens - 2048, tokens, dtype=torch.int32),
    )
    assert torch.all(selected[0, 2048:].cpu() == -1)
    assert torch.all(scores[0, -512:].cpu() == 1)
    assert torch.all(scores[0, :-512].cpu() == 0)
