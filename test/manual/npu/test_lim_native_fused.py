"""Fused native Indexer/manager selection, lifecycle and replay qualification."""

import os

import pytest
import torch

from test_lim_copy_sfa_device_plan import reference_module

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an explicitly authorized NPU component environment",
)


@pytest.mark.parametrize("batch,routes", [(1, 1), (1, 6), (10, 1), (10, 6)])
def test_native_fused_attention_chain(batch, routes):
    from test_lim_copy_sfa_component import run_request_shared_attention

    run_request_shared_attention(batch, routes, include_indexer=True, native_fused=True)


@pytest.mark.parametrize("batch,routes", [(1, 1), (1, 6), (10, 1), (10, 6)])
def test_fused_native_indexer_and_manager(batch, routes):
    import torch_npu

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import LimCopySfaSpec
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeIndexerManager,
    )

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    spec = LimCopySfaSpec(16384, routes, 4096 if routes == 1 else 12416)
    planner = NativeIndexerManager(spec, batch + 1, batch, "npu:0")
    rows = batch * routes
    generator = torch.Generator().manual_seed(44)
    query = torch.randn(rows, 64, 128, generator=generator).bfloat16().to("npu")
    keys = (
        torch.randn(batch * 32 + 1, 128, 1, 128, generator=generator)
        .bfloat16()
        .to("npu")
    )
    weights = torch.rand(rows, 64, generator=generator).bfloat16().to("npu")
    index_table = (
        torch.arange(1, batch * 32 + 1, dtype=torch.int32, device="npu")
        .view(batch, 32)
        .repeat_interleave(routes, dim=0)
    )
    host_table_cpu = torch.arange(1, batch * 128 + 1, dtype=torch.int32).view(
        batch, 128
    )
    host_table = host_table_cpu.to("npu")
    lengths_cpu = torch.tensor(
        [16384 - routes + route + 1 for _ in range(batch) for route in range(routes)],
        dtype=torch.int32,
    )
    lengths = lengths_cpu.to("npu")
    ends = torch.arange(routes, rows + 1, routes, dtype=torch.int32, device="npu")
    requests = torch.arange(1, batch + 1, dtype=torch.int64, device="npu")
    generations = torch.ones(batch + 1, dtype=torch.int64, device="npu")
    states = [
        reference_module().ReferenceResident(
            spec.resident_tokens, spec.source_capacity, "protected_hash"
        )
        for _ in range(batch)
    ]
    graph = torch.npu.NPUGraph()

    def invoke():
        output = planner.select_and_plan(
            query,
            keys,
            weights,
            index_table,
            ends,
            lengths,
            host_table,
            requests,
            generations,
            host_tokens=(batch * 128 + 1) * 128,
        )
        planner.commit(requests, ends, lengths)
        return output

    invoke()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        output_ids, output_slots, membership = invoke()
    generations.add_(1)
    try:
        for iteration in range(8):
            if iteration == 2:
                query.mul_(0.875).add_(0.03125)
            if iteration == 3:
                host_table_cpu = host_table_cpu.flip(1).contiguous()
                host_table.copy_(host_table_cpu)
            if iteration == 4:
                lengths_cpu = lengths_cpu.view(batch, routes).flip(1).flatten()
                lengths.copy_(lengths_cpu)
            if iteration == 5:
                lengths_cpu.sub_(128)
                lengths.copy_(lengths_cpu)
            if iteration == 6:
                generations.add_(1)
            if iteration == 7 and routes > 1:
                index_table[1::routes].copy_(index_table[1::routes].flip(1))
            graph.replay()
            torch.npu.synchronize()
            baseline, _ = torch_npu.npu_lightning_indexer(
                query=query,
                key=keys,
                weights=weights,
                actual_seq_lengths_query=planner.indexer.ends,
                actual_seq_lengths_key=planner.indexer.lengths,
                block_table=index_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=3,
            )
            actual_ids = output_ids.cpu()
            status = membership.cpu()
            order = status[:, 4096:6144].long()
            mixed = (status[:, 16394] > 0) & (status[:, 16394] < 2048)
            order[~mixed] = torch.arange(2048)
            torch.testing.assert_close(
                order.sort(dim=1).values,
                torch.arange(2048).expand(rows, -1),
            )
            ordered_plan = status[:, 14333:16381].gather(1, order)
            ordered_miss = torch.arange(2048)[None, :] < status[:, 16394, None]
            torch.testing.assert_close(ordered_plan < 0, ordered_miss)
            for row in range(rows):
                expected_presence = 0
                for tail in range(12):
                    token = int(lengths_cpu[row]) - 12 + tail
                    if token >= 0 and bool((actual_ids[row] == token).any()):
                        expected_presence |= 1 << tail
                actual_presence = (int(status[row, 16392]) & 0xFFFF) | (
                    (int(status[row, 16393]) & 0xFFFF) << 16
                )
                assert actual_presence == expected_presence
            union_slots = planner.union_slots.cpu()
            union_counts = planner.state[1 : batch + 1, 3].cpu()
            assert torch.all(
                (union_counts >= 0) & (union_counts <= planner.union_width)
            )
            slots_cpu = output_slots.cpu()
            misses_cpu = membership[:, 14333:16384].cpu() < 0
            for request in range(batch):
                subset = slice(request * routes, (request + 1) * routes)
                expected_union = (
                    slots_cpu[subset][misses_cpu[subset]].unique().sort().values
                )
                actual_union = (
                    union_slots[request, : int(union_counts[request])].sort().values
                )
                torch.testing.assert_close(actual_union, expected_union)
            torch.testing.assert_close(
                planner.indexer.indices.cpu().sort(-1).values,
                baseline.squeeze(1).cpu().sort(-1).values,
                atol=0,
                rtol=0,
            )
            raw = planner.indexer.raw_scores.cpu().long() & 0xFFFFFFFF
            bits = torch.where(raw & 0x80000000 != 0, raw ^ 0x80000000, ~raw)
            scores = bits.to(torch.int32).view(torch.float32)
            for row in range(rows):
                scores[row, int(lengths_cpu[row]) // 4 :] = -torch.inf
                selected_pools = planner.indexer.indices[row].cpu().long()
                expected = torch.cat(
                    (
                        (selected_pools[:, None] * 4 + torch.arange(4)).flatten(),
                        torch.arange(
                            int(lengths_cpu[row]) // 4 * 4, int(lengths_cpu[row])
                        ),
                    )
                ).to(torch.int32)
                torch.testing.assert_close(
                    actual_ids[row, : expected.numel()], expected
                )
                assert torch.all(actual_ids[row, expected.numel() :] == -1)
            for request in range(batch):
                subset = slice(request * routes, (request + 1) * routes)
                plan = states[request].plan(
                    actual_ids[subset],
                    host_table_cpu[request],
                    lengths_cpu[subset].tolist(),
                    3 if iteration >= 6 else 2,
                    scores=scores[subset],
                )
                torch.testing.assert_close(
                    output_slots[subset].cpu().long(), plan.slots
                )
                torch.testing.assert_close(
                    membership[subset, 14333:16384].cpu() < 0, plan.misses
                )
                states[request].commit(plan)
    finally:
        torch.npu.synchronize()
        graph.reset()
