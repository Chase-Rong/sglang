"""Fused native Indexer/manager selection, lifecycle and replay qualification."""

import pytest
import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.npu_lim_reference import ReferenceResident
from sglang.test.npu_offload_device import offload_test_device
from sglang.test.npu_test_paths import run_pytest

register_npu_ci(
    est_time=30,
    suite="base-a-test-npu",
    disabled="Opt-in matched A5 LIM/CopySFA and MemFabric libraries require an idle device",
)


@pytest.fixture
def device():
    return offload_test_device(extensions=("lim",))


@pytest.fixture
def host_device():
    return offload_test_device(extensions=("lim", "fsa"), memfabric=True)


@pytest.mark.parametrize("batch,routes", [(1, 1), (2, 6)])
def test_fused_native_indexer_and_manager(device, batch, routes):
    import torch_npu

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import (
        LimCopySfaSpec,
    )
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeIndexerManager,
    )

    spec = LimCopySfaSpec(16384, routes, 4096 if routes == 1 else 12416)
    planner = NativeIndexerManager(spec, batch + 1, batch, device)
    rows = batch * routes
    generator = torch.Generator().manual_seed(44)
    query = torch.randn(rows, 64, 128, generator=generator).bfloat16().to(device)
    keys = (
        torch.randn(batch * 32 + 1, 128, 1, 128, generator=generator)
        .bfloat16()
        .to(device)
    )
    weights = torch.rand(rows, 64, generator=generator).bfloat16().to(device)
    index_table = (
        torch.arange(1, batch * 32 + 1, dtype=torch.int32, device=device)
        .view(batch, 32)
        .repeat_interleave(routes, dim=0)
    )
    host_table_cpu = torch.arange(1, batch * 128 + 1, dtype=torch.int32).view(
        batch, 128
    )
    host_table = host_table_cpu.to(device)
    lengths_cpu = torch.tensor(
        [16384 - routes + route + 1 for _ in range(batch) for route in range(routes)],
        dtype=torch.int32,
    )
    lengths = lengths_cpu.to(device)
    ends = torch.arange(routes, rows + 1, routes, dtype=torch.int32, device=device)
    requests = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    generations = torch.ones(batch + 1, dtype=torch.int64, device=device)
    states = [
        ReferenceResident(spec.resident_tokens, spec.source_capacity, "protected_hash")
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


@pytest.mark.parametrize(
    "batch,routes,kv_length,inactive_last",
    [
        (1, 1, 16384, False),
        (1, 6, 2047, False),
        (1, 6, 2048, False),
        (2, 6, 2049, False),
        (2, 6, 24, True),
    ],
    ids=["decode", "mask2047", "mask2048", "mask2049", "padding"],
)
def test_native_pair(host_device, batch, routes, kv_length, inactive_last):
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_attention import (
        NativeCopySparseAttention,
    )
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import (
        LimCopySfaSpec,
    )
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeIndexerManager,
    )
    from sglang.srt.hardware_backend.npu.memfabric_host_kv import MemFabricHostKvPool
    from sglang.test.npu_lim_reference import copy_attention

    device = host_device
    context, rows = 16384, batch * routes
    spec = LimCopySfaSpec(context, routes, 4096 if routes == 1 else 12416)
    manager = NativeIndexerManager(spec, batch + 1, batch, device)
    attention = NativeCopySparseAttention(device, row_capacity=rows)
    resident = torch.empty(
        batch + 1, spec.resident_tokens, 512, dtype=torch.bfloat16, device=device
    )
    pages = batch * context // 128 + 1
    pool = MemFabricHostKvPool(device.index, required_bytes=pages * 128 * 512 * 2)
    graph = None
    try:
        region = pool.allocate_component(
            "kv", (pages * 128, 512), torch.bfloat16, page_size=128
        ).region
        assert pool.physical_bytes == 1 << 30
        generator = torch.Generator().manual_seed(615)
        source = (torch.randn(pages * 128, 512, generator=generator) * 0.1).bfloat16()
        region.tensor.copy_(source)
        query_cpu = (torch.randn(rows, 8, 512, generator=generator) * 0.1).bfloat16()
        query = query_cpu.to(device)
        index_query = (
            (torch.randn(rows, 32, 128, generator=generator) * 0.1)
            .bfloat16()
            .to(device)
        )
        index_keys = (
            (torch.randn(batch * 32 + 1, 128, 1, 128, generator=generator) * 0.1)
            .bfloat16()
            .to(device)
        )
        weights = torch.rand(rows, 32, generator=generator).bfloat16().to(device)
        index_table = (
            torch.arange(1, batch * 32 + 1, dtype=torch.int32)
            .view(batch, 32)
            .repeat_interleave(routes, dim=0)
            .to(device)
        )
        table_cpu = torch.arange(1, pages, dtype=torch.int32).view(
            batch, context // 128
        )
        table = table_cpu.to(device)
        lengths_cpu = torch.tensor(
            [
                kv_length - routes + route + 1
                for _ in range(batch)
                for route in range(routes)
            ],
            dtype=torch.int32,
        )
        lengths = lengths_cpu.to(device)
        kv_lengths = torch.full((batch,), kv_length, dtype=torch.int32, device=device)
        ends = torch.arange(routes, rows + 1, routes, dtype=torch.int32, device=device)
        requests = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
        if inactive_last:
            requests[-1] = 0
        generations = torch.ones(batch + 1, dtype=torch.int64, device=device)
        states = [
            ReferenceResident(spec.resident_tokens, context, "protected_hash")
            for _ in range(batch)
        ]
        reference_kv = torch.zeros(
            batch, spec.resident_tokens, 512, dtype=torch.bfloat16
        )
        generation = 1

        def invoke():
            manager.select_and_plan(
                index_query,
                index_keys,
                weights,
                index_table,
                ends,
                lengths,
                table,
                requests,
                generations,
                host_tokens=pages * 128,
            )
            return attention.forward(
                query,
                region.tensor,
                region,
                resident,
                manager,
                ends,
                lengths,
                kv_lengths,
                table,
                requests,
                scale=0.0625,
            )

        def check(output):
            torch.npu.synchronize()
            selected = manager.topk.cpu()
            actual_slots = manager.slots.cpu()
            payload = resident.cpu()
            expected = torch.zeros_like(query_cpu)
            for request in range(batch):
                subset = slice(request * routes, (request + 1) * routes)
                active = not (inactive_last and request == batch - 1)
                plan = states[request].plan(
                    selected[subset],
                    table_cpu[request],
                    lengths_cpu[subset].tolist(),
                    generation,
                    active=active,
                )
                torch.testing.assert_close(selected[subset].long(), plan.logical_ids)
                torch.testing.assert_close(actual_slots[subset].long(), plan.slots)
                expected[subset] = copy_attention(
                    query_cpu[subset], source, reference_kv[request], plan, 0.0625
                )
                valid = plan.slots >= 0
                torch.testing.assert_close(
                    payload[request + 1][plan.slots[valid]],
                    source[plan.physical_ids[valid]],
                    rtol=0,
                    atol=0,
                )
                states[request].commit(plan)
            torch.testing.assert_close(output.cpu(), expected, atol=0.002, rtol=0.025)

        check(invoke())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = invoke()
        generations.add_(1)
        generation += 1
        addresses = tuple(
            tensor.data_ptr()
            for tensor in (
                manager.topk,
                manager.slots,
                manager.membership,
                resident,
                output,
            )
        )
        for step in range(6):
            if step == 2:
                for request in range(batch):
                    logical = int(lengths_cpu[(request + 1) * routes - 1]) - 1
                    physical = (
                        int(table_cpu[request, logical // 128]) * 128 + logical % 128
                    )
                    source[physical].add_(0.125)
                region.tensor.copy_(source)
            if step == 3:
                table_cpu = table_cpu.flip(1).contiguous()
                table.copy_(table_cpu)
            if step == 4:
                generations.add_(1)
                generation += 1
            if step == 5:
                lengths_cpu.sub_(1)
                lengths.copy_(lengths_cpu)
                kv_lengths.sub_(1)
            graph.replay()
            check(output)
            assert addresses == tuple(
                tensor.data_ptr()
                for tensor in (
                    manager.topk,
                    manager.slots,
                    manager.membership,
                    resident,
                    output,
                )
            )
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        pool.close(drained=True)


if __name__ == "__main__":
    run_pytest(__file__)
