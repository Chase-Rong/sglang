"""Grouped LIM scores retain independent KPool4 causal limits and Top512.

At most 512 visible pools use the exact mathematical selection as the oracle;
the installed Indexer leaves output uninitialized for an all-zero-length batch.
Variable routes use the serving six-route reservation, with empty request groups.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an explicitly authorized NPU component environment",
)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize(
    "request_rows",
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (1, 6),
        (3, 2),
        (0, 6),
        (6, 0),
        (2, 0, 3),
    ],
)
def test_grouped_lim_per_query_causality_and_table_fallback(heads, request_rows):
    import torch_npu

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import LimCopySfaSpec
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeIndexerManager,
    )

    torch.npu.set_device(0)
    torch.npu.empty_cache()
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    device = "npu:0"
    context, batch = 32768, len(request_rows)
    rows, routes = sum(request_rows), max(request_rows)
    spec = LimCopySfaSpec(
        context, 1 if routes == 1 else 6, 4096 if routes == 1 else 12416
    )
    planner = NativeIndexerManager(spec, batch + 1, batch, device)
    generator = torch.Generator().manual_seed(9941 + rows + heads)
    query = (
        torch.randn(rows, heads, 128, generator=generator)
        .mul_(0.1)
        .bfloat16()
        .to(device)
    )
    weights = torch.rand(rows, heads, generator=generator).bfloat16().to(device)
    pooled_pages = context // 512
    keys = (
        torch.randn(batch * pooled_pages + 1, 128, 1, 128, generator=generator)
        .mul_(0.1)
        .bfloat16()
        .to(device)
    )
    shared_table = torch.cat(
        [
            torch.arange(
                request * pooled_pages + 1,
                (request + 1) * pooled_pages + 1,
                dtype=torch.int32,
            ).repeat(count, 1)
            for request, count in enumerate(request_rows)
        ]
    )
    index_table = shared_table.to(device)
    host_table_cpu = torch.arange(
        1, batch * context // 128 + 1, dtype=torch.int32
    ).view(batch, -1)
    host_table = host_table_cpu.to(device)
    ends = torch.tensor(request_rows, dtype=torch.int32).cumsum(0).int().to(device)
    requests = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    generations = torch.ones(batch + 1, dtype=torch.int64, device=device)
    baseline_ends = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    lengths_cpu = context - 1 - torch.arange(rows, dtype=torch.int32) % 7
    lengths = lengths_cpu.to(device)
    baseline_lengths = (lengths_cpu // 4).to(device)
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
            host_tokens=(batch * context // 128 + 1) * 128,
        )
        planner.commit(requests, ends, lengths)
        return output

    try:
        invoke()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            output_ids, output_slots, membership = invoke()
        for phase in (
            "long",
            "boundaries",
            "zero",
            "long_again",
            "different_tables",
            "restored",
        ):
            if phase == "boundaries":
                boundaries = torch.tensor(
                    [0, 1, 3, 4, 2047, 2048, 2049, 2051, 8191, context],
                    dtype=torch.int32,
                )
                lengths_cpu = boundaries[torch.arange(rows) % boundaries.numel()]
            elif phase == "zero":
                lengths_cpu = torch.zeros(rows, dtype=torch.int32)
            else:
                lengths_cpu = context - 1 - torch.arange(rows, dtype=torch.int32) % 7
            lengths.copy_(lengths_cpu)
            baseline_lengths.copy_(lengths_cpu // 4)
            query.copy_(
                torch.randn(rows, heads, 128, generator=generator).mul_(0.1).bfloat16()
            )
            current_table = shared_table.clone()
            if phase == "different_tables":
                start = 0
                for count in request_rows:
                    if count > 1:
                        current_table[start + 1] = current_table[start + 1].flip(0)
                    start += count
            index_table.copy_(current_table)
            graph.replay()
            torch.npu.synchronize()
            baseline, _ = torch_npu.npu_lightning_indexer(
                query=query,
                key=keys,
                weights=weights,
                actual_seq_lengths_query=baseline_ends,
                actual_seq_lengths_key=baseline_lengths,
                block_table=index_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=3,
            )
            native_pools = planner.indexer.indices[:rows].cpu().long()
            expected_pools = baseline.cpu().reshape(rows, 512).long()
            for row in range(rows):
                pool_count = int(lengths_cpu[row]) // 4
                if pool_count <= 512:
                    expected_pools[row].fill_(-1)
                    expected_pools[row, :pool_count] = torch.arange(pool_count)
            torch.testing.assert_close(
                native_pools.sort(dim=1).values,
                expected_pools.sort(dim=1).values,
                atol=0,
                rtol=0,
                msg=lambda message: f"{phase}: {message}",
            )
            ids, slots = output_ids.cpu().long(), output_slots.cpu().long()
            plans = membership[:, 14333:16384].cpu().long()
            mapping, sources = planner.source_map.cpu(), planner.sources.cpu()
            first = 0
            for request, count in enumerate(request_rows):
                for row in range(first, first + count):
                    length = int(lengths_cpu[row])
                    pooled = expected_pools[row][expected_pools[row] >= 0]
                    assert pooled.numel() == min(512, length // 4), (phase, row)
                    assert (pooled < length // 4).all(), (phase, row)
                    expected = (
                        torch.arange(length)
                        if length <= 2048
                        else torch.cat(
                            (
                                (pooled[:, None] * 4 + torch.arange(4)).flatten(),
                                torch.arange(length // 4 * 4, length),
                            )
                        )
                    )
                    valid = ids[row] >= 0
                    torch.testing.assert_close(
                        ids[row, valid].sort().values, expected.sort().values
                    )
                    assert (slots[row, ~valid] == -1).all()
                    assert (
                        (slots[row, valid] >= 0)
                        & (slots[row, valid] < spec.resident_tokens)
                    ).all()
                    torch.testing.assert_close(
                        mapping[request + 1, ids[row, valid]].long(), slots[row, valid]
                    )
                    torch.testing.assert_close(
                        sources[request + 1, slots[row, valid]].long(), ids[row, valid]
                    )
                    torch.testing.assert_close(
                        plans[row, valid].abs() - 1, slots[row, valid]
                    )
                first += count
    finally:
        torch.npu.synchronize()
        graph.reset()
