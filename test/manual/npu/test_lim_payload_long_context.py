"""Long-context native LIM payload qualification without Host KV allocation.

The metadata-only commit simulates completed resident writes. This validates
selection, slot payload and residency bookkeeping, not CopySFA data transfers.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an explicitly authorized NPU component environment",
)


@pytest.mark.parametrize("context", [131072, 1048576])
@pytest.mark.parametrize("batch,routes", [(1, 1), (2, 6)])
@pytest.mark.parametrize("heads", [32, 64])
def test_fused_lim_long_context_slot_payload(context, batch, routes, heads):
    import torch_npu

    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import LimCopySfaSpec
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeIndexerManager,
        NativeUnionPlanner,
    )

    device = "npu:0"
    torch.npu.set_device(0)
    torch.npu.empty_cache()
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    rows = batch * routes
    pooled_capacity = context // 4
    pooled_pages = pooled_capacity // 128
    host_pages = context // 128
    spec = LimCopySfaSpec(context, routes, 4096 if routes == 1 else 12416)
    planner = NativeIndexerManager(spec, batch + 1, batch, device)
    reference = NativeUnionPlanner(spec, batch + 1, batch, device)
    generator = torch.Generator().manual_seed(71123 + context + rows)
    query_cpu = torch.randn(rows, heads, 128, generator=generator).mul_(0.1)
    query_cpu[:, :, 0] = 0.125
    query = query_cpu.bfloat16().to(device)
    weights = (
        torch.rand(rows, heads, generator=generator)
        .mul_(0.1)
        .add_(0.1)
        .bfloat16()
        .to(device)
    )
    keys_cpu = torch.randn(
        batch * pooled_pages + 1,
        128,
        1,
        128,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.1)
    anchors = torch.arange(0, pooled_capacity, 8192, dtype=torch.long) + 128
    for request in range(batch):
        physical_pools = (request * pooled_pages + 1) * 128 + anchors
        keys_cpu.view(-1, 128)[physical_pools, 0] = 16
    keys = keys_cpu.to(device)
    del keys_cpu
    index_table = (
        torch.arange(1, batch * pooled_pages + 1, dtype=torch.int32)
        .view(batch, pooled_pages)
        .repeat_interleave(routes, dim=0)
        .to(device)
    )
    host_table_cpu = torch.arange(1, batch * host_pages + 1, dtype=torch.int32).view(
        batch, host_pages
    )
    host_table = host_table_cpu.to(device)
    lengths_cpu = torch.tensor(
        [context - routes + route - 1 for _ in range(batch) for route in range(routes)],
        dtype=torch.int32,
    )
    lengths = lengths_cpu.to(device)
    baseline_ends = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    baseline_lengths = (lengths_cpu // 4).to(device)
    ends = torch.arange(routes, rows + 1, routes, dtype=torch.int32, device=device)
    requests = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    generations = torch.ones(batch + 1, dtype=torch.int64, device=device)
    host_tokens = (batch * host_pages + 1) * 128
    graph = torch.npu.NPUGraph()

    def invoke():
        outputs = planner.select_and_plan(
            query,
            keys,
            weights,
            index_table,
            ends,
            lengths,
            host_table,
            requests,
            generations,
            host_tokens=host_tokens,
        )
        planner.commit(requests, ends, lengths)
        return outputs

    try:
        invoke()
        invoke()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            output_ids, output_slots, membership = invoke()
        torch.npu.synchronize()
        prior_selection = None
        phases = (
            "warm",
            "query_changed",
            "rewarm",
            "remapped",
            "remap_rewarm",
            "query_changed_again",
            "final_rewarm",
        )
        for phase in phases:
            if phase in ("query_changed", "query_changed_again"):
                query_cpu = torch.randn(rows, heads, 128, generator=generator).mul_(0.1)
                query_cpu[:, :, 0] = 0.125
                query.copy_(query_cpu.bfloat16())
            if phase == "remapped":
                host_table_cpu = host_table_cpu.flip(1).contiguous()
                host_table.copy_(host_table_cpu)
            old_mapping = planner.source_map.cpu()
            old_physical = planner.physical.cpu()
            for name in ("source_map", "sources", "physical", "state"):
                getattr(reference, name).copy_(getattr(planner, name))
            graph.replay()
            torch.npu.synchronize()
            reference.plan(
                planner.indexer.indices,
                planner.indexer.raw_scores,
                ends,
                lengths,
                host_table,
                requests,
                generations,
                host_tokens=host_tokens,
                pooled=True,
                encoded=True,
            )
            torch.npu.synchronize()
            for candidate, expected in (
                (output_ids, reference.topk),
                (output_slots, reference.slots),
                (
                    planner.membership[:, 14333:16384],
                    reference.membership[:, 14333:16384],
                ),
                (planner.source_map, reference.source_map),
                (planner.sources, reference.sources),
                (planner.state[:, :4], reference.state[:, :4]),
            ):
                torch.testing.assert_close(
                    candidate.cpu(), expected.cpu(), atol=0, rtol=0
                )
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
            expected_pools = baseline.cpu().reshape(rows, 512).long()
            actual_pools = planner.indexer.indices.cpu().long()
            actual_ids = output_ids.cpu().long()
            actual_slots = output_slots.cpu().long()
            plans = membership[:, 14333:16384].cpu().long()
            mapping = planner.source_map.cpu()
            sources = planner.sources.cpu()
            physical = planner.physical.cpu()
            counts = planner.state[1 : batch + 1, 3].cpu().long()
            union_slots = planner.union_slots.cpu().long()
            union_sources = planner.union_sources.cpu().long()
            union_physical = planner.union_physical.cpu()
            selection = actual_pools.sort(dim=1).values
            torch.testing.assert_close(
                selection, expected_pools.sort(dim=1).values, atol=0, rtol=0
            )
            if phase in ("query_changed", "query_changed_again"):
                assert not torch.equal(selection, prior_selection), phase
            else:
                if prior_selection is not None:
                    assert torch.equal(selection, prior_selection), phase
            prior_selection = selection.clone()

            for row in range(rows):
                request, owner = row // routes, row // routes + 1
                length = int(lengths_cpu[row])
                pools = expected_pools[row]
                assert pools.unique().numel() == 512
                assert torch.isin(anchors, pools).all(), (phase, row)
                assert (pools < 8192).any() and (pools >= 8192).any()
                expected_ids = torch.cat(
                    (
                        (pools[:, None] * 4 + torch.arange(4)).flatten(),
                        torch.arange(length // 4 * 4, length),
                    )
                )
                valid = actual_ids[row] >= 0
                tokens = actual_ids[row, valid]
                slots = actual_slots[row, valid]
                signed = plans[row, valid]
                torch.testing.assert_close(
                    tokens.sort().values, expected_ids.sort().values, atol=0, rtol=0
                )
                assert tokens.unique().numel() == tokens.numel()
                assert torch.all(actual_slots[row, ~valid] == -1)
                assert torch.all((slots >= 0) & (slots < spec.resident_tokens))
                torch.testing.assert_close(signed.abs() - 1, slots, atol=0, rtol=0)
                torch.testing.assert_close(mapping[owner, tokens].long(), slots)
                torch.testing.assert_close(sources[owner, slots].long(), tokens)
                address = host_table_cpu[request, tokens // 128].long() * 128
                address += tokens % 128
                previous_slots = old_mapping[owner, tokens].long()
                previous_valid = (previous_slots >= 0) & (
                    previous_slots < spec.resident_tokens
                )
                previous_address = old_physical[
                    owner, previous_slots.clamp(0, spec.resident_tokens - 1)
                ]
                expected_hit = (
                    previous_valid
                    & (previous_address == address)
                    & (tokens < length - 12)
                )
                torch.testing.assert_close(signed > 0, expected_hit)
                if phase == "remapped":
                    assert not expected_hit.any(), (phase, row)
                if phase in ("warm", "rewarm", "remap_rewarm", "final_rewarm"):
                    stable_all_routes = tokens < int(lengths_cpu[request * routes]) - 12
                    assert expected_hit[stable_all_routes].all(), (phase, row)
                if context == 1048576:
                    assert (tokens >= (1 << 18)).any(), (phase, row)

            for request in range(batch):
                owner = request + 1
                subset = slice(request * routes, (request + 1) * routes)
                misses = plans[subset] < 0
                expected_union = actual_slots[subset][misses].unique().sort().values
                count = int(counts[request])
                assert 0 <= count <= planner.union_width
                torch.testing.assert_close(
                    union_slots[request, :count].sort().values, expected_union
                )
                copy_tokens = union_sources[request, :count]
                copy_slots = union_slots[request, :count]
                torch.testing.assert_close(
                    sources[owner, copy_slots].long(), copy_tokens
                )
                expected_address = (
                    host_table_cpu[request, copy_tokens // 128].long() * 128
                )
                expected_address += copy_tokens % 128
                torch.testing.assert_close(
                    union_physical[request, :count], expected_address
                )
                valid_tokens = actual_ids[subset]
                unique_tokens = valid_tokens[valid_tokens >= 0].unique()
                unique_slots = mapping[owner, unique_tokens].long()
                reusable = torch.ones_like(unique_tokens, dtype=torch.bool)
                for row in range(request * routes, (request + 1) * routes):
                    selected = actual_ids[row]
                    mutable = selected[
                        (selected >= int(lengths_cpu[row]) - 12)
                        & (selected < int(lengths_cpu[row]))
                    ]
                    reusable &= ~torch.isin(unique_tokens, mutable)
                expected_physical = (
                    host_table_cpu[request, unique_tokens // 128].long() * 128
                    + unique_tokens % 128
                )
                expected_physical[~reusable] = -1
                torch.testing.assert_close(
                    physical[owner, unique_slots], expected_physical
                )
                resident_sources = sources[owner][sources[owner] >= 0]
                assert resident_sources.unique().numel() == resident_sources.numel()
    finally:
        torch.npu.synchronize()
        graph.reset()
