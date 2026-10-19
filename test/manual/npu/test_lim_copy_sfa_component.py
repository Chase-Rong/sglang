"""Actual Host-backed NoPE attention using a request-shared device plan."""

import hashlib
import os
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an explicitly authorized NPU component environment",
)


@pytest.mark.parametrize("batch,routes", [(1, 1), (1, 6), (10, 1), (10, 6)])
def test_request_shared_copy_attention_eager_and_graph(batch, routes):
    run_request_shared_attention(batch, routes)


def run_request_shared_attention(
    batch,
    routes,
    *,
    include_indexer=False,
    native_fused=False,
    native_copy=False,
    framework=False,
    inactive_last=False,
    reserved_context=16384,
    kv_length=16384,
    three_way=False,
    attention_heads=8,
    changing_topk=False,
    context=16384,
    indexer_heads=64,
    pooled_shift=103,
):
    import torch_npu  # noqa: F401

    if three_way and (
        not include_indexer
        or not native_fused
        or not native_copy
        or framework
        or not os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK")
    ):
        raise ValueError(
            "Three-way benchmarking requires the native pair and a results directory"
        )

    from sglang.srt.hardware_backend.npu.hisparse.decode_host_kv import DecodeHostKvBacking
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import LimCopySfaSpec
    from lim_copy_sfa_reference import LegacyFsaComparator

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    if context % 512 or not routes < kv_length <= context:
        raise ValueError(
            "Context must be page aligned and cover the selected KV length"
        )
    pages_per_request = context // 128
    index_pages_per_request = context // 512
    pages = batch * pages_per_request + 1
    backing = DecodeHostKvBacking(
        layers=1,
        pages=pages,
        page_size=128,
        latent_dim=512,
        rope_dim=0,
        dtype=torch.bfloat16,
        device_id=0,
        host_budget=max(1 << 30, pages * 128 * 1024 + (1 << 20)),
        integrated=True,
    )
    region = backing.regions[0][0]
    source = (
        torch.randn(pages * 128, 512, generator=torch.Generator().manual_seed(11))
        .mul_(0.1)
        .bfloat16()
    )
    region.tensor.copy_(source)
    spec = LimCopySfaSpec(context, routes, 4096 if routes == 1 else 12416)
    options = {}
    if native_fused:
        from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
            NativeIndexerManager,
        )

        options["planner_type"] = NativeIndexerManager
    component = LegacyFsaComparator(
        spec, batch + 1, batch, "npu:0", torch.bfloat16, **options
    )
    if native_copy:
        from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_attention import (
            NativeCopySparseAttention,
        )

        copy_attention = NativeCopySparseAttention("npu:0", row_capacity=batch * routes)
    rows = batch * routes
    query = (
        torch.randn(
            rows, attention_heads, 512, generator=torch.Generator().manual_seed(12)
        )
        .mul_(0.1)
        .bfloat16()
        .to("npu")
    )
    ids = torch.full((rows, 2051), -1, dtype=torch.int32, device="npu")
    ids[:, :2048] = torch.arange(2048, device="npu")
    lengths = torch.tensor(
        [
            kv_length - routes + route + 1
            for _ in range(batch)
            for route in range(routes)
        ],
        dtype=torch.int32,
        device="npu",
    )
    ids[:, 2048:2050] = lengths[:, None] - 2
    scores = torch.ones(rows, context // 4, device="npu")
    query_ends = torch.arange(routes, rows + 1, routes, device="npu", dtype=torch.int32)
    kv_lengths = torch.full((batch,), kv_length, dtype=torch.int32, device="npu")
    table = torch.arange(1, pages, dtype=torch.int32, device="npu").view(
        batch, pages_per_request
    )
    request_slots = torch.arange(1, batch + 1, dtype=torch.int64, device="npu")
    if inactive_last:
        request_slots[-1] = 0
    generations = torch.ones(batch + 1, dtype=torch.int64, device="npu")
    full_kv, full_rope = backing.layer_views(0)
    if include_indexer:
        import triton

        from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_indexer import (
            NativePooledIndexer,
            _expand,
            _prepare_lengths,
        )

        indexer = (
            component.planner.indexer
            if native_fused
            else NativePooledIndexer(spec, rows, "npu:0")
        )
        index_generator = torch.Generator().manual_seed(615)
        index_query = (
            torch.randn(rows, indexer_heads, 128, generator=index_generator)
            .mul_(0.1)
            .bfloat16()
            .to("npu")
        )
        index_keys = (
            torch.randn(
                batch * index_pages_per_request + 1,
                128,
                1,
                128,
                generator=index_generator,
            )
            .mul_(0.1)
            .bfloat16()
            .to("npu")
        )
        index_weights = (
            torch.rand(rows, indexer_heads, generator=index_generator)
            .bfloat16()
            .to("npu")
        )
        index_table = (
            torch.arange(
                1, batch * index_pages_per_request + 1, dtype=torch.int32, device="npu"
            )
            .view(batch, index_pages_per_request)
            .repeat_interleave(routes, dim=0)
        )
        ids = component.planner.topk if native_fused else indexer.topk
        baseline_ids = torch.empty_like(ids)
        baseline_pool_lengths = torch.empty_like(lengths)

    def change_selection(step):
        index_query.zero_()
        index_query[:, 0, 0] = 1
        index_weights.zero_()
        index_weights[:, 0] = 1
        key_cpu = torch.zeros(index_keys.shape, dtype=torch.bfloat16)
        values = (
            1
            + torch.randperm(
                512, generator=torch.Generator().manual_seed(1700 + step)
            ).float()
            / 512
        )
        for request in range(batch):
            begin = 128 + request * (context // 4) + step * pooled_shift
            key_cpu.view(-1, 128)[begin : begin + 512, 0] = values.bfloat16()
        index_keys.copy_(key_cpu)

    if changing_topk:
        if not three_way:
            raise ValueError("Changing TopK requires the complete three-way chain")
        if (
            not 0 <= pooled_shift <= 512
            or 24 * pooled_shift + 512 > (kv_length - routes) // 4
        ):
            raise ValueError(
                "Changing selection must remain within the causal pooled extent"
            )
        change_selection(0)
    graph = None
    baseline_graph = None
    third_graph = None
    baseline_state = None
    phase_graphs = []
    phase_outputs = []

    if framework:
        from sglang.srt.hardware_backend.npu.hisparse.fsa_backend import forward_fsa
        from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_backend import (
            LimCopySfaState,
            select_offloaded_topk,
        )

        request_pool = SimpleNamespace(
            req_generation=torch.ones(batch + 1, dtype=torch.int64)
        )
        runtime = LimCopySfaState(
            "npu:0",
            request_pool,
            source_tokens=reserved_context,
            routes=routes,
            capacity=spec.resident_tokens,
            batch_capacity=batch,
            layer_anchors={0: 0},
            mutable_tail=12,
        )
        runtime.generations = generations
        runtime_layer = runtime.layer(0)
        component.planner = runtime_layer.planner
        component.kv = runtime_layer.kv
        indexer = runtime.indexer
        ids = component.planner.topk
        backend = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                start_layer=0,
                decode_host_backing=backing,
                get_kv_buffer=lambda _: (full_kv, full_rope),
                get_index_k_with_scale_buffer=lambda _: index_keys,
            ),
            fused_sparse_state=runtime,
            fused_sparse_topk=2051,
            forward_metadata=SimpleNamespace(
                host_page_table=table,
                real_page_table=table,
                cache_seqlens_int32=kv_lengths,
            ),
            _query_lens_device=lambda _: torch.full(
                (batch,), routes, dtype=torch.int32, device="npu"
            ),
        )
        forward_batch = SimpleNamespace(
            positions=lengths - 1,
            req_pool_indices=request_slots,
            forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: False,
                is_decode_or_idle=lambda: routes == 1,
            ),
        )
        token_table = index_table[::routes].repeat_interleave(4, dim=-1)
        index_metadata = SimpleNamespace(
            get_page_table_64=lambda: token_table,
            get_token_to_batch_idx=lambda: torch.arange(
                batch, device="npu"
            ).repeat_interleave(routes),
        )
        index_layer = SimpleNamespace(index_kpool=4, n_heads=indexer_heads)
        attention_layer = SimpleNamespace(layer_id=0, scaling=0.0625)

    def forward(*, selection_only=False, attention_only=False):
        if framework:
            selected = select_offloaded_topk(
                SimpleNamespace(full_attn_backend=backend),
                index_layer,
                forward_batch,
                0,
                index_query,
                index_weights,
                index_metadata,
            )
            return forward_fsa(
                backend,
                query,
                None,
                None,
                attention_layer,
                forward_batch,
                False,
                None,
                None,
                selected,
            )
        if native_fused:
            if not attention_only:
                component.planner.select_and_plan(
                    index_query,
                    index_keys,
                    index_weights,
                    index_table,
                    query_ends,
                    lengths,
                    table,
                    request_slots,
                    generations,
                    host_tokens=pages * 128,
                )
            selected = component.planner.topk[:rows]
            membership = component.planner.membership[:rows]
            if selection_only:
                return selected
            if native_copy:
                return copy_attention.forward(
                    query,
                    full_kv,
                    region,
                    component.kv,
                    component.planner,
                    query_ends,
                    lengths,
                    kv_lengths,
                    table,
                    request_slots,
                    scale=0.0625,
                )
            output = component.attend(
                query,
                selected,
                membership,
                query_ends,
                kv_lengths,
                table,
                full_kv,
                full_rope,
                region,
                scale=0.0625,
            )
            component.planner.commit(request_slots, query_ends, lengths)
            return output
        selected, priorities = (
            indexer.select(index_query, index_keys, index_weights, lengths, index_table)
            if include_indexer and not attention_only
            else (ids, indexer.scores if include_indexer else scores)
        )
        if selection_only:
            return selected
        return component.forward(
            query,
            selected,
            priorities,
            query_ends,
            lengths,
            kv_lengths,
            table,
            request_slots,
            generations,
            full_kv,
            full_rope,
            region,
            scale=0.0625,
        )

    def check(output):
        torch.npu.synchronize()
        selected, query_cpu, actual = (
            ids.cpu(),
            query.cpu().float(),
            output.cpu().float(),
        )
        for row in range(rows):
            logical = selected[row]
            logical = logical[(logical >= 0) & (logical < int(lengths[row].item()))]
            request = row // routes
            mapping = table[request].cpu().long()
            physical = mapping[logical.long() // 128] * 128 + logical.long() % 128
            keys = region.tensor[physical].float()
            expected = torch.softmax(query_cpu[row] @ keys.T * 0.0625, dim=-1) @ keys
            try:
                torch.testing.assert_close(
                    actual[row], expected, atol=0.002, rtol=0.025
                )
            except AssertionError:
                destination = os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK")
                if destination and native_fused and native_copy:
                    path = Path(destination)
                    path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        dict(
                            row=row,
                            context=context,
                            query=query_cpu,
                            actual=actual,
                            expected=expected,
                            keys=keys,
                            selected=selected,
                            lengths=lengths.cpu(),
                            slots=component.planner.slots.cpu(),
                            membership=component.planner.membership.cpu(),
                            state=component.planner.state.cpu(),
                            physical=component.planner.physical.cpu(),
                            resident=component.kv.cpu(),
                        ),
                        path / f"failure_b{batch}_q{routes}.pt",
                    )
                raise

    try:
        check(forward())
        check(forward())
        if framework:
            seed = ids.clone()
            check(
                forward_fsa(
                    backend,
                    query,
                    None,
                    None,
                    attention_layer,
                    forward_batch,
                    False,
                    None,
                    None,
                    seed,
                )
            )
        if native_copy:
            for request in range(batch):
                finish = request * context + 128 + kv_length
                region.tensor[finish - 12 : finish].add_(0.03125)
            check(forward())
            table.copy_(table.flip(-1))
            check(forward())
            table.copy_(table.flip(-1))
            region.tensor.copy_(source)
            generations.add_(1)
            check(forward())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = forward()
        for iteration in range(3):
            if include_indexer:
                index_query.mul_(0.95).add_(iteration * 0.001)
            else:
                ids[:, :2048].copy_(
                    (torch.arange(2048, device="npu") + iteration * 128).flip(0)
                )
            region.tensor.copy_(source + iteration * 0.03125)
            generations.add_(1)
            graph.replay()
            check(output)
        if (
            directory := os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK")
        ) and not inactive_last:
            from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
                FusedSparseAttentionState,
                paged_kv_view,
                run_fused_sparse_attention,
            )

            baseline_state = FusedSparseAttentionState(
                "npu:0",
                SimpleNamespace(
                    req_generation=torch.full((batch + 1,), 4, dtype=torch.int64)
                ),
                external_planner=True,
                resident_capacity=4096,
                row_capacity=rows,
                mutable_tail_tokens=12,
            )
            query_rope = query.new_empty((rows, attention_heads, 0))

            def baseline(*, selection_only=False, attention_only=False):
                if include_indexer and not attention_only:
                    _prepare_lengths[(triton.cdiv(rows, 256),)](
                        lengths, baseline_pool_lengths, rows, 256
                    )
                    pool_ids, _ = torch_npu.npu_lightning_indexer(
                        query=index_query,
                        key=index_keys,
                        weights=index_weights,
                        actual_seq_lengths_query=indexer.ends,
                        actual_seq_lengths_key=baseline_pool_lengths,
                        block_table=index_table,
                        layout_query="TND",
                        layout_key="PA_BSND",
                        sparse_count=512,
                        sparse_mode=3,
                    )
                    _expand[(rows, triton.cdiv(2051, 256))](
                        pool_ids, lengths, baseline_ids, 256
                    )
                if selection_only:
                    return baseline_ids
                inputs = baseline_state.build_inputs(
                    layer_id=0,
                    q_nope=query,
                    q_rope=query_rope,
                    topk_indices=baseline_ids if include_indexer else ids,
                    full_kv=paged_kv_view(full_kv, 128, 512),
                    full_rope=paged_kv_view(full_rope, 128, 0),
                    block_table=table,
                    kv_lengths=kv_lengths,
                    query_lengths=query_ends,
                    request_slots=request_slots,
                    scale=0.0625,
                )
                return run_fused_sparse_attention(
                    replace(inputs, full_kv_region=region)
                )

            check(baseline())
            if include_indexer:
                torch.testing.assert_close(
                    ids.cpu().sort(dim=-1).values,
                    baseline_ids.cpu().sort(dim=-1).values,
                    atol=0,
                    rtol=0,
                )
            baseline_graph = torch.npu.NPUGraph()
            with torch.npu.graph(baseline_graph):
                baseline_output = baseline()
            captured_paths = [("baseline", baseline_graph), ("candidate", graph)]
            if three_way:
                third_component = LegacyFsaComparator(
                    spec,
                    batch + 1,
                    batch,
                    "npu:0",
                    torch.bfloat16,
                    planner_type=NativeIndexerManager,
                )

                def lim_fsa():
                    selected, _, membership = third_component.planner.select_and_plan(
                        index_query,
                        index_keys,
                        index_weights,
                        index_table,
                        query_ends,
                        lengths,
                        table,
                        request_slots,
                        generations,
                        host_tokens=pages * 128,
                    )
                    result = third_component.attend(
                        query,
                        selected,
                        membership,
                        query_ends,
                        kv_lengths,
                        table,
                        full_kv,
                        full_rope,
                        region,
                        scale=0.0625,
                    )
                    third_component.planner.commit(request_slots, query_ends, lengths)
                    return result

                check(lim_fsa())
                check(lim_fsa())
                third_graph = torch.npu.NPUGraph()
                with torch.npu.graph(third_graph):
                    third_output = lim_fsa()
                captured_paths.append(("lim_fsa", third_graph))
            for _ in range(10):
                for _, captured in captured_paths:
                    captured.replay()
            check(baseline_output)
            check(output)
            torch.testing.assert_close(
                baseline_output.cpu(), output.cpu(), atol=0.002, rtol=0.025
            )
            if three_way:
                check(third_output)
                torch.testing.assert_close(
                    third_output.cpu(), output.cpu(), atol=0.002, rtol=0.025
                )
                torch.testing.assert_close(
                    third_component.planner.topk.cpu(), ids.cpu()
                )
            samples = []
            dynamic_misses = []
            for cycle in range(24 if changing_topk else 6):
                if changing_topk:
                    change_selection(cycle + 1)
                offset = cycle % len(captured_paths) if three_way else 0
                order = captured_paths[offset:] + captured_paths[:offset]
                for label, captured in (
                    order if changing_topk else order + list(reversed(order))
                ):
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    replay_count = 1 if changing_topk else 10
                    for _ in range(replay_count):
                        captured.replay()
                    torch.npu.synchronize()
                    samples.append(
                        dict(
                            cycle=cycle,
                            variant=label,
                            us=(time.perf_counter_ns() - start) / 1000 / replay_count,
                        )
                    )
                if changing_topk:
                    check(baseline_output)
                    check(third_output)
                    check(output)
                    torch.testing.assert_close(
                        baseline_output.cpu(), output.cpu(), atol=0.002, rtol=0.025
                    )
                    torch.testing.assert_close(
                        third_output.cpu(), output.cpu(), atol=0.002, rtol=0.025
                    )
                    torch.testing.assert_close(
                        ids.cpu().sort(-1).values, baseline_ids.cpu().sort(-1).values
                    )
                    counts = component.planner.state[1 : batch + 1, 3].cpu().tolist()
                    body_misses = 4 * pooled_shift
                    assert all(
                        body_misses <= count <= body_misses + routes * 12
                        for count in counts
                    )
                    dynamic_misses.append(counts)
            report = dict(
                scope=(
                    "Native LIM + native CopySFA including residency publication vs installed Indexer + expansion + Host planner + FSA; warm fixed-input Graph; not E2E or TPOT"
                    if native_copy
                    else (
                        "Indexer and resident manager + existing CopySFA + commit vs installed Indexer + expansion + Host planner + FSA; warm fixed-input Graph; not final dual-op or TPOT"
                        if include_indexer
                        else "Supplied TopK/scores: planner + CopySFA + commit, warm Graph, shared NPU; not fused Indexer or TPOT"
                    )
                ),
                batch=batch,
                native_fused=native_fused,
                native_copy=native_copy,
                framework=framework,
                query_routes=routes,
                context=context,
                indexer_heads=indexer_heads,
                native_extension=os.environ.get("SGLANG_LIM_NATIVE_EXTENSION"),
                native_extension_sha256=(
                    hashlib.sha256(
                        Path(os.environ["SGLANG_LIM_NATIVE_EXTENSION"]).read_bytes()
                    ).hexdigest()
                    if os.environ.get("SGLANG_LIM_NATIVE_EXTENSION")
                    else None
                ),
                pooled_shift=pooled_shift if changing_topk else 0,
                topk=2051,
                baseline_resident_tokens=batch * routes * 4096,
                candidate_resident_tokens=(batch + 1) * spec.resident_tokens,
                median_us={
                    label: statistics.median(
                        sample["us"] for sample in samples if sample["variant"] == label
                    )
                    for label, _ in captured_paths
                },
                samples=samples,
                candidate_valid=int((ids >= 0).sum().item()),
                candidate_misses=int(
                    (component.planner.membership[:rows, 16384 - 2051 : 16384] < 0)
                    .sum()
                    .item()
                ),
                peak_reserved_bytes=torch.npu.max_memory_reserved(),
            )
            if three_way:
                check(baseline_output)
                check(output)
                check(third_output)
                report.update(
                    scope="Three-way full chains on identical fixed inputs and registered Host KV; warm Graph; not serving TPOT",
                    paths={
                        "baseline": "installed Indexer + expansion + Host planner + existing FSA",
                        "candidate": "native LIM + native CopySFA including validity commit",
                        "lim_fsa": "same native LIM + existing FSA + device validity commit",
                    },
                    warmup_replays=10,
                    replays_per_sample=10,
                    schedule="6 cycles, rotate ABC then reverse; 12 samples per path",
                    lim_fsa_resident_tokens=(batch + 1) * spec.resident_tokens,
                    lim_fsa_valid=int((third_component.planner.topk >= 0).sum().item()),
                    lim_fsa_misses=int(
                        (third_component.planner.membership[:rows, 14333:16384] < 0)
                        .sum()
                        .item()
                    ),
                )
            path = Path(directory)
            if changing_topk:
                report.update(
                    scope="Changing-indexer-input complete chains; one Graph replay per timed step; excludes input updates and correctness checks; not serving TPOT",
                    warmup_replays=10,
                    replays_per_sample=1,
                    schedule="24 steps, rotate ABC, identical new pooled scores for all paths each step",
                    retained_body_tokens=1636,
                    new_body_tokens=412,
                    union_misses_per_request=dynamic_misses,
                )
            if (
                native_copy
                and native_fused
                and not framework
                and os.environ.get("SGLANG_LIM_BREAKDOWN") == "1"
            ):
                phase_samples = {}
                for name, operation in (
                    ("native_lim", lambda: forward(selection_only=True)),
                    ("native_copy", lambda: forward(attention_only=True)),
                ):
                    operation()
                    captured = torch.npu.NPUGraph()
                    phase_graphs.append(captured)
                    with torch.npu.graph(captured):
                        phase_outputs.append(operation())
                    for _ in range(5):
                        captured.replay()
                    phase_samples[name] = []
                for cycle in range(6):
                    order = range(2) if cycle % 2 == 0 else reversed(range(2))
                    for index in order:
                        name = tuple(phase_samples)[index]
                        torch.npu.synchronize()
                        start = time.perf_counter_ns()
                        for _ in range(10):
                            phase_graphs[index].replay()
                        torch.npu.synchronize()
                        phase_samples[name].append(
                            (time.perf_counter_ns() - start) / 10000
                        )
                check(phase_outputs[1])
                report["phase_samples_us"] = phase_samples
                report["phase_median_us"] = {
                    name: statistics.median(values)
                    for name, values in phase_samples.items()
                }
                report["phase_scope"] = (
                    "Independent Graphs on the native Indexer selection; CopySFA uses a frozen plan; phase sums are not whole-graph latency"
                )
            if (
                include_indexer
                and not native_fused
                and os.environ.get("SGLANG_LIM_BREAKDOWN") == "1"
            ):
                from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
                    FusedSparseAttentionInputs,
                )

                attention_inputs = FusedSparseAttentionInputs(
                    query=query,
                    selection_k_rope=component.rope,
                    selection_kv_cache=component.kv,
                    selection_kv_block_table=component.table[:rows],
                    selection_kv_block_status=component.status[:rows],
                    selection_membership_map=component.planner.membership[:rows],
                    selection_topk_indices=component.planner.topk[:rows, None],
                    full_k_rope=paged_kv_view(full_rope, 128, 0),
                    full_kv_cache=paged_kv_view(full_kv, 128, 512),
                    full_kv_block_table=table,
                    full_kv_actual_seq=kv_lengths,
                    full_q_actual_seq=query_ends,
                    scale_value=0.0625,
                    full_kv_region=region,
                )

                def manage_fixed_selection():
                    planned = component.planner.plan(
                        ids,
                        indexer.scores,
                        query_ends,
                        lengths,
                        table,
                        request_slots,
                        generations,
                        host_tokens=pages * 128,
                    )
                    component.planner.commit(request_slots, query_ends, lengths)
                    return planned

                phase_samples = {}
                for name, operation in (
                    ("baseline_indexer", lambda: baseline(selection_only=True)),
                    ("candidate_indexer", lambda: forward(selection_only=True)),
                    ("baseline_manager_fsa", lambda: baseline(attention_only=True)),
                    ("candidate_manager_fsa", lambda: forward(attention_only=True)),
                    ("candidate_manager", manage_fixed_selection),
                    (
                        "candidate_fsa",
                        lambda: run_fused_sparse_attention(attention_inputs),
                    ),
                ):
                    operation()
                    captured = torch.npu.NPUGraph()
                    phase_graphs.append(captured)
                    with torch.npu.graph(captured):
                        phase_outputs.append(operation())
                    for _ in range(5):
                        captured.replay()
                    phase_samples[name] = []
                for cycle in range(6):
                    order = range(6) if cycle % 2 == 0 else reversed(range(6))
                    for index in order:
                        name = tuple(phase_samples)[index]
                        torch.npu.synchronize()
                        start = time.perf_counter_ns()
                        for _ in range(10):
                            phase_graphs[index].replay()
                        torch.npu.synchronize()
                        phase_samples[name].append(
                            (time.perf_counter_ns() - start) / 1000 / 10
                        )
                check(phase_outputs[2])
                check(phase_outputs[3])
                check(phase_outputs[5])
                report["phase_samples_us"] = phase_samples
                report["phase_median_us"] = {
                    name: statistics.median(values)
                    for name, values in phase_samples.items()
                }
                report["phase_scope"] = (
                    "Separately captured warm phases; sums are not the whole-graph critical path"
                )
            if len(phase_graphs) == 6 and (
                profile_directory := os.environ.get("SGLANG_LIM_PROFILE")
            ):
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    schedule=torch_npu.profiler.schedule(
                        wait=1, warmup=1, active=3, repeat=1
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        profile_directory
                    ),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                    ),
                ) as profiler:
                    for _ in range(6):
                        phase_graphs[4].replay()
                        torch.npu.synchronize()
                        profiler.step()
            path.mkdir(parents=True, exist_ok=True)
            prefix = "indexed_component" if include_indexer else "component"
            if three_way:
                prefix = "changing_three_way" if changing_topk else "three_way"
            if context != 16384 or indexer_heads != 64:
                prefix += f"_s{context}_h{indexer_heads}"
            if changing_topk and pooled_shift != 103:
                prefix += f"_shift{pooled_shift}"
            if kv_length != context:
                prefix += f"_len{kv_length}"
            (path / f"{prefix}_b{batch}_q{routes}.json").write_text(
                json.dumps(report, indent=2)
            )
    finally:
        torch.npu.synchronize()
        for captured in phase_graphs:
            captured.reset()
        if baseline_graph is not None:
            baseline_graph.reset()
        if third_graph is not None:
            third_graph.reset()
        if graph is not None:
            graph.reset()
        if baseline_state is not None:
            for planner in baseline_state.planners.values():
                planner.close(drained=True)
        backing.pool.close(drained=True)
