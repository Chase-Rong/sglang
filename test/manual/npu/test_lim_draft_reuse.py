"""Real Host KV MTP management/attention chain, excluding model GEMM and FFN."""

import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an authorized NPU component environment",
)


@pytest.mark.parametrize("batch", [1, 8, 16])
@pytest.mark.parametrize("accepted", [1, 4, 6])
def test_seeded_draft_chain(batch, accepted):
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.hisparse.decode_host_kv import DecodeHostKvBacking
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_backend import LimCopySfaState
    from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    context = 131072
    pages_per_request = context // 128
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
    generator = torch.Generator().manual_seed(61010 + batch)
    region.tensor.copy_(
        torch.randn(
            region.tensor.shape, dtype=torch.bfloat16, generator=generator
        ).mul_(0.1)
    )
    host, _ = backing.layer_views(0)
    table = torch.arange(1, pages, dtype=torch.int32, device="npu").view(
        batch, pages_per_request
    )
    requests = torch.arange(1, batch + 1, dtype=torch.int64, device="npu")
    accept_tensor = torch.full((batch,), accepted, dtype=torch.int64, device="npu")
    pool = SimpleNamespace(
        req_generation=torch.ones(batch + 1, dtype=torch.int64), req_generation_epoch=0
    )
    sparse_pool = SimpleNamespace(start_layer=0, decode_host_backing=backing)
    kv_pool = SimpleNamespace(
        full_kv_pool=sparse_pool, _transfer_full_attention_id=lambda _: 0
    )
    body = []
    for _ in range(batch * 6):
        selected = torch.randperm((context - 4096) // 4, generator=generator)[:512]
        body.append((selected[:, None] * 4 + torch.arange(4)).reshape(-1))
    body = torch.stack(body).to(device="npu", dtype=torch.int32)
    extend_query = (
        torch.randn(batch * 6, 8, 512, generator=generator)
        .mul_(0.1)
        .bfloat16()
        .to("npu")
    )
    draft_query = (
        torch.randn(batch, 8, 512, generator=generator).mul_(0.1).bfloat16().to("npu")
    )
    layer = SimpleNamespace(layer_id=0, scaling=0.0625)
    runtimes = {}
    graphs = {}
    captured = {}
    for name in ("full_manager", "reused_plan"):
        state = LimCopySfaState(
            "npu:0",
            pool,
            source_tokens=context,
            routes=6,
            capacity=12416,
            batch_capacity=batch,
            layer_anchors={},
            mutable_tail=12,
        )
        state.layer(0)
        backend = SimpleNamespace(
            token_to_kv_pool=kv_pool,
            forward_metadata=SimpleNamespace(host_page_table=table),
            _query_lens_device=lambda forward_batch: forward_batch.counts,
        )
        runtimes[name] = (
            state,
            backend,
            torch.full((batch,), context - 2048, dtype=torch.int64, device="npu"),
        )

    def chain(name):
        state, backend, base = runtimes[name]
        lengths = (base[:, None] + torch.arange(1, 7, device="npu")).reshape(-1)
        tail = lengths[:, None] // 4 * 4 + torch.arange(3, device="npu")
        tail = torch.where(tail < lengths[:, None], tail, -1).to(torch.int32)
        ids = torch.cat((body, tail), dim=1)
        extend_batch = SimpleNamespace(
            positions=lengths - 1,
            req_pool_indices=requests,
            counts=torch.full((batch,), 6, dtype=torch.int32, device="npu"),
        )
        backend.forward_metadata.cache_seqlens_int32 = (base + 6).to(torch.int32)
        output = state.forward(
            backend, extend_query, layer, extend_batch, ids, host, region
        )
        records = [(output, ids, lengths, extend_query)]
        seed_rows = torch.arange(batch, device="npu") * 6 + accept_tensor - 1
        seed = ids[seed_rows]
        draft_batch = SimpleNamespace(
            req_pool_indices=requests,
            counts=torch.ones(batch, dtype=torch.int32, device="npu"),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            spec_info=SimpleNamespace(dsa_topk_indices=seed),
            draft_sparse_plan_cache=None,
            reuse_dsa_topk_indices=False,
        )
        with IndexTopKShareState.mtp_iteration(draft_batch, keep_carry_seed=True):
            if name == "full_manager":
                draft_batch.draft_sparse_plan_cache = None
            for step in range(4):
                draft_lengths = base + accept_tensor + step + 1
                draft_batch.positions = draft_lengths - 1
                backend.forward_metadata.cache_seqlens_int32 = draft_lengths.to(
                    torch.int32
                )
                output = state.forward(
                    backend, draft_query, layer, draft_batch, seed, host, region
                )
                records.append((output, seed, draft_lengths, draft_query))
        base.add_(accept_tensor)
        return records

    def check(records):
        torch.npu.synchronize()
        mapping = table.cpu().long()
        for output, ids, lengths, query in records:
            actual, selected, limits, queries = (
                output.cpu().float(),
                ids.cpu(),
                lengths.cpu(),
                query.cpu().float(),
            )
            routes = selected.shape[0] // batch
            for row in range(selected.shape[0]):
                tokens = selected[row]
                tokens = tokens[(tokens >= 0) & (tokens < limits[row])].long()
                addresses = mapping[row // routes, tokens // 128] * 128 + tokens % 128
                keys = region.tensor[addresses].float()
                expected = torch.softmax(queries[row] @ keys.T * 0.0625, dim=-1) @ keys
                torch.testing.assert_close(
                    actual[row], expected, atol=0.002, rtol=0.025
                )

    samples = {name: [] for name in runtimes}
    try:
        for name in runtimes:
            check(chain(name))
            graph = torch.npu.NPUGraph()
            graphs[name] = graph
            with torch.npu.graph(graph):
                captured[name] = chain(name)
            graph.replay()
            check(captured[name])
        for phase in ("remap", "generation", "acceptance", "restore", "rewrite_tail"):
            if phase in ("remap", "restore"):
                table.copy_(table.flip(1))
                requests.copy_(requests.flip(0))
            if phase == "generation":
                pool.req_generation.add_(1)
                for state, _, _ in runtimes.values():
                    state.refresh_generations()
            if phase == "acceptance":
                accept_tensor.fill_(1 if accepted == 6 else 6)
            if phase == "restore":
                accept_tensor.fill_(accepted)
            if phase == "rewrite_tail":
                bases = runtimes["full_manager"][2].cpu().tolist()
                mapping = table.cpu().long()
                for request, base in enumerate(bases):
                    tokens = torch.arange(base - 12, base + 16)
                    addresses = mapping[request, tokens // 128] * 128 + tokens % 128
                    region.tensor[addresses] = region.tensor[addresses] + 0.03125
            for name in runtimes:
                graphs[name].replay()
                check(captured[name])
            for old, new in zip(captured["full_manager"], captured["reused_plan"]):
                torch.testing.assert_close(
                    old[0].cpu(), new[0].cpu(), atol=0.002, rtol=0.025
                )
        if directory := os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK"):
            for _ in range(3):
                for graph in graphs.values():
                    graph.replay()
            for _ in range(6):
                for name in (
                    "full_manager",
                    "reused_plan",
                    "reused_plan",
                    "full_manager",
                ):
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(10):
                        graphs[name].replay()
                    torch.npu.synchronize()
                    samples[name].append((time.perf_counter_ns() - start) / 10000)
            for records in captured.values():
                check(records)
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / f"draft_b{batch}_accept{accepted}.json").write_text(
                json.dumps(
                    dict(
                        scope="Six-query extend management/attention plus four one-query draft management/attention steps; excludes Indexer, projections and FFN; not model step or TPOT",
                        batch=batch,
                        accepted=accepted,
                        context=context,
                        samples_us=samples,
                        median_us={
                            name: statistics.median(values)
                            for name, values in samples.items()
                        },
                    ),
                    indent=2,
                )
            )
    finally:
        torch.npu.synchronize()
        for graph in graphs.values():
            graph.reset()
        backing.pool.close(drained=True)
