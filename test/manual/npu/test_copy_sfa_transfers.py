"""Isolated CopySFA Graph latency with fixed Host backing and hit/miss plans."""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_LIM_COMPONENT_TEST") != "1",
    reason="Requires an authorized NPU component environment",
)


@pytest.mark.parametrize("batch,routes", [(1, 1), (10, 6)])
@pytest.mark.parametrize(
    "scenario",
    ["hit100", "hit80", "hit80_scattered", "miss512", "miss513", "hit0", "cold"],
)
def test_copy_sfa_transfer_cases(batch, routes, scenario):
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.hisparse.decode_host_kv import DecodeHostKvBacking
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_attention import (
        NativeCopySparseAttention,
    )
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_contract import LimCopySfaSpec
    from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_native_plan import (
        NativeUnionPlanner,
    )

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    pages = batch * 128 + 1
    backing = DecodeHostKvBacking(
        layers=1,
        pages=pages,
        page_size=128,
        latent_dim=512,
        rope_dim=0,
        dtype=torch.bfloat16,
        device_id=0,
        host_budget=1 << 30,
        integrated=True,
    )
    graph = None
    try:
        region = backing.regions[0][0]
        generator = torch.Generator().manual_seed(441)
        source = torch.randn(pages * 128, 512, generator=generator).mul_(0.1).bfloat16()
        region.tensor.copy_(source)
        rows = batch * routes
        query_cpu = torch.randn(rows, 8, 512, generator=generator).mul_(0.1).bfloat16()
        query = query_cpu.to("npu")
        spec = LimCopySfaSpec(16384, routes, 4096 if routes == 1 else 12416)
        planner = NativeUnionPlanner(spec, batch + 1, batch, "npu:0")
        operator = NativeCopySparseAttention("npu:0", row_capacity=rows)
        resident = torch.empty(
            (batch + 1, spec.resident_tokens, 512), dtype=torch.bfloat16, device="npu"
        )
        ids = torch.full((rows, 2051), -1, dtype=torch.int32, device="npu")
        ids[:, :2048] = torch.arange(2048, device="npu")
        scores = torch.ones(rows, 4096, device="npu")
        ends = torch.arange(routes, rows + 1, routes, dtype=torch.int32, device="npu")
        lengths = torch.tensor(
            [
                16384 - routes + route + 1
                for _ in range(batch)
                for route in range(routes)
            ],
            dtype=torch.int32,
            device="npu",
        )
        kv_lengths = torch.full((batch,), 16384, dtype=torch.int32, device="npu")
        table = torch.arange(1, pages, dtype=torch.int32, device="npu").view(batch, 128)
        requests = torch.arange(1, batch + 1, dtype=torch.int64, device="npu")
        generations = torch.ones(batch + 1, dtype=torch.int64, device="npu")

        def plan():
            planner.plan(
                ids,
                scores,
                ends,
                lengths,
                table,
                requests,
                generations,
                host_tokens=pages * 128,
            )

        def invoke():
            return operator.forward(
                query,
                region.tensor,
                region,
                resident,
                planner,
                ends,
                lengths,
                kv_lengths,
                table,
                requests,
                scale=0.0625,
            )

        plan()
        invoke()
        torch.npu.synchronize()
        shift = {
            "hit100": 0,
            "hit80": 410,
            "hit80_scattered": 410,
            "miss512": 512,
            "miss513": 513,
            "hit0": 2048,
            "cold": 2048,
        }[scenario]
        selected_cpu = (torch.arange(2048) + shift).repeat(rows, 1)
        if scenario == "hit80_scattered":
            selected = torch.cat(
                (
                    torch.randperm(2048, generator=generator)[:1638],
                    torch.arange(2048, 2458),
                )
            )
            for row in range(rows):
                selected_cpu[row] = selected[torch.randperm(2048, generator=generator)]
        ids[:, :2048] = selected_cpu.to(device="npu", dtype=torch.int32)
        if scenario == "cold":
            generations.add_(1)
        plan()
        torch.npu.synchronize()
        signed = planner.membership[:, 14333:16384].cpu()
        expected_misses = 2048 if scenario in ("cold", "hit0") else shift
        assert torch.all((signed < 0).sum(1) == expected_misses)

        def check(output):
            actual = output.cpu().float()
            for row in range(rows):
                request = row // routes
                keys = source[request * 16384 + 128 + selected_cpu[row]].float()
                expected = (
                    torch.softmax(query_cpu[row].float() @ keys.T * 0.0625, -1) @ keys
                )
                torch.testing.assert_close(
                    actual[row], expected, atol=0.002, rtol=0.025
                )

        check(invoke())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = invoke()
        for _ in range(5):
            graph.replay()
        samples = []
        for _ in range(8):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            for _ in range(10):
                graph.replay()
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 10000)
        check(output)
        if directory := os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK"):
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"copy_b{batch}_q{routes}_{scenario}.json").write_text(
                json.dumps(
                    dict(
                        batch=batch,
                        routes=routes,
                        scenario=scenario,
                        median_us=statistics.median(samples),
                        samples_us=samples,
                        misses_per_row=expected_misses,
                        scope="CopySFA only; frozen plan forces the stated transfers on every replay; cold repeats first-fill; excludes LIM and serving TPOT",
                    ),
                    indent=2,
                )
            )
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        backing.pool.close(drained=True)
