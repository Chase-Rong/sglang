"""Host mapping plus graph consumers; component timing, not full draft TPOT."""

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


def binding_for_test():
    import torch_npu  # noqa: F401

    from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import OffloadPageBinding

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    binding = OffloadPageBinding.__new__(OffloadPageBinding)
    binding.device_to_host = torch.arange(20000, dtype=torch.int64, device="npu")
    return binding


@pytest.mark.parametrize("batch", [1, 8, 16])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_host_table_graph_remap_padding_and_strides(batch, dtype):
    binding = binding_for_test()
    storage = torch.full((batch + 1, 4096), 999, dtype=dtype, device="npu")
    output = storage[:, ::2]
    source_cpu = torch.arange(batch * 2050, dtype=dtype).view(batch, 2050)[:, ::2]
    source_cpu[:, :4] = torch.tensor([-1, 0, 20000, 2**31 - 1], dtype=dtype)
    source = source_cpu.to("npu")
    binding.fill_host_table(source, output)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            binding.fill_host_table(source, output)
            consumed = output + 1
        for iteration in range(4):
            source.copy_(source_cpu.flip(0) if iteration % 2 else source_cpu)
            binding.device_to_host.copy_(
                torch.arange(20000, dtype=torch.int64, device="npu").roll(iteration)
            )
            graph.replay()
            torch.npu.synchronize()
            expected = torch.zeros_like(output)
            expected[:batch, :1025] = binding.host_table(source)
            torch.testing.assert_close(output.cpu(), expected.cpu(), atol=0, rtol=0)
            torch.testing.assert_close(
                consumed.cpu(), expected.cpu() + 1, atol=0, rtol=0
            )
            assert torch.all(storage[:, 1::2].cpu() == 999)
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_draft_host_table_prepare_and_consumer_benchmark(batch):
    destination = os.environ.get("SGLANG_LIM_COMPONENT_BENCHMARK")
    if not destination:
        pytest.skip("Benchmark output directory required")
    from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import DraftHostTableScope

    binding = binding_for_test()
    source = torch.arange(1, batch * 1024 + 1, dtype=torch.int32, device="npu").view(
        batch, 1024
    )
    outputs = {
        name: [
            torch.empty(batch, 2048, dtype=torch.int32, device="npu") for _ in range(4)
        ]
        for name in ("original", "fused", "shared")
    }
    outputs["shared"] = [outputs["shared"][0]] * 4

    def prepare(name):
        if name == "original":
            for output in outputs[name]:
                output.zero_()
                output[:, : source.shape[1]].copy_(binding.host_table(source))
        elif name == "fused":
            for output in outputs[name]:
                binding.fill_host_table(source, output)
        else:
            scope = DraftHostTableScope(3)
            for output in outputs[name]:
                scope.host_table(binding, source, output)

    graphs = {}
    consumed = {}
    samples = {name: [] for name in outputs}
    try:
        for name in outputs:
            prepare(name)
            graph = torch.npu.NPUGraph()
            graphs[name] = graph
            with torch.npu.graph(graph):
                consumed[name] = [output + 1 for output in outputs[name]]
            for _ in range(5):
                prepare(name)
                graph.replay()
        for cycle in range(8):
            names = tuple(outputs)
            order = names if cycle % 2 == 0 else names[::-1]
            for name in order:
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                for _ in range(20):
                    prepare(name)
                    graphs[name].replay()
                torch.npu.synchronize()
                samples[name].append((time.perf_counter_ns() - start) / 20000)
        expected = torch.ones_like(outputs["original"][0])
        expected[:, :1024] += binding.host_table(source)
        for values in consumed.values():
            for value in values:
                torch.testing.assert_close(value.cpu(), expected.cpu(), atol=0, rtol=0)
        Path(destination).mkdir(parents=True, exist_ok=True)
        (Path(destination) / f"host_table_b{batch}.json").write_text(
            json.dumps(
                dict(
                    scope="Four draft Host table preparations plus four graph consumers; not complete draft/model step",
                    batch=batch,
                    context=131072,
                    capture_context=262144,
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
