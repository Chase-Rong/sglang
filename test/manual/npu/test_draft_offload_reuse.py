"""Complete Draft offload-chain simulation; no model weights or TP collectives.

Run on a qualified NPU with the standalone FSA extension and matching vendor.
The fixture uses production MemFabric writeback, metadata, planner, attention
and commit. Projected Q/K inputs replace neural layers; timings are not TPOT.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="Requires explicitly selected NPU and offload dependencies",
)


class DraftChain:
    def __init__(self, batch, steps, reuse):
        from sglang.srt.hardware_backend.npu.hisparse.decode_host_kv import DecodeHostKvBacking
        from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
            FusedSparseAttentionState,
        )
        from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import OffloadPageBinding

        self.batch, self.steps, self.reuse = batch, steps, reuse
        self.sequence = 8192
        self.page_size = 128
        self.pages_per_request = 66
        self.device = torch.device("npu:0")
        pages = batch * self.pages_per_request + 1
        self.backing = DecodeHostKvBacking(
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
        binding = OffloadPageBinding.__new__(OffloadPageBinding)
        binding.page_size = 128
        binding.device_to_host = torch.arange(
            pages, dtype=torch.int64, device=self.device
        )
        self.backing.page_binding = binding
        self.region = self.backing.regions[0][0]
        generator = torch.Generator().manual_seed(831)
        self.source = (
            torch.randn((pages * 128, 512), generator=generator).mul_(0.2).bfloat16()
        )
        self.region.tensor.copy_(self.source)
        self.request_pool = SimpleNamespace(
            req_generation=torch.ones(batch + 1, dtype=torch.int64)
        )
        self.state = FusedSparseAttentionState(
            self.device,
            self.request_pool,
            external_planner=True,
            resident_capacity=4096,
            mutable_tail_tokens=12,
        )
        self.table = torch.arange(
            1, pages, dtype=torch.int32, device=self.device
        ).reshape(batch, -1)
        self.positions = [
            torch.full(
                (batch,), self.sequence + step, dtype=torch.int64, device=self.device
            )
            for step in range(steps)
        ]
        self.lengths = [position.int() + 1 for position in self.positions]
        self.locations = [
            self.table[:, 0].long() * 128 + position for position in self.positions
        ]
        self.queries = [
            torch.randn(batch, 8, 512, generator=generator)
            .mul_(0.2)
            .bfloat16()
            .to(self.device)
            for _ in range(steps)
        ]
        self.values = [
            torch.randn(batch, 1, 512, generator=generator)
            .mul_(0.2)
            .bfloat16()
            .to(self.device)
            for _ in range(steps)
        ]
        self.rope = torch.empty(batch, 1, 0, dtype=torch.bfloat16, device=self.device)
        self.topk = torch.full((batch, 2051), -1, dtype=torch.int32, device=self.device)
        self.set_seed(0)
        self.forward = SimpleNamespace(
            reuse_dsa_topk_indices=False,
            draft_sparse_plan_cache=None,
            spec_info=SimpleNamespace(dsa_topk_indices=self.topk),
            req_pool_indices=torch.arange(1, batch + 1, device=self.device),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
        )
        pool = SimpleNamespace(
            start_layer=0,
            decode_host_backing=self.backing,
            get_kv_buffer=lambda layer: self.backing.layer_views(layer),
            set_kv_buffer=lambda layer,
            locations,
            key,
            rope: self.backing.write_current(0, locations, key, rope),
        )
        self.backend = SimpleNamespace(
            token_to_kv_pool=pool,
            fused_sparse_draft_reuse=reuse,
            fused_sparse_state=self.state,
            fused_sparse_mode="fused",
            fused_sparse_topk=2051,
            page_size=128,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            _query_lens_device=lambda forward: torch.ones(
                batch, dtype=torch.int32, device=self.device
            ),
        )
        self.layer = SimpleNamespace(layer_id=0, scaling=0.0625)
        self.graph = None
        self.outputs = None
        self.rewrite_tail = False

    def set_seed(self, shift):
        self.topk[:, :2048].copy_(
            torch.arange(2048, device=self.device, dtype=torch.int32) + shift
        )
        self.topk[:, 2048:2050].copy_(
            (self.positions[0] - 2).int()[:, None].expand(-1, 2)
        )
        self.topk[:, 2050] = -1

    def run(self):
        from sglang.srt.hardware_backend.npu.hisparse.fsa_backend import forward_fsa
        from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState

        self.forward.spec_info.dsa_topk_indices = self.topk
        outputs = []
        with IndexTopKShareState.mtp_iteration(self.forward, keep_carry_seed=True):
            for step in range(self.steps):
                if self.rewrite_tail:
                    tail_locations = (
                        self.table[:, 0].long() * 128 + self.topk[:, 2048].long()
                    )
                    self.backing.write_current(
                        0, tail_locations, self.values[step], self.rope
                    )
                self.forward.positions = self.positions[step]
                self.forward.out_cache_loc = self.locations[step]
                self.backend.forward_metadata = SimpleNamespace(
                    cache_seqlens_int32=self.lengths[step],
                    real_page_table=self.table,
                    host_page_table=self.table,
                )
                outputs.append(
                    forward_fsa(
                        self.backend,
                        self.queries[step],
                        self.values[step],
                        self.values[step],
                        self.layer,
                        self.forward,
                        True,
                        None,
                        self.rope,
                        self.forward.spec_info.dsa_topk_indices,
                    )
                )
        return outputs

    def capture(self):
        self.run()
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            self.outputs = self.run()

    def close(self):
        torch.npu.synchronize()
        if self.graph is not None:
            self.graph.reset()
        for planner in self.state.planners.values():
            planner.close(drained=True)
        self.backing.pool.close(drained=True)


def verify_pair(before, after):
    for original, updated in zip(before.outputs, after.outputs):
        torch.testing.assert_close(
            original.cpu(), updated.cpu(), atol=0.002, rtol=0.025
        )
    torch.testing.assert_close(
        before.region.tensor, after.region.tensor, rtol=0, atol=0
    )
    slots = after.state.planners[0].slots.cpu()
    selected = (
        after.state.layers[0]
        .cache.selection_kv_cache.cpu()
        .reshape(after.batch, 4096, 512)
    )
    for row in range(after.batch):
        if int(after.forward.req_pool_indices[row].item()) == 0:
            for output in after.outputs:
                torch.testing.assert_close(
                    output[row], torch.zeros_like(output[row]), rtol=0, atol=0
                )
            continue
        ids = after.topk[row].cpu().long()
        valid = ids >= 0
        physical = int(after.table[row, 0].item()) * 128 + ids[valid]
        torch.testing.assert_close(
            selected[row, slots[row, valid]],
            after.region.tensor[physical],
            rtol=0,
            atol=0,
        )
        keys = after.region.tensor[physical].float()
        for step, output in enumerate(after.outputs):
            if after.rewrite_tail:
                break
            query = after.queries[step][row].cpu().float()
            expected = torch.softmax(query @ keys.T * 0.0625, dim=-1) @ keys
            torch.testing.assert_close(
                output[row].cpu().float(), expected, atol=0.002, rtol=0.025
            )


@pytest.mark.parametrize("batch,steps", [(1, 3), (1, 4), (10, 3), (10, 4)])
def test_draft_chain_eager_graph_seed_tail_and_generation(batch, steps):
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.hisparse.external_sparse_planner import (
        NativeSparsePlanner,
    )

    torch.npu.set_device(0)
    before, after = DraftChain(batch, steps, False), DraftChain(batch, steps, True)
    counts = []
    original = NativeSparsePlanner.enqueue
    try:
        for chain in (before, after):
            calls = []

            def counted(planner, *args, **kwargs):
                calls.append(1)
                return original(planner, *args, **kwargs)

            with patch.object(NativeSparsePlanner, "enqueue", counted):
                chain.outputs = chain.run()
            counts.append(len(calls))
        assert counts == [steps, 1]
        verify_pair(before, after)
        for chain in (before, after):
            chain.rewrite_tail = True
            chain.capture()
        for iteration in range(4):
            for chain in (before, after):
                chain.set_seed(iteration * 413)
                chain.request_pool.req_generation.add_(1)
                chain.state.refresh_generations()
                chain.region.tensor.copy_(chain.source + iteration * 0.03125)
                if iteration == 2:
                    chain.table.copy_(chain.table.flip(0))
                    chain.forward.req_pool_indices.copy_(
                        chain.forward.req_pool_indices.flip(0)
                    )
                    for step in range(steps):
                        chain.positions[step].sub_(128)
                        chain.lengths[step].sub_(128)
                        chain.locations[step].copy_(
                            chain.table[:, 0].long() * 128 + chain.positions[step]
                        )
                    chain.topk[:, 2048:2050] = chain.sequence - 130
                if iteration == 3 and batch > 1:
                    chain.forward.req_pool_indices[-1] = 0
                    for step in range(steps):
                        chain.lengths[step][-1] = 0
                        chain.locations[step][-1] = 0
                chain.graph.replay()
            torch.npu.synchronize()
            for chain in (before, after):
                active = chain.forward.req_pool_indices.cpu() > 0
                epochs = chain.state.planners[0].row_state[:batch, 2]
                expected = 1 if chain.reuse else steps
                assert (epochs[active] == expected).all()
            verify_pair(before, after)
    finally:
        after.close()
        before.close()


def benchmark(output, steps, graph_mode):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(
        (2 << 30) / torch.npu.get_device_properties(0).total_memory, 0
    )
    torch.set_num_threads(1)
    results = []
    for batch in (1, 10):
        before, after = DraftChain(batch, steps, False), DraftChain(batch, steps, True)
        try:
            for chain in (before, after):
                if graph_mode:
                    chain.capture()
                for iteration in range(10):
                    if graph_mode:
                        chain.graph.replay()
                    else:
                        chain.outputs = chain.run()
            torch.npu.synchronize()
            verify_pair(before, after)
            samples = []
            for cycle in range(6):
                for name, chain in (
                    ("baseline", before),
                    ("candidate", after),
                    ("candidate", after),
                    ("baseline", before),
                ):
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    for iteration in range(10):
                        if graph_mode:
                            chain.graph.replay()
                        else:
                            chain.outputs = chain.run()
                    torch.npu.synchronize()
                    samples.append(
                        dict(
                            cycle=cycle,
                            variant=name,
                            us=(time.perf_counter_ns() - start) / 1000 / 10,
                        )
                    )
            summary = {
                name: statistics.median(
                    row["us"] for row in samples if row["variant"] == name
                )
                for name in ("baseline", "candidate")
            }
            results.append(
                dict(
                    batch=batch,
                    draft_forwards=steps,
                    graph=graph_mode,
                    summary_us=summary,
                    samples=samples,
                )
            )
            print(json.dumps(results[-1]), flush=True)
        finally:
            after.close()
            before.close()
    output.write_text(
        json.dumps(
            dict(
                scope="Draft offload chain with supplied Q/K, no weights/MoE/TP; shared-NPU exploratory timing",
                results=results,
                peak_reserved_bytes=torch.npu.max_memory_reserved(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(3, 4), default=4)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    benchmark(args.output, args.steps, not args.eager)
