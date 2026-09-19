"""Decode transfer correctness on an explicitly reserved idle NPU."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires an idle A5 and the matching standalone extension/OPP",
)


class DecodeCase:
    def __init__(self, batch=1, *, mapped=True, tail=3, reserved_rows=None):
        import torch_npu

        from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion
        from sglang.srt.hardware_backend.npu.fused_sparse_attention import (
            FusedSparseAttentionState,
        )

        torch_npu.npu.set_device(int(os.environ.get("SGLANG_FSA_TEST_DEVICE_ID", "0")))
        self.device = torch.device("npu", torch.npu.current_device())
        self.batch, self.sequence, self.topk, self.capacity = batch, 4096, 2051, 4096
        self.region = RegisteredHostRegion.allocate(
            (batch * 32, 128, 512), torch.bfloat16, device_id=self.device.index
        )
        torch.manual_seed(918)
        self.source = (
            torch.randn(batch, self.sequence, 512, dtype=torch.bfloat16) * 0.25
        )
        self.region.tensor.copy_(self.source.view_as(self.region.tensor))
        self.query_cpu = torch.randn(batch, 16, 512, dtype=torch.bfloat16) * 0.25
        self.query = self.query_cpu.to(self.device)
        self.rope = torch.empty(batch * 32, 128, 0, dtype=torch.bfloat16)
        self.ids_cpu = (
            torch.cat((torch.arange(2048), torch.arange(4093, 4096)))
            .int()
            .repeat(batch, 1)
        )
        self.ids = self.ids_cpu.to(self.device)
        self.table = torch.arange(
            batch * 32, device=self.device, dtype=torch.int32
        ).view(batch, 32)
        self.lengths = torch.full(
            (batch,), self.sequence, dtype=torch.int32, device=self.device
        )
        self.qlens = torch.arange(1, batch + 1, dtype=torch.int32, device=self.device)
        self.requests = torch.arange(
            1, batch + 1, dtype=torch.int64, device=self.device
        )
        self.pool = SimpleNamespace(
            req_generation=torch.ones(batch + 1, dtype=torch.int64)
        )
        self.state = FusedSparseAttentionState(
            self.device,
            self.pool,
            external_planner=True,
            resident_capacity=self.capacity,
            mapped_host_plan=mapped,
            mutable_tail_tokens=tail,
            layer_anchors={0: 0, 1: 0},
        )
        for layer in (0, 1):
            self.state.reserve(
                layer, reserved_rows or batch, self.topk, self.rope, self.region.tensor
            )
        self.graphs = []

    def invoke(self, layer=0):
        from sglang.srt.hardware_backend.npu.fused_sparse_attention import (
            run_fused_sparse_attention,
        )

        inputs = self.state.build_inputs(
            layer_id=layer,
            q_nope=self.query,
            q_rope=self.query[..., :0],
            topk_indices=self.ids,
            full_kv=self.region.tensor,
            full_rope=self.rope,
            block_table=self.table,
            kv_lengths=self.lengths,
            query_lengths=self.qlens,
            request_slots=self.requests,
            scale=0.0625,
        )
        inputs = replace(inputs, full_kv_region=self.region)
        return run_fused_sparse_attention(inputs), inputs

    def expected(self):
        return torch.stack(
            [
                torch.softmax(
                    self.query_cpu[row].float()
                    @ self.source[row, self.ids_cpu[row].long()].float().T
                    * 0.0625,
                    dim=-1,
                )
                @ self.source[row, self.ids_cpu[row].long()].float()
                for row in range(self.batch)
            ]
        )

    def check(self, output):
        torch.npu.synchronize()
        torch.testing.assert_close(
            output.cpu().float(), self.expected(), atol=0.002, rtol=0.025
        )

    def close(self):
        torch.npu.synchronize()
        for graph in self.graphs:
            graph.reset()
        for planner in self.state.planners.values():
            planner.close(drained=True)
        self.region.close(drained=True)


@pytest.mark.parametrize("batch", [1, 4, 16, 64])
@pytest.mark.parametrize("mapped", [False, True])
def test_mapped_plan_dynamic_replay_and_mutable_tail(batch, mapped):
    case = DecodeCase(batch, mapped=mapped)
    try:
        for layer in (0, 1):
            output, inputs = case.invoke(layer)
            case.check(output)
            assert inputs.selection_plan.miss_mask.cpu().all()
        output, inputs = case.invoke()
        case.check(output)
        assert inputs.selection_plan.miss_mask.sum().item() == batch * 3
        graph = torch.npu.NPUGraph()
        case.graphs.append(graph)
        with torch.npu.graph(graph):
            output, inputs = case.invoke()
        for iteration in range(4):
            case.ids_cpu = case.ids_cpu[:, torch.randperm(case.topk)]
            case.ids.copy_(case.ids_cpu)
            case.source[:, -3:] = torch.randn_like(case.source[:, -3:])
            case.region.tensor.copy_(case.source.view_as(case.region.tensor))
            if iteration == 2:
                case.pool.req_generation.add_(1)
                case.state.refresh_generations()
            graph.replay()
            case.check(output)
            expected_misses = batch * (case.topk if iteration == 2 else 3)
            assert inputs.selection_plan.miss_mask.sum().item() == expected_misses
    finally:
        case.close()


@pytest.mark.parametrize("batch", [1, 64])
def test_mapped_plan_padding_then_reuse_and_length_rollback(batch):
    case = DecodeCase(batch, mapped=True)
    try:
        case.check(case.invoke()[0])
        graph = torch.npu.NPUGraph()
        case.graphs.append(graph)
        with torch.npu.graph(graph):
            output, inputs = case.invoke()
        case.requests.zero_()
        graph.replay()
        torch.npu.synchronize()
        assert not inputs.selection_plan.miss_mask.any().item()
        assert torch.isfinite(output).all().item()
        case.requests.copy_(torch.arange(1, batch + 1, device=case.device))
        graph.replay()
        case.check(output)
        assert inputs.selection_plan.miss_mask.all().item()
        case.lengths.fill_(3072)
        case.ids_cpu[:, -3:] = torch.arange(3069, 3072)
        case.ids.copy_(case.ids_cpu)
        graph.replay()
        case.check(output)
        assert inputs.selection_plan.miss_mask.all().item()
    finally:
        case.close()


def test_mapped_plan_reserved_rows_and_shared_layer_replay():
    case = DecodeCase(1, mapped=True, reserved_rows=64)
    try:
        for layer in (0, 1):
            case.check(case.invoke(layer)[0])
        graph = torch.npu.NPUGraph()
        case.graphs.append(graph)
        with torch.npu.graph(graph):
            first, first_inputs = case.invoke(0)
            second, second_inputs = case.invoke(1)
        for iteration in range(4):
            case.ids_cpu = case.ids_cpu[:, torch.randperm(case.topk)]
            case.ids.copy_(case.ids_cpu)
            case.source[:, -3:] = torch.randn_like(case.source[:, -3:])
            case.region.tensor.copy_(case.source.view_as(case.region.tensor))
            graph.replay()
            case.check(first)
            case.check(second)
            assert first_inputs.selection_plan.miss_mask.sum().item() == 3
            assert second_inputs.selection_plan.miss_mask.sum().item() == 3
    finally:
        case.close()
