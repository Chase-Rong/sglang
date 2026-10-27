"""Opt-in correctness contract for the original external-plan A5 FSA.

Run this file in a fresh process for each baseline/candidate OPP, with the
matching vendor already on ASCEND_CUSTOM_OPP_PATH, SGLANG_NPU_OFFLOAD_TEST=1,
SGLANG_FSA_TEST_DEVICE_ID and SGLANG_NPU_FSA_EXTENSION explicitly configured.
No model, LIM kernel, MemFabric allocation or private result script is used.

BF16 absorbed512/NoPE, page128, K2051 and scale1/sqrt(256) are intentional.
The CPU float32 softmax oracle reads controlled Host/matching resident inputs;
hit payloads differ from Host to expose an erroneous hit-to-Host fallback.
Every miss writeback and every untouched resident byte are checked exactly.
Generation invalidation is supplied by the test's external plan producer;
the attention ABI itself has no request/generation argument.

Graph shapes/addresses stay fixed while cumulative query lengths, active
requests, Host pages, query data and signed plans change. Only active output
rows are specified; padded resident rows must remain untouched. Boundary
workloads use the reported Cube count to cover per-core query transitions.
Tensor budgets reserve at most512 MiB each for CPU data and the NPU allocator,
including conservative scratch allowances. Runtime/library RSS is excluded.
These are component correctness checks, not performance or model/PD evidence.
"""

from contextlib import contextmanager

import pytest
import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.npu_offload_device import offload_test_device
from sglang.test.npu_test_paths import run_pytest

register_npu_ci(
    est_time=240,
    suite="base-a-test-npu",
    disabled="Opt-in original FSA OPP/extension and an explicitly idle NPU required",
)

PAGE = 128
DIM = 512
WIDTH = 2051
CAPACITY = 4096
CONTEXT = 4096
CONTROL = 16384
MEMBERSHIP = 16400
SCALE = 0.0625
HALF_GIB = 512 << 20


@pytest.fixture
def device():
    selected = offload_test_device(extensions=("fsa",))
    try:
        yield selected
    finally:
        torch.npu.synchronize()
        torch.npu.empty_cache()


def _query_counts(total):
    return [min(6, total - begin) for begin in range(0, total, 6)]


def _attention_oracle(query, host, topk, table, counts, lengths, plans, resident):
    """CPU math and causal/page decoding independent of production helpers."""
    output = torch.zeros_like(query)
    row = 0
    for request, count in enumerate(counts):
        for route in range(count):
            limit = lengths[request] - count + route + 1
            logical = topk[row].long()
            valid = (logical >= 0) & (logical < limit)
            columns = valid.nonzero().flatten()
            if columns.numel():
                tokens = logical[columns]
                physical = table[request, tokens // PAGE].long() * PAGE + tokens % PAGE
                keys = host[physical].float()
                encoded = plans[row, columns].long()
                assert torch.all(encoded != 0)
                hits = encoded > 0
                if hits.any():
                    keys[hits] = resident[row, encoded[hits] - 1].float()
                scores = query[row].float() @ keys.T * SCALE
                output[row] = (scores.softmax(dim=-1) @ keys).bfloat16()
            row += 1
    return output[:row]


@contextmanager
def _attention_case(device, rows, batch, heads=16, direct=True):
    from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion

    source_bytes = (batch * (CONTEXT // PAGE) + 1) * PAGE * DIM * 2
    resident_bytes = rows * CAPACITY * DIM * 2
    query_bytes = rows * heads * DIM * 2
    cpu_bound = source_bytes + resident_bytes + 4 * query_bytes + (48 << 20)
    device_bound = resident_bytes + 6 * query_bytes + rows * MEMBERSHIP * 2
    device_bound += 128 << 20
    if max(cpu_bound, device_bound) > HALF_GIB:
        pytest.skip("geometry exceeds the explicit 512 MiB CPU/NPU tensor budgets")
    torch.npu.empty_cache()
    region = RegisteredHostRegion.allocate(
        (batch * (CONTEXT // PAGE) + 1, PAGE, DIM),
        torch.bfloat16,
        device_id=device.index,
    )
    case = None
    try:
        case = _AttentionCase(device, rows, batch, heads, direct, region)
        yield case
    finally:
        torch.npu.synchronize()
        if case is not None:
            for graph in case.graphs:
                graph.reset()
            if case.membership_region is not None:
                case.membership_region.close(drained=True)
        region.close(drained=True)


class _AttentionCase:
    def __init__(self, device, rows, batch, heads, direct, region):
        self.device = device
        self.rows = rows
        self.batch = batch
        self.heads = heads
        self.direct = direct
        self.region = region
        self.graphs = []
        self.membership_region = None
        self.host = region.tensor.view(-1, DIM)
        self.resident_cpu = torch.empty(rows, CAPACITY, DIM, dtype=torch.bfloat16)
        self.resident = torch.empty_like(self.resident_cpu, device=device)
        self.query_cpu = torch.empty(rows, heads, DIM, dtype=torch.bfloat16)
        self.query = torch.empty_like(self.query_cpu, device=device)
        self.topk_cpu = torch.empty(rows, WIDTH, dtype=torch.int32)
        self.topk = torch.empty(rows, 1, WIDTH, dtype=torch.int32, device=device)
        self.plans = torch.empty(rows, WIDTH, dtype=torch.int16)
        self.membership_cpu = torch.empty(rows, MEMBERSHIP, dtype=torch.int16)
        self.membership = torch.empty_like(self.membership_cpu, device=device)
        self.table_cpu = torch.empty(batch, CONTEXT // PAGE, dtype=torch.int32)
        self.table = torch.empty_like(self.table_cpu, device=device)
        self.ends = torch.empty(batch, dtype=torch.int32, device=device)
        self.kv_lengths = torch.empty_like(self.ends)
        self.selection_table = torch.arange(
            rows * (CAPACITY // PAGE), dtype=torch.int32, device=device
        ).view(rows, CAPACITY // PAGE)
        self.status = torch.full((rows, 1, 2056), -1, dtype=torch.int32, device=device)
        self.full_rope = torch.empty(
            region.tensor.shape[0], PAGE, 0, dtype=torch.bfloat16
        )
        self.selection_rope = torch.empty(
            rows * (CAPACITY // PAGE),
            PAGE,
            0,
            dtype=torch.bfloat16,
            device=device,
        )

    def prepare(
        self,
        counts,
        *,
        hit=0,
        order="random",
        length=3075,
        generation=1,
        holes=False,
        empty_last=False,
        preserve_resident=False,
    ):
        assert 0 < sum(counts) <= self.rows and len(counts) <= self.batch
        assert all(0 <= count <= 6 for count in counts)
        assert max(counts) <= length <= CONTEXT
        self.counts = list(counts) + [0] * (self.batch - len(counts))
        self.lengths = [length] * self.batch
        generator = torch.Generator().manual_seed(711 + generation)
        for begin in range(0, self.host.shape[0], PAGE):
            payload = torch.randn(PAGE, DIM, generator=generator) * 0.375
            self.host[begin : begin + PAGE].copy_(payload.bfloat16())
        self.query_cpu.copy_(
            (
                torch.randn(self.rows, self.heads, DIM, generator=generator) * 0.5
            ).bfloat16()
        )
        if not preserve_resident:
            self.resident_cpu.fill_(-7)
        self.topk_cpu.fill_(-1)
        self.plans.zero_()
        self.membership_cpu.fill_(-1)
        for request in range(self.batch):
            pages = torch.randperm(CONTEXT // PAGE, generator=generator)
            self.table_cpu[request] = pages + request * (CONTEXT // PAGE) + 1
        row = 0
        for request, count in enumerate(self.counts):
            for route in range(count):
                limit = length - count + route + 1
                pooled = limit // 4 * 4
                logical = self.topk_cpu[row]
                selected = min(2048, pooled)
                logical[:selected] = torch.arange(selected, dtype=torch.int32)
                tail = limit % 4
                logical[2048 : 2048 + tail] = torch.arange(
                    pooled, limit, dtype=torch.int32
                )
                if holes:
                    logical.copy_(logical[torch.randperm(WIDTH, generator=generator)])
                    last = (logical == limit - 1).nonzero().flatten()
                    if last.numel():
                        previous = int(logical[-1])
                        logical[-1] = limit - 1
                        logical[int(last[0])] = previous
                    invalid = (logical < 0).nonzero().flatten()
                    if invalid.numel():
                        logical[int(invalid[0])] = limit
                if empty_last and row == sum(self.counts) - 1:
                    logical.fill_(-1)
                valid = (logical >= 0) & (logical < limit)
                columns = valid.nonzero().flatten()
                if order == "contiguous":
                    slots = torch.arange(WIDTH)
                elif order == "reverse":
                    slots = CAPACITY - 1 - torch.arange(WIDTH)
                else:
                    assert order == "random"
                    slots = torch.cat(
                        (
                            torch.tensor([0, CAPACITY - 1]),
                            torch.randperm(CAPACITY - 2, generator=generator)[
                                : WIDTH - 2
                            ]
                            + 1,
                        )
                    )
                encoded = -(slots + 1).to(torch.int16)
                hits = columns[torch.randperm(columns.numel(), generator=generator)]
                hits = hits[: columns.numel() * hit // 100]
                encoded[hits] *= -1
                encoded[~valid] = 0
                self.plans[row] = encoded
                tokens = logical[hits].long()
                physical = self.table_cpu[request, tokens // PAGE].long() * PAGE
                physical += tokens % PAGE
                self.resident_cpu[row, slots[hits]] = (
                    self.host[physical].float() + 0.125 * generation
                ).bfloat16()
                row += 1
        self.query.copy_(self.query_cpu)
        self.resident.copy_(self.resident_cpu)
        self.topk.copy_(self.topk_cpu[:, None])
        self.table.copy_(self.table_cpu)
        self.ends.copy_(torch.tensor(self.counts, dtype=torch.int32).cumsum(0))
        self.kv_lengths.copy_(torch.tensor(self.lengths, dtype=torch.int32))
        self.status.fill_(-1)
        self.publish_plan()

    def publish_plan(self):
        self.membership_cpu[:, CONTROL - WIDTH : CONTROL] = self.plans
        control = self.membership_cpu[:, CONTROL : CONTROL + 8]
        control[:, 0] = 0x5A4D
        control[:, 1] = 0x5A45
        control[:, 2] = WIDTH
        control[:, 3] = CONTROL - WIDTH
        control[:, 4] = torch.arange(self.rows, dtype=torch.int16)
        control[:, 5] = 0x5A44 if self.direct else 0
        control[:, 6] = CAPACITY
        control[:, 7] = 0
        self.membership.copy_(self.membership_cpu)

    def invoke(self):
        from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
            fused_sparse_attention_overlap,
        )

        return fused_sparse_attention_overlap(
            query=self.query,
            selection_k_rope=self.selection_rope,
            selection_kv_cache=self.resident.view(-1, PAGE, DIM),
            selection_kv_block_table=self.selection_table,
            selection_kv_block_status=self.status,
            selection_membership_map=self.membership,
            selection_topk_indices=self.topk,
            full_k_rope=self.full_rope,
            full_kv_cache=self.region.tensor,
            full_kv_block_table=self.table,
            full_kv_actual_seq=self.kv_lengths,
            full_q_actual_seq=self.ends,
            full_kv_region=self.region,
            membership_region=self.membership_region,
            scale_value=SCALE,
        )

    def use_host_plan(self):
        from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion

        assert self.membership_region is None
        self.membership_region = RegisteredHostRegion.allocate(
            (self.rows, MEMBERSHIP), torch.int16, device_id=self.device.index
        )
        self.membership = self.membership_region.tensor

    def capture(self):
        graph = torch.npu.NPUGraph()
        self.graphs.append(graph)
        with torch.npu.graph(graph):
            output = self.invoke()
        torch.npu.synchronize()
        return graph, output

    def check(self, output):
        torch.npu.synchronize()
        expected = _attention_oracle(
            self.query_cpu,
            self.host,
            self.topk_cpu,
            self.table_cpu,
            self.counts,
            self.lengths,
            self.plans,
            self.resident_cpu,
        )
        assert output.shape == self.query.shape
        assert output.dtype == torch.bfloat16
        actual = output[: expected.shape[0]].cpu()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.025)
        row = 0
        for request, count in enumerate(self.counts):
            for route in range(count):
                misses = self.plans[row] < 0
                tokens = self.topk_cpu[row, misses].long()
                physical = self.table_cpu[request, tokens // PAGE].long() * PAGE
                physical += tokens % PAGE
                slots = -self.plans[row, misses].long() - 1
                self.resident_cpu[row, slots] = self.host[physical]
                row += 1
        for row in range(self.rows):
            actual_payload = self.resident[row].cpu().view(torch.int16)
            assert torch.equal(
                actual_payload, self.resident_cpu[row].view(torch.int16)
            ), f"selection payload/guard mismatch in row {row}"

    def warm(self):
        self.plans.abs_()
        self.publish_plan()

    def addresses(self, output):
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.query,
                self.resident,
                self.membership,
                self.topk,
                self.table,
                self.ends,
                self.kv_lengths,
                self.selection_table,
                self.status,
                output,
            )
        ) + (self.region.host_ptr, self.region.device_ptr)


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize(
    "batch,queries",
    [(1, queries) for queries in range(1, 7)]
    + [(2, 6), (4, 6), (8, 6), (10, 2), (10, 6), (16, 1), (32, 1)],
)
def test_attention_precision_and_miss_writeback(device, batch, queries, graph_mode):
    with _attention_case(device, batch * queries, batch) as case:
        case.prepare([queries] * batch)
        case.check(case.invoke())
        graph, output = case.capture() if graph_mode else (None, None)
        addresses = case.addresses(output) if graph_mode else None
        for step, hit in enumerate((0, 80, 99, 100)):
            for order in ("contiguous", "reverse", "random"):
                case.prepare(
                    [queries] * batch, hit=hit, order=order, generation=step + 2
                )
                if graph_mode:
                    graph.replay()
                    assert case.addresses(output) == addresses
                else:
                    output = case.invoke()
                case.check(output)


@pytest.mark.parametrize(
    "length",
    [1, 3, 15, 16, 17, 31, 32, 33, 127, 128, 129, 2047, 2048, 2049, 2051],
)
def test_short_causal_masks_holes_and_odd_tail(device, length):
    queries = min(6, length)
    with _attention_case(device, 2 * queries, 2) as case:
        case.prepare([queries, queries], length=length, holes=True, hit=80)
        case.check(case.invoke())
        graph, output = case.capture()
        for hit, order in ((0, "reverse"), (99, "random"), (100, "contiguous")):
            case.prepare(
                [queries, queries], length=length, holes=True, hit=hit, order=order
            )
            graph.replay()
            case.check(output)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("direct", [False, True], ids=["paged", "direct"])
def test_production_heads_and_selection_table_stride(device, heads, direct):
    with _attention_case(device, 2, 1, heads=heads, direct=direct) as case:
        case.prepare([2])
        case.check(case.invoke())
        graph, output = case.capture()
        for hit in (0, 80, 100):
            case.prepare([2], hit=hit, order="reverse")
            graph.replay()
            case.check(output)
            case.warm()
            graph.replay()
            case.check(output)


def test_generation_cold_warm_and_repeated_graph_replay(device):
    with _attention_case(device, 6, 2) as case:
        case.prepare([3, 3])
        case.check(case.invoke())
        graph, output = case.capture()
        addresses = case.addresses(output)
        for generation in (2, 3):
            case.prepare(
                [1, 5],
                generation=generation,
                order="reverse",
                preserve_resident=True,
            )
            graph.replay()
            case.check(output)
            case.warm()
            for replay in range(70):
                case.query_cpu.add_(0.00390625)
                case.query.copy_(case.query_cpu)
                graph.replay()
                case.check(output)
                assert case.addresses(output) == addresses


def test_graph_dynamic_requests_queries_and_core_boundaries(device, record_property):
    cores = int(torch.npu.get_device_properties(device).cube_core_num)
    assert cores >= 4
    half = cores // 2
    totals = [1, half - 1, half, half + 1, cores, cores + 2, half, 1]
    record_property("cube_core_num", cores)
    record_property("active_query_totals", str(totals))
    maximum = max(totals)
    with _attention_case(device, maximum, len(_query_counts(maximum))) as case:
        case.prepare(_query_counts(maximum))
        case.check(case.invoke())
        graph, output = case.capture()
        addresses = case.addresses(output)
        for step, total in enumerate(totals * 2):
            counts = _query_counts(total)
            if step % 2:
                counts.reverse()
                if len(counts) < case.batch:
                    counts.insert(0, 0)
            case.prepare(
                counts,
                hit=(0, 80, 99, 100)[step % 4],
                order=("random", "reverse", "contiguous")[step % 3],
                generation=step + 2,
                empty_last=step == 3,
            )
            graph.replay()
            case.check(output)
            assert case.addresses(output) == addresses


def test_graph_control_header_plan_relocation_and_layout_changes(device):
    with _attention_case(device, 6, 2) as case:
        case.prepare([3, 3], hit=80)
        case.check(case.invoke())
        graph, output = case.capture()
        addresses = case.addresses(output)
        for step, offset in enumerate((0, 4097, CONTROL - WIDTH, 2049, 0)):
            case.direct = step % 2 == 0
            case.prepare(
                [1, 5] if step % 2 else [4, 2],
                hit=(0, 80, 99, 100, 80)[step],
                generation=step + 2,
                holes=True,
            )
            case.membership_cpu[:, :CONTROL].fill_(-1)
            case.membership_cpu[:, offset : offset + WIDTH] = case.plans
            control = case.membership_cpu[:, CONTROL : CONTROL + 8]
            control[:, 2] = WIDTH + (7 if step % 2 else 0)
            control[:, 3] = offset
            case.membership.copy_(case.membership_cpu)
            graph.replay()
            case.check(output)
            assert case.addresses(output) == addresses


@pytest.mark.parametrize("field,value", [(1, 0), (2, 127), (2, 128), (3, CONTROL)])
@pytest.mark.parametrize("hit", [0, 100])
def test_graph_control_header_fallback_preserves_selection(device, field, value, hit):
    with _attention_case(device, 6, 2) as case:
        case.prepare([3, 3], hit=100)
        case.check(case.invoke())
        graph, output = case.capture()
        for generation in (2, 3):
            case.prepare([2, 4], hit=hit, generation=generation)
            case.membership_cpu[:, CONTROL + field] = value
            case.membership.copy_(case.membership_cpu)
            effective_plans = -case.plans.abs()
            if field == 2 and value == 128:
                effective_plans[:, :128] = case.plans[:, :128]
            expected = _attention_oracle(
                case.query_cpu,
                case.host,
                case.topk_cpu,
                case.table_cpu,
                case.counts,
                case.lengths,
                effective_plans,
                case.resident_cpu,
            )
            expected_resident = case.resident_cpu.clone()
            if hit == 0 and field == 2 and value == 128:
                row = 0
                for request, count in enumerate(case.counts):
                    for route in range(count):
                        tokens = case.topk_cpu[row, :128].long()
                        physical = case.table_cpu[request, tokens // PAGE].long()
                        physical = physical * PAGE + tokens % PAGE
                        slots = -case.plans[row, :128].long() - 1
                        expected_resident[row, slots] = case.host[physical]
                        row += 1
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected, atol=0.002, rtol=0.025)
            assert torch.equal(
                case.resident.cpu().view(torch.int16),
                expected_resident.view(torch.int16),
            )
            case.publish_plan()
            graph.replay()
            case.check(output)


@pytest.mark.parametrize("heads", [32, 128])
def test_graph_control_header_remaps_selection_rows_pages_and_capacity(device, heads):
    rows = 18
    with _attention_case(device, rows, 3, heads=heads) as case:
        case.prepare([6, 6, 6], hit=100)
        case.check(case.invoke())
        graph, output = case.capture()
        generator = torch.Generator().manual_seed(920)
        pages_per_row = CAPACITY // PAGE
        for step, capacity in enumerate((CAPACITY, CAPACITY - 1024, CAPACITY)):
            case.direct = step % 2 == 0
            case.prepare([6, 6, 6], hit=100, generation=step + 2)
            if capacity < CAPACITY:
                for row in range(rows):
                    columns = (case.plans[row] > 0).nonzero().flatten()
                    assert columns.numel() <= capacity
                    slots = torch.randperm(capacity, generator=generator)
                    case.plans[row, columns] = slots[: columns.numel()].short() + 1
                case.publish_plan()
            data_rows = torch.randperm(rows, generator=generator)
            page_map = torch.randperm(rows * pages_per_row, generator=generator)
            physical_resident = torch.full_like(case.resident_cpu, -9).view(-1, DIM)
            physical_indices = torch.arange(CAPACITY).expand(rows, -1).clone()
            for row in range(rows):
                data_row = int(data_rows[row])
                slots = torch.arange(CAPACITY)
                if case.direct:
                    physical_indices[row] = data_row * capacity + slots
                else:
                    pages = page_map.view(rows, pages_per_row)[data_row]
                    physical_indices[row] = pages[slots // PAGE] * PAGE + slots % PAGE
                hits = (case.plans[row] > 0).nonzero().flatten()
                hit_slots = case.plans[row, hits].long() - 1
                payload = torch.full((hits.numel(), DIM), 0.125 + row / 32)
                case.resident_cpu[row, hit_slots] = payload.bfloat16()
                physical_resident[physical_indices[row, hit_slots]] = payload.bfloat16()
            control = case.membership_cpu[:, CONTROL : CONTROL + 8]
            control[:, 4] = data_rows.short()
            control[:, 6] = capacity
            case.resident.copy_(physical_resident.view_as(case.resident))
            case.selection_table.copy_(page_map.view(rows, pages_per_row).int())
            case.membership.copy_(case.membership_cpu)
            expected = _attention_oracle(
                case.query_cpu,
                case.host,
                case.topk_cpu,
                case.table_cpu,
                case.counts,
                case.lengths,
                case.plans,
                case.resident_cpu,
            )
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected, atol=0.002, rtol=0.025)
            assert torch.equal(
                case.resident.cpu().view(torch.int16),
                physical_resident.view_as(case.resident_cpu).view(torch.int16),
            )


@pytest.mark.parametrize("batch,queries", [(1, 6), (10, 1), (10, 6)])
def test_mapped_host_control_header_dynamic_graph_replay(device, batch, queries):
    with _attention_case(device, batch * queries, batch, heads=32) as case:
        case.use_host_plan()
        case.prepare([queries] * batch, hit=80)
        case.check(case.invoke())
        graph, output = case.capture()
        addresses = case.addresses(output)
        for generation, hit in enumerate((0, 80, 99, 100, 80), start=2):
            case.prepare(
                [queries] * batch,
                hit=hit,
                generation=generation,
                holes=generation % 2 == 0,
            )
            if generation % 2:
                offset = 4097
                case.membership_cpu[:, :CONTROL].fill_(-1)
                case.membership_cpu[:, offset : offset + WIDTH] = case.plans
                case.membership_cpu[:, CONTROL + 3] = offset
                case.membership.copy_(case.membership_cpu)
            graph.replay()
            case.check(output)
            assert case.addresses(output) == addresses


if __name__ == "__main__":
    run_pytest(__file__)
