"""Exercise production metadata methods without loading model weights.

Set SGLANG_FRAMEWORK_UT_DEVICE=npu:0 for device and Graph checks. The optional
benchmark uses an untouched source checkout supplied by --baseline-root.
"""

import argparse
import ast
import json
import os
import statistics
import time
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path("python/sglang/srt")
DEVICE = os.environ.get("SGLANG_FRAMEWORK_UT_DEVICE", "cpu")
if DEVICE.startswith("npu"):
    import torch_npu

    assert torch_npu.npu.is_available()
    torch.npu.set_device(DEVICE)


def production_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def profile_namespace(root):
    namespace = dict(
        torch=torch,
        ExitStack=ExitStack,
        contextmanager=contextmanager,
        wraps=wraps,
        _NULL_CONTEXT=nullcontext(),
        _nvtx_module=None,
    )
    production_functions(
        root / SOURCE / "utils/nvtx_utils.py",
        {"_profile_range_impl", "profile_range", "profile_method"},
        namespace,
    )
    return namespace


def production_class(root, relative, name, methods, namespace=None):
    namespace = profile_namespace(root) if namespace is None else namespace
    path = root / SOURCE / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == name)
    cls.bases = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def metadata_class(root):
    namespace = profile_namespace(root)
    namespace["AscendDSAForwardMetadata"] = SimpleNamespace

    def expand(
        lengths, query_lens, device, total, maximum, *, seqlens_out, token_to_batch_out
    ):
        offsets = torch.arange(maximum, device=device)
        seqlens_out.copy_(
            (lengths[:, None] - query_lens[:, None] + offsets + 1).reshape(-1)
        )
        token_to_batch_out.copy_(
            torch.arange(lengths.numel(), device=device).repeat_interleave(maximum)
        )
        return seqlens_out, token_to_batch_out

    namespace["_expand_causal_seqlens_device"] = expand
    return production_class(
        root,
        "hardware_backend/npu/attention/ascend_dsa_backend.py",
        "AscendDSAAttnBackend",
        {
            "_cache_seqlens",
            "_cache_seqlens_cpu_max",
            "_query_lens_device",
            "_query_layout_sizes",
            "_replay_dsa_metadata",
        },
        namespace,
    )


def mode(name):
    return SimpleNamespace(
        is_target_verify=lambda: name == "verify",
        is_draft_extend_v2=lambda: name == "draft_extend",
        is_extend=lambda: name in ("verify", "draft_extend"),
        is_decode_or_idle=lambda: name in ("draft", "decode"),
        is_idle=lambda: False,
    )


def metadata_case(root, batch, length, name="verify", page_size=128):
    backend = metadata_class(root)()
    capacity = (length + 256 + page_size - 1) // page_size * page_size
    table = torch.arange(
        (batch + 2) * capacity, dtype=torch.int32, device=DEVICE
    ).reshape(batch + 2, capacity)
    table[0].zero_()
    backend.req_to_token_pool = SimpleNamespace(req_to_token=table)
    backend.page_size = page_size
    backend.speculative_num_draft_tokens = 6
    backend.speculative_step_id = 4
    backend.dsa_index_kpool = 1
    width = 6 if name in ("verify", "draft_extend") else 1
    backend.forward_metadata = SimpleNamespace(
        cache_seqlens_int32=torch.empty(batch, dtype=torch.int32, device=DEVICE),
        real_page_table=torch.zeros(
            batch, capacity // page_size, dtype=torch.int32, device=DEVICE
        ),
        host_page_table=None,
        dsa_seqlens_expanded=torch.empty(
            batch * width, dtype=torch.int32, device=DEVICE
        ),
        token_to_batch_idx=torch.empty(batch * width, dtype=torch.int64, device=DEVICE),
        kpool_write_plan=None,
    )
    lengths = torch.full((batch,), length, dtype=torch.int64)
    forward = SimpleNamespace(
        batch_size=batch,
        seq_lens=lengths.to(DEVICE),
        seq_lens_cpu=lengths,
        req_pool_indices=torch.arange(1, batch + 1, dtype=torch.int64, device=DEVICE),
        forward_mode=mode(name),
        spec_info=SimpleNamespace() if name == "draft" else None,
        input_ids=torch.zeros(batch * width, dtype=torch.int64, device=DEVICE),
        extend_seq_lens=None,
    )
    return backend, forward


@pytest.mark.parametrize("batch", [1, 10])
@pytest.mark.parametrize("length", [127, 128, 129, 256000])
@pytest.mark.parametrize("name", ["verify", "draft_extend", "draft", "decode"])
def test_replay_metadata_preserves_values_and_addresses(batch, length, name):
    backend, forward = metadata_case(ROOT, batch, length, name)
    metadata = backend.forward_metadata
    pointers = {
        key: value.data_ptr()
        for key, value in vars(metadata).items()
        if isinstance(value, torch.Tensor)
    }
    for generation in range(3):
        forward.seq_lens_cpu.fill_(length + generation)
        forward.seq_lens.copy_(forward.seq_lens_cpu)
        if generation == 1:
            forward.req_pool_indices.copy_(forward.req_pool_indices.flip(0))
        before = forward.seq_lens.clone()
        backend._replay_dsa_metadata(forward)
        offset = 6 if name == "verify" else 5 if name == "draft" else 0
        torch.testing.assert_close(
            metadata.cache_seqlens_int32, (before + offset).int(), rtol=0, atol=0
        )
        maximum = length + generation + offset
        golden = (
            backend.req_to_token_pool.req_to_token[forward.req_pool_indices, :maximum][
                :, ::128
            ]
            // 128
        )
        torch.testing.assert_close(
            metadata.real_page_table[:, : golden.shape[1]], golden, rtol=0, atol=0
        )
        torch.testing.assert_close(before, forward.seq_lens, rtol=0, atol=0)
        assert all(
            getattr(metadata, key).data_ptr() == pointer
            for key, pointer in pointers.items()
        )


def allocator_class(root):
    namespace = profile_namespace(root)
    namespace["get_num_new_pages"] = lambda *, seq_lens, page_size, decode: int(
        (seq_lens % page_size == 1).sum()
    )
    return production_class(
        root,
        "hardware_backend/npu/allocator_npu.py",
        "NPUPagedTokenToKVPoolAllocator",
        {"alloc_decode", "_reserve_host_pages"},
        namespace,
    )


@pytest.mark.parametrize("batch", [1, 10])
@pytest.mark.parametrize("new_pages", [False, True])
def test_allocator_page_boundaries_and_ownership(batch, new_pages):
    allocator = allocator_class(ROOT)()
    allocator.page_size = 128
    allocator.debug_mode = True
    allocator.free_pages = torch.arange(40, 80, device=DEVICE, dtype=torch.int64)
    calls = []
    allocator.offload_backing = SimpleNamespace(
        page_binding=SimpleNamespace(
            ensure_pages=lambda pages: calls.append(pages.clone()) or True
        )
    )
    lengths = torch.full((batch,), 130, dtype=torch.int64)
    if new_pages:
        lengths[::2] = 129
    last = torch.arange(1, batch + 1, device=DEVICE, dtype=torch.int64) * 128
    last += (lengths.to(DEVICE) - 2) % 128
    golden = (last + 1).int()
    mask = lengths % 128 == 1
    count = int(mask.sum())
    golden[mask.to(DEVICE)] = allocator.free_pages[:count].int() * 128
    result = allocator.alloc_decode(lengths.to(DEVICE), lengths, last)
    torch.testing.assert_close(result, golden, rtol=0, atol=0)
    assert len(calls) == int(new_pages)
    assert allocator.free_pages.numel() == 40 - count


def test_replay_profile_range_is_visible():
    backend, forward = metadata_case(ROOT, 1, 128)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        backend._replay_dsa_metadata(forward)
    assert "npu.dsa.replay_metadata" in {event.key for event in profiler.key_averages()}


def test_host_page_table_replay_clears_padding_and_follows_remapping():
    from sglang.srt.hardware_backend.npu.hicache_offload import OffloadPageBinding

    backend, forward = metadata_case(ROOT, 10, 129)
    binding = OffloadPageBinding.__new__(OffloadPageBinding)
    binding.device_to_host = torch.arange(100, dtype=torch.int64, device=DEVICE) + 20
    binding.device_to_host[0] = 0
    backend.token_to_kv_pool = SimpleNamespace(
        integrated_offload=True,
        decode_host_backing=SimpleNamespace(page_binding=binding),
    )
    metadata = backend.forward_metadata
    metadata.host_page_table = torch.full_like(metadata.real_page_table, 999)
    address = metadata.host_page_table.data_ptr()
    for _ in range(3):
        backend._replay_dsa_metadata(forward)
        columns = (int(forward.seq_lens_cpu.max()) + 6 + 127) // 128
        golden = binding.device_to_host[
            metadata.real_page_table[:, :columns].long()
        ].int()
        torch.testing.assert_close(
            metadata.host_page_table[:, :columns], golden, rtol=0, atol=0
        )
        assert torch.count_nonzero(metadata.host_page_table[:, columns:]).item() == 0
        assert metadata.host_page_table.data_ptr() == address
        binding.device_to_host.add_(1)
        binding.device_to_host[0] = 0
        forward.req_pool_indices.copy_(forward.req_pool_indices.flip(0))


@pytest.mark.parametrize("batch", [1, 10])
def test_resolved_lengths_match_published_request_order(batch):
    namespace = profile_namespace(ROOT)
    namespace.update(_DEBUG_ASSERT=False, _is_hip=False)
    future_type = production_class(
        ROOT,
        "managers/overlap_utils.py",
        "FutureMap",
        {"resolve_seq_lens_cpu"},
        namespace,
    )
    future = future_type()
    future.new_seq_lens_buf = (
        torch.arange(20, dtype=torch.int64, device=DEVICE) + 256000
    )
    future.publish_ready = None
    future.needs_cpu_seq_lens = True
    future.fwd_prepare_d2h_stream = None
    indices = torch.arange(batch, dtype=torch.int64, device=DEVICE).flip(0)
    forward = SimpleNamespace(spec_info=SimpleNamespace(future_indices=indices))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        future.resolve_seq_lens_cpu(forward)
    expected = torch.arange(batch, dtype=torch.int64).flip(0) + 256000
    torch.testing.assert_close(forward.seq_lens_cpu, expected, rtol=0, atol=0)
    assert forward.seq_lens_sum == int(expected.sum())
    assert "overlap.resolve_seq_lens_cpu" in {
        event.key for event in profiler.key_averages()
    }


def test_allocator_rejects_failed_host_reservation_without_consuming_pages():
    allocator = allocator_class(ROOT)()
    allocator.page_size = 128
    allocator.debug_mode = False
    allocator.free_pages = torch.arange(40, 80, device=DEVICE, dtype=torch.int64)
    allocator.offload_backing = SimpleNamespace(
        page_binding=SimpleNamespace(ensure_pages=lambda pages: False)
    )
    before = allocator.free_pages.clone()
    lengths = torch.tensor([129], dtype=torch.int64)
    assert (
        allocator.alloc_decode(
            lengths.to(DEVICE), lengths, torch.tensor([255], device=DEVICE)
        )
        is None
    )
    torch.testing.assert_close(allocator.free_pages, before, rtol=0, atol=0)


@pytest.mark.parametrize("padding", [False, True])
def test_graph_registry_preserves_live_rows_and_padding(padding):
    from sglang.srt.model_executor.cuda_graph_buffer_registry import (
        CudaGraphBufferRegistry,
        GraphSlot,
        PaddingPolicy,
    )

    registry = CudaGraphBufferRegistry(max_bs=10, max_num_tokens=60, device=DEVICE)
    slot = registry.register_slot(
        GraphSlot(
            "req_pool_indices",
            lambda batch, tokens: (batch,),
            torch.int64,
            axis="bs",
            padding_policy=PaddingPolicy.ZERO,
        )
    )
    source = torch.arange(7 if padding else 10, dtype=torch.int64, device=DEVICE)
    forward = SimpleNamespace(req_pool_indices=source)
    slot.buffer.fill_(99)
    address = slot.buffer.data_ptr()
    registry.fill_from(
        forward,
        raw_bs=source.numel(),
        padded_bs=10,
        raw_num_tokens=source.numel() * 6,
        padded_num_tokens=60,
    )
    torch.testing.assert_close(slot.buffer[: source.numel()], source, rtol=0, atol=0)
    if padding:
        assert torch.count_nonzero(slot.buffer[source.numel() :]).item() == 0
    assert slot.buffer.data_ptr() == address


@pytest.mark.skipif(not DEVICE.startswith("npu"), reason="Requires idle Ascend NPU")
def test_metadata_graph_reads_updated_fixed_buffers():
    backend, forward = metadata_case(ROOT, 10, 256000)
    backend._replay_dsa_metadata(forward)
    metadata = backend.forward_metadata
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        read_lengths = metadata.cache_seqlens_int32.clone()
        read_pages = metadata.real_page_table.clone()
    for generation in range(4):
        forward.seq_lens_cpu.add_(1)
        forward.seq_lens.copy_(forward.seq_lens_cpu)
        forward.req_pool_indices.copy_(forward.req_pool_indices.flip(0))
        backend._replay_dsa_metadata(forward)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            read_lengths, metadata.cache_seqlens_int32, rtol=0, atol=0
        )
        torch.testing.assert_close(read_pages, metadata.real_page_table, rtol=0, atol=0)


def measure(function, iterations):
    synchronize = torch.npu.synchronize if DEVICE.startswith("npu") else lambda: None
    for _ in range(5):
        function()
    synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return statistics.median(samples)


def benchmark(baseline_root, output):
    assert DEVICE.startswith("npu"), "Performance qualification requires NPU"
    results = []
    for batch in (1, 10):
        for label, checkout in (("baseline", baseline_root), ("candidate", ROOT)):
            backend, forward = metadata_case(checkout, batch, 256000)
            value = measure(lambda: backend._replay_dsa_metadata(forward), 40)
            results.append(
                dict(
                    case="metadata_replay",
                    batch=batch,
                    variant=label,
                    microseconds=value,
                )
            )
            out = torch.empty(batch, device=DEVICE, dtype=torch.int32)
            if label == "baseline":
                function = lambda: out.copy_(backend._cache_seqlens(forward))
            else:
                function = lambda: backend._cache_seqlens(forward, out=out)
            results.append(
                dict(
                    case="length_buffer_update",
                    batch=batch,
                    variant=label,
                    microseconds=measure(function, 100),
                )
            )
            allocator = allocator_class(checkout)()
            allocator.page_size = 128
            allocator.debug_mode = False
            allocator.free_pages = torch.arange(
                40, 80, device=DEVICE, dtype=torch.int64
            )
            allocator.offload_backing = None
            lengths_cpu = torch.full((batch,), 130, dtype=torch.int64)
            lengths = lengths_cpu.to(DEVICE)
            last = torch.arange(batch, device=DEVICE, dtype=torch.int64) * 128 + 128
            results.append(
                dict(
                    case="decode_no_new_pages",
                    batch=batch,
                    variant=label,
                    microseconds=measure(
                        lambda: allocator.alloc_decode(lengths, lengths_cpu, last), 100
                    ),
                )
            )
            print(json.dumps(results[-3:]), flush=True)
        table = backend.req_to_token_pool.req_to_token
        request_ids = forward.req_pool_indices
        old_gather = (
            lambda: (table[request_ids, :256006][:, ::128] // 128).int().contiguous()
        )
        new_gather = (
            lambda: (table[:, :256006:128][request_ids] // 128).int().contiguous()
        )
        torch.testing.assert_close(old_gather(), new_gather(), rtol=0, atol=0)
        for label, function in (("baseline", old_gather), ("candidate", new_gather)):
            results.append(
                dict(
                    case="page_table_gather",
                    batch=batch,
                    variant=label,
                    microseconds=measure(function, 100),
                )
            )
        destinations = [
            torch.empty(batch * width, dtype=dtype, device=DEVICE)
            for width, dtype in (
                (1, torch.int64),
                (1, torch.int64),
                (6, torch.int32),
                (6, torch.int64),
            )
        ]
        sources = [torch.ones_like(destination) for destination in destinations]
        namespace = dict(
            torch=torch, _has_foreach_copy=hasattr(torch, "_foreach_copy_")
        )
        production_functions(
            baseline_root / SOURCE / "model_executor/cuda_graph_buffer_registry.py",
            {"_grouped_foreach_copy_"},
            namespace,
        )
        original = namespace["_grouped_foreach_copy_"]
        results.append(
            dict(
                case="graph_grouped_input_copy",
                batch=batch,
                variant="production",
                microseconds=measure(lambda: original(destinations, sources), 100),
            )
        )
        for label, asynchronous in (
            ("baseline", False),
            ("nonblocking_candidate", True),
        ):

            def copy_inputs():
                for destination, source in zip(destinations, sources):
                    destination.copy_(source, non_blocking=asynchronous)

            results.append(
                dict(
                    case="graph_d2d_copy",
                    batch=batch,
                    variant=label,
                    microseconds=measure(copy_inputs, 100),
                )
            )
        print(json.dumps(results[-4:]), flush=True)
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.baseline_root, args.output)
