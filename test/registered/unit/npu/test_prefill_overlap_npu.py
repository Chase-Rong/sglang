"""Real NPU pipelined Prefill Host<->device overlap for the integrated offload.

These tests drive ``PipelinedIntegratedOffloadRuntime`` against a real
``NPUMLATokenToKVPool``, the registered Host KV backing and the real NPU
scatter/copy kernels; nothing about the hardware is mocked.  They opt in like
the other NPU pool tests::

    SGLANG_FSA_TEST_ATTENTION=1 SGLANG_FSA_TEST_DEVICE_ID=3 \
        python -m pytest test/registered/unit/npu/test_prefill_overlap_npu.py

``exercise_pipelined_overlap`` and its siblings are reusable so a torchrun TP
runner can call them once per rank with the shared Host group.  Every rank must
issue the identical logical writes, and no rank reads a Host KV region before
``runtime.drain`` plus a device synchronize; no metadata file is shared
between ranks.
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires an explicitly selected idle NPU and the FSA native helpers",
)

LAYER_NUM = 4
PREFILL_SLOTS = 2
KV_LORA_RANK = 512
ROPE_DIM = 64
KV_WIDTH = KV_LORA_RANK + ROPE_DIM
PAGE_SIZE = 128
POOL_SIZE = 512
POOL_ROWS = (POOL_SIZE // PAGE_SIZE + 1) * PAGE_SIZE
MAPPING_ROWS = 256
SLOT_ROW_SPAN = POOL_SIZE // 2
REQUEST_IDS = ("pipelined-a", "pipelined-b")
CHUNKS = ((0, 128), (128, 192), (192, 256))
SETTLED_PASSES = ((0, 128), (128, 128))
DELAYED_PASSES = ((0, 128), (128, 192))
DECODE_INDEX = 200
DECODE_GENERATION = 7
ATTENTION_TOLERANCE = 2e-3


def _extend_mode():
    """Extend-without-speculative batch mode, tolerant of the checked accessor."""
    return SimpleNamespace(
        is_extend=lambda: True,
        is_extend_without_speculative=lambda: True,
        is_decode=lambda: False,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )


def _request_rows(device):
    """Two scattered, non-zero-based pool row mappings in disjoint row halves.

    The halves must stay disjoint: if one request's current write touched a row
    that the other request restores from an older generation, the expected
    value of that row would legitimately be ambiguous.
    """
    base = torch.arange(MAPPING_ROWS, dtype=torch.int64)
    first = PAGE_SIZE + (base * 5) % SLOT_ROW_SPAN
    second = PAGE_SIZE + SLOT_ROW_SPAN + (base * 7) % SLOT_ROW_SPAN
    rows = torch.stack((first, second)).to(device)
    assert int(rows.min()) >= PAGE_SIZE and int(rows.max()) < POOL_ROWS
    assert len(set(first.tolist())) == MAPPING_ROWS
    assert len(set(second.tolist())) == MAPPING_ROWS
    assert not set(first.tolist()) & set(second.tolist())
    return rows


def bind_hicache_rows(pool, rows):
    """Bind real borrowed Host pages in reverse order before any forward."""
    from sglang.srt.hardware_backend.npu.hicache_host_pool import MemFabricHiCachePool

    backing = pool.decode_host_backing
    host_pool = backing.bind_hicache(pool)
    assert isinstance(host_pool, MemFabricHiCachePool)
    assert host_pool.binding is backing.page_binding
    pages = torch.unique(rows.detach().cpu() // pool.page_size)
    assert pages.numel() > 1 and pages.min() > 0
    for page in pages.flip(0):
        assert backing.page_binding.ensure_pages(page.reshape(1))
    mapped = backing.page_binding.device_to_host_cpu[pages]
    assert torch.all(mapped > 0) and torch.any(mapped != pages)
    for layer, regions in enumerate(backing.regions):
        for view, region in zip(host_pool.layer_views(layer), regions):
            if region is not None:
                assert view.data_ptr() == region.tensor.data_ptr()
    shared = backing.pool.shared_group
    if shared is not None:
        mappings = shared.group.all_gather_object(mapped.tolist())
        assert all(mapping == mappings[0] for mapping in mappings)
    return host_pool


def _row_generations(length, passes):
    """Generation that last wrote each sequence position within ``passes``."""
    generations = torch.zeros(length, dtype=torch.int64)
    for generation, (prefix, chunk_length) in enumerate(passes):
        if prefix < length:
            generations[prefix : min(chunk_length, length)] = generation
    return generations


def _logical_rows(layer, positions, generation, *, width, dtype):
    """Deterministic KV rows: distinct per row/layer/generation, exact in fp16/bf16."""
    rows = torch.as_tensor(positions, dtype=torch.int64).reshape(-1, 1)
    columns = torch.arange(width, dtype=torch.int64, device=rows.device).reshape(1, -1)
    if not torch.is_tensor(generation):
        generation = torch.full_like(rows, int(generation))
    generation = generation.reshape(-1, 1).to(torch.int64).to(rows.device)
    code = (rows * 13 + columns * 7 + generation * 211 + layer * 97 + width * 3) % 509
    return (code - 254).to(torch.float32).div(512).to(dtype)


def _logical_kv(layer, positions, generation, *, dtype):
    return torch.cat(
        (
            _logical_rows(
                layer, positions, generation, width=KV_LORA_RANK, dtype=dtype
            ),
            _logical_rows(layer, positions, generation, width=ROPE_DIM, dtype=dtype),
        ),
        dim=-1,
    )


def _attention_query(layer, chunk, device):
    generator = torch.Generator().manual_seed(10_000 + layer * 97 + chunk * 13)
    query = torch.randn(2, KV_WIDTH, generator=generator) * (KV_WIDTH**-0.5) * 8.0
    return query.to(device)


def _attention(query, keys):
    """Plain torch matmul/softmax attention: every query row reads every key row."""
    keys = keys.to(torch.float32)
    weights = torch.softmax(query.to(torch.float32) @ keys.T, dim=-1)
    return weights @ keys


def _prefill_batch(device, rows, prefix, length):
    locations = torch.cat([rows[slot, prefix:length] for slot in range(rows.shape[0])])
    return SimpleNamespace(
        rids=list(REQUEST_IDS),
        seq_lens_cpu=torch.tensor([length] * len(REQUEST_IDS), dtype=torch.int64),
        extend_prefix_lens_cpu=[prefix] * len(REQUEST_IDS),
        req_pool_indices=torch.arange(
            len(REQUEST_IDS), dtype=torch.int64, device=device
        ),
        out_cache_loc=locations.to(torch.int64).contiguous(),
        forward_mode=_extend_mode(),
    )


def _scatter_layer(pool, layer, rows, prefix, length, generation, *, dtype):
    """Write this chunk's new rows for one layer through the real pool API."""
    new_rows = rows[:, prefix:length].reshape(-1)
    latent = _logical_rows(layer, new_rows, generation, width=KV_LORA_RANK, dtype=dtype)
    rope = _logical_rows(layer, new_rows, generation, width=ROPE_DIM, dtype=dtype)
    pool.set_kv_buffer(
        SimpleNamespace(layer_id=layer),
        new_rows,
        latent.unsqueeze(1).contiguous(),
        rope.unsqueeze(1).contiguous(),
    )


def _add_check(checks, actual, expected, label, tolerance):
    checks.append((actual, expected, label, tolerance))


def _collect_restore_checks(
    checks, layer, latent_rows, rope_rows, rows, prefix, chunk, passes, *, dtype
):
    restored_rows = rows[:, :prefix].reshape(-1)
    generations = _row_generations(prefix, passes).repeat(rows.shape[0])
    label = f"layer {layer} chunk {chunk} prefix restore"
    _add_check(
        checks,
        latent_rows[restored_rows],
        _logical_rows(
            layer, restored_rows, generations, width=KV_LORA_RANK, dtype=dtype
        ),
        f"{label} latent",
        0.0,
    )
    _add_check(
        checks,
        rope_rows[restored_rows],
        _logical_rows(layer, restored_rows, generations, width=ROPE_DIM, dtype=dtype),
        f"{label} rope",
        0.0,
    )


def _collect_attention_checks(
    checks, layer, latent_rows, rope_rows, rows, length, chunk, passes, *, dtype
):
    generations = _row_generations(length, passes)
    query = _attention_query(layer, chunk, latent_rows.device)
    for slot in range(rows.shape[0]):
        sequence_rows = rows[slot, :length]
        actual = torch.cat(
            (latent_rows[sequence_rows], rope_rows[sequence_rows]), dim=-1
        )
        oracle = _logical_kv(layer, sequence_rows, generations, dtype=dtype)
        _add_check(
            checks,
            _attention(query, actual),
            _attention(query, oracle),
            f"layer {layer} chunk {chunk} slot {slot} attention",
            ATTENTION_TOLERANCE,
        )


def _compare_checks(checks):
    for actual, expected, label, tolerance in checks:
        torch.testing.assert_close(
            actual.float().cpu(),
            expected.float().cpu(),
            rtol=tolerance,
            atol=tolerance,
            msg=label,
        )
    checks.clear()


def _check_host_kv(backing, rows, length, passes, *, dtype):
    if (
        backing.pool.shared_group is not None
        and not backing.pool.shared_group.is_writer
    ):
        return
    cpu_rows = rows[:, :length].reshape(-1).cpu()
    host_rows = backing.page_binding.host_rows(rows[:, :length].reshape(-1)).cpu()
    generations = _row_generations(length, passes).repeat(rows.shape[0])
    for layer in range(LAYER_NUM):
        latent_region, rope_region = backing.regions[layer]
        torch.testing.assert_close(
            latent_region.tensor[host_rows].cpu(),
            _logical_rows(
                layer, cpu_rows, generations, width=KV_LORA_RANK, dtype=dtype
            ).cpu(),
            rtol=0,
            atol=0,
            msg=f"host latent layer {layer}",
        )
        torch.testing.assert_close(
            rope_region.tensor[host_rows].cpu(),
            _logical_rows(
                layer, cpu_rows, generations, width=ROPE_DIM, dtype=dtype
            ).cpu(),
            rtol=0,
            atol=0,
            msg=f"host rope layer {layer}",
        )


def _check_decode_replay(pool, backing, rows, device, *, dtype):
    """A later ordinary Decode current write must still replay inside a Graph."""
    layer = LAYER_NUM - 1
    locations = torch.tensor(
        (
            int(rows[0, DECODE_INDEX]),
            int(rows[1, DECODE_INDEX + 1]),
            int(rows[0, DECODE_INDEX + 2]),
        ),
        dtype=torch.int64,
        device=device,
    )
    layer_view = SimpleNamespace(layer_id=layer)
    values = (
        _logical_rows(
            layer, locations, DECODE_GENERATION, width=KV_LORA_RANK, dtype=dtype
        )
        .unsqueeze(1)
        .contiguous()
    )
    rope = (
        _logical_rows(layer, locations, DECODE_GENERATION, width=ROPE_DIM, dtype=dtype)
        .unsqueeze(1)
        .contiguous()
    )
    graph = torch.npu.NPUGraph()
    try:
        pool.set_kv_buffer(layer_view, locations, values, rope)
        with torch.npu.graph(graph):
            pool.set_kv_buffer(layer_view, locations, values, rope)
        updated = _logical_rows(
            layer, locations, DECODE_GENERATION + 1, width=KV_LORA_RANK, dtype=dtype
        ).unsqueeze(1)
        updated_rope = _logical_rows(
            layer, locations, DECODE_GENERATION + 1, width=ROPE_DIM, dtype=dtype
        ).unsqueeze(1)
        values.copy_(updated)
        rope.copy_(updated_rope)
        graph.replay()
        torch.npu.synchronize()
        if (
            backing.pool.shared_group is not None
            and not backing.pool.shared_group.is_writer
        ):
            return
        host_rows = backing.page_binding.host_rows(locations).cpu()
        torch.testing.assert_close(
            backing.regions[layer][0].tensor[host_rows].cpu(),
            updated.squeeze(1).cpu(),
            rtol=0,
            atol=0,
            msg="Decode replay latent current write",
        )
        torch.testing.assert_close(
            backing.regions[layer][1].tensor[host_rows].cpu(),
            updated_rope.squeeze(1).cpu(),
            rtol=0,
            atol=0,
            msg="Decode replay rope current write",
        )
    finally:
        graph.reset()


def _exercise_pipelined_overlap(
    dtype,
    device_id,
    *,
    h2d_prefetch,
    d2h_async,
    passes,
    interleave=None,
    decode_replay=True,
    shared_group=None,
):
    """Run ``passes`` of (prefix, length) Prefill chunks over a real NPU Host pool.

    The layer loop only enqueues work: restored prefixes, attention outputs and
    Host rows are collected as device tensors and compared after ``drain``, so
    the asynchronous H2D/D2H overlap is actually exercised.  ``interleave`` is
    called as ``interleave("pass", chunk)`` and ``interleave("layer", layer)``
    so a caller can queue real device work between the Host writes and their
    later readers.
    """
    import torch_npu

    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.prefill_overlap import (
        PipelinedIntegratedOffloadRuntime,
    )

    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    rows = _request_rows(device)
    pool = NPUMLATokenToKVPool(
        size=POOL_SIZE,
        page_size=PAGE_SIZE,
        dtype=dtype,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=ROPE_DIM,
        index_head_dim=128,
        layer_num=LAYER_NUM,
        device=f"npu:{device_id}",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYER_NUM,
        index_kpool=4,
        index_kpool_compress=True,
        max_running_requests=len(REQUEST_IDS),
        prefill_layer_slots=PREFILL_SLOTS,
        decode_host_budget=1 << 24,
        integrated_offload=True,
        shared_host_group=shared_group,
    )
    assert pool.k_buffer.shape[0] == PREFILL_SLOTS
    assert pool.index_k_buffer.shape[0] == LAYER_NUM
    runtime = PipelinedIntegratedOffloadRuntime(
        pool,
        SimpleNamespace(req_to_token=rows),
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
    )
    pool.prefill_offload_runtime = runtime
    backing = pool.decode_host_backing
    checks = []
    slot_pointers = []
    drained = False
    try:
        bind_hicache_rows(pool, rows)
        for chunk, (prefix, length) in enumerate(passes):
            if interleave is not None:
                interleave("pass", chunk)
            with runtime.forward(_prefill_batch(device, rows, prefix, length)):
                for layer in range(LAYER_NUM):
                    runtime.prepare_layer(layer)
                    latent_slot, rope_slot = pool.get_kv_buffer(layer)
                    if chunk == 0:
                        slot_pointers.append(latent_slot.data_ptr())
                    if length > prefix:
                        _scatter_layer(
                            pool, layer, rows, prefix, length, chunk, dtype=dtype
                        )
                    if interleave is not None:
                        interleave("layer", layer)
                    runtime.on_compute_start(layer)
                    latent_rows = latent_slot.view(-1, KV_LORA_RANK)
                    rope_rows = rope_slot.view(-1, ROPE_DIM)
                    if prefix:
                        _collect_restore_checks(
                            checks,
                            layer,
                            latent_rows,
                            rope_rows,
                            rows,
                            prefix,
                            chunk,
                            passes,
                            dtype=dtype,
                        )
                    _collect_attention_checks(
                        checks,
                        layer,
                        latent_rows,
                        rope_rows,
                        rows,
                        length,
                        chunk,
                        passes,
                        dtype=dtype,
                    )
                    runtime.finish_layer(layer)
            assert backing.page_binding.ready_event is runtime.host_visible
            assert runtime.prefix_plan.shape[1] == runtime.new_plan.shape[1] == 3
        runtime.drain()
        drained = True
        torch.npu.synchronize()
        assert not runtime.active and not runtime.failed
        assert bool(runtime.d2h_async) == d2h_async
        assert len(slot_pointers) == LAYER_NUM
        assert (
            slot_pointers[0] == slot_pointers[2]
            and slot_pointers[1] == slot_pointers[3]
        )
        assert slot_pointers[0] != slot_pointers[1]
        _compare_checks(checks)
        host_error = None
        try:
            _check_host_kv(backing, rows, passes[-1][1], passes, dtype=dtype)
        except AssertionError as error:
            host_error = str(error)
        host_errors = (
            [host_error]
            if shared_group is None
            else shared_group.group.all_gather_object(host_error)
        )
        assert not any(host_errors), host_errors
        if decode_replay:
            _check_decode_replay(pool, backing, rows, device, dtype=dtype)
    finally:
        torch.npu.synchronize()
        try:
            if not drained:
                runtime.drain()
        finally:
            pool.decode_host_backing.pool.close(drained=True)


def exercise_pipelined_overlap(
    dtype, device_id, *, h2d_prefetch, d2h_async, shared_group=None
):
    """Three growing-prefix chunks over two slots, then a settled Decode replay."""
    _exercise_pipelined_overlap(
        dtype,
        device_id,
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
        passes=CHUNKS,
        shared_group=shared_group,
    )


def exercise_pipelined_settled_restore(
    dtype, device_id, *, h2d_prefetch, d2h_async, shared_group=None
):
    """One write generation followed by a Prefill pass that adds no new rows."""
    _exercise_pipelined_overlap(
        dtype,
        device_id,
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
        passes=SETTLED_PASSES,
        shared_group=shared_group,
    )


def exercise_pipelined_delayed_work(
    dtype, device_id, *, h2d_prefetch, d2h_async, shared_group=None
):
    """Queue real device work between each Host write and its later readers."""
    import torch_npu

    torch_npu.npu.set_device(device_id)
    heavy = torch.randn(
        2048, 2048, dtype=dtype, device=torch.device(f"npu:{device_id}")
    )

    def interleave(kind, index):
        torch.mm(heavy, heavy)
        torch.softmax(heavy, dim=-1)

    _exercise_pipelined_overlap(
        dtype,
        device_id,
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
        passes=DELAYED_PASSES,
        interleave=interleave,
        shared_group=shared_group,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "h2d_prefetch,d2h_async",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_pipelined_overlap_three_chunk_prefix(dtype, h2d_prefetch, d2h_async):
    exercise_pipelined_overlap(
        dtype,
        int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"]),
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("h2d_prefetch,d2h_async", [(False, False), (True, True)])
def test_pipelined_overlap_no_new_row_generation(dtype, h2d_prefetch, d2h_async):
    exercise_pipelined_settled_restore(
        dtype,
        int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"]),
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("h2d_prefetch,d2h_async", [(True, True), (False, True)])
def test_pipelined_overlap_delayed_writer_and_reader(dtype, h2d_prefetch, d2h_async):
    exercise_pipelined_delayed_work(
        dtype,
        int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"]),
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
    )


def profile_pipelined_overlap(
    device_id,
    destination,
    *,
    dtype=torch.float16,
    h2d_prefetch=True,
    d2h_async=True,
    shared_group=None,
):
    """Opt-in focused torch_npu profile of one overlap pass; no timing claims."""
    import torch_npu

    os.makedirs(destination, exist_ok=True)
    exercise_pipelined_overlap(
        dtype,
        device_id,
        h2d_prefetch=h2d_prefetch,
        d2h_async=d2h_async,
        shared_group=shared_group,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(destination),
    ):
        exercise_pipelined_overlap(
            dtype,
            device_id,
            h2d_prefetch=h2d_prefetch,
            d2h_async=d2h_async,
            shared_group=shared_group,
        )


@pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_PROFILE") != "1",
    reason="opt-in torch_npu profile entrypoint",
)
def test_profile_pipelined_overlap(tmp_path):
    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    destination = tmp_path / f"pipelined-overlap-device-{device_id}"
    profile_pipelined_overlap(device_id, str(destination))
    assert any(destination.iterdir())
