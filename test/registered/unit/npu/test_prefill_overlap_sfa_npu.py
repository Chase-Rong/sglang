"""Real NoPE SFA pipeline and matched component timing; no model TPOT claim."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import pytest
import torch

from test_prefill_overlap_npu import bind_hicache_rows

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires a verified idle NPU, native copy helpers and the A5 OPP",
)


def exercise_sfa_pipeline(prefix, chunk, repeats, *, profile_dir=None, modes=None):
    import torch_npu

    from sglang.srt.hardware_backend.npu.fused_sparse_attention import (
        FusedSparseAttentionState,
        run_fused_sparse_attention,
    )
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.prefill_overlap import (
        PipelinedIntegratedOffloadRuntime,
    )

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    dtype = torch.bfloat16
    sequence = prefix + chunk
    assert prefix >= 2048 and sequence % 128 == 0
    pool = NPUMLATokenToKVPool(
        size=sequence + 128,
        page_size=128,
        dtype=dtype,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
        layer_num=4,
        device=str(device),
        enable_memory_saver=False,
        start_layer=0,
        end_layer=4,
        index_kpool=4,
        index_kpool_compress=True,
        max_running_requests=1,
        prefill_layer_slots=2,
        decode_host_budget=4 * (sequence + 256) * 512 * 2,
        integrated_offload=True,
    )
    backing = pool.decode_host_backing
    mapping = torch.arange(128, sequence + 128, device=device).view(1, -1)
    bind_hicache_rows(pool, mapping)
    host_rows = backing.page_binding.host_rows(mapping[0]).cpu()
    generator = torch.Generator().manual_seed(217)
    source_cpu = (torch.randn(sequence, 512, generator=generator) * 0.125).to(dtype)
    for layer in range(4):
        backing.regions[layer][0].tensor[host_rows] = source_cpu + layer * 0.03125
    request_pool = SimpleNamespace(
        req_to_token=torch.cat((torch.zeros_like(mapping), mapping)),
        req_generation=torch.tensor([0, 1]),
    )
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[prefix],
        req_pool_indices=torch.tensor([1], device=device),
        out_cache_loc=mapping[0, prefix:],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    decode_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False),
    )
    query = (
        (torch.randn(chunk, 16, 512, generator=generator) * 0.125).to(dtype).to(device)
    )
    query_rope = query.new_zeros(chunk, 16, 64)
    zero_rope = query.new_zeros(backing.pages, 128, 1, 64)
    selected = torch.linspace(0, prefix - 1, 2048).long()
    indices = (
        selected.to(dtype=torch.int32, device=device)
        .view(1, 1, -1)
        .expand(chunk, 1, -1)
        .contiguous()
    )
    table = torch.arange(1, sequence // 128 + 1, device=device, dtype=torch.int32).view(
        1, -1
    )
    host_table = backing.page_binding.host_table(table)
    assert not torch.equal(host_table, table)
    qlength = torch.tensor([chunk], dtype=torch.int32, device=device)
    klength = torch.tensor([sequence], dtype=torch.int32, device=device)
    writes = [
        (source_cpu[prefix:] + layer * 0.03125).to(device).unsqueeze(1)
        for layer in range(4)
    ]
    no_rope = query.new_empty(chunk, 1, 0)
    outputs = []
    runtime = None
    state = None
    graph = None

    def execute():
        result = []
        with runtime.forward(batch):
            for layer in range(4):
                pool.set_kv_buffer(
                    SimpleNamespace(layer_id=layer),
                    batch.out_cache_loc,
                    writes[layer],
                    no_rope,
                )
                key, _ = pool.get_kv_buffer(layer)
                runtime.on_compute_start(layer)
                output, _, _ = torch_npu.npu_sparse_flash_attention(
                    query=query,
                    key=key,
                    value=key,
                    query_rope=query_rope,
                    key_rope=zero_rope,
                    sparse_indices=indices,
                    block_table=table,
                    actual_seq_lengths_query=qlength,
                    actual_seq_lengths_kv=klength,
                    scale_value=0.0625,
                    sparse_block_size=1,
                    layout_query="TND",
                    layout_kv="PA_BSND",
                    sparse_mode=3,
                    attention_mode=2,
                    return_softmax_lse=False,
                )
                runtime.finish_layer(layer)
                result.append(output)
        return result

    records = {}
    try:
        for h2d, d2h in (
            modes
            if modes is not None
            else ((False, False), (True, False), (False, True), (True, True))
        ):
            if runtime is not None:
                torch.npu.synchronize()
            for layer in range(4):
                backing.regions[layer][0].tensor[host_rows[prefix:]] = float("nan")
            runtime = PipelinedIntegratedOffloadRuntime(
                pool, request_pool, h2d_prefetch=h2d, d2h_async=d2h
            )
            pool.prefill_offload_runtime = runtime
            execute()
            torch.npu.synchronize()
            samples = []
            for _ in range(repeats):
                start, end = (
                    torch.npu.Event(enable_timing=True),
                    torch.npu.Event(enable_timing=True),
                )
                start.record()
                output = execute()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            mode = f"{int(h2d)}{int(d2h)}"
            records[mode] = {
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
            outputs.append([item.cpu() for item in output])
            for layer in range(4):
                torch.testing.assert_close(
                    backing.regions[layer][0].tensor[host_rows[prefix:]],
                    source_cpu[prefix:] + layer * 0.03125,
                    rtol=0,
                    atol=0,
                )
            if profile_dir:
                destination = str(Path(profile_dir) / mode)
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        destination
                    ),
                ):
                    execute()
                    torch.npu.synchronize()
        for result in outputs[1:]:
            for actual, expected in zip(result, outputs[0]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        selected_keys = source_cpu[selected].float()
        logits = query[:2].cpu().float() @ selected_keys.T * 0.0625
        expected = torch.softmax(logits, dim=-1) @ selected_keys
        torch.testing.assert_close(
            outputs[0][0][:2].float(), expected, rtol=0.03, atol=0.002
        )

        state = FusedSparseAttentionState(
            device, request_pool, external_planner=True, resident_capacity=4096
        )
        host_key, host_rope = backing.layer_views(3)
        host_key, host_rope = host_key.squeeze(2), host_rope.squeeze(2)
        decode_query = query[:1].contiguous()
        decode_indices = indices[:1, 0].contiguous()
        request_ids = torch.tensor([1], dtype=torch.int64, device=device)
        decode_qlength = torch.tensor([1], dtype=torch.int32, device=device)

        def decode():
            inputs = state.build_inputs(
                layer_id=3,
                q_nope=decode_query,
                q_rope=decode_query.new_empty(1, 16, 0),
                topk_indices=decode_indices,
                full_kv=host_key,
                full_rope=host_rope,
                block_table=host_table,
                kv_lengths=klength,
                query_lengths=decode_qlength,
                request_slots=request_ids,
                scale=0.0625,
            )
            return run_fused_sparse_attention(
                replace(inputs, full_kv_region=backing.regions[3][0])
            )

        execute()
        with runtime.forward(decode_batch):
            decoded = decode()
        torch.npu.synchronize()
        torch.testing.assert_close(
            decoded.cpu().float(), outputs[-1][3][:1].float(), atol=0.002, rtol=0.03
        )
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            decoded = decode()
        execute()
        with runtime.forward(decode_batch):
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            decoded.cpu().float(), outputs[-1][3][:1].float(), atol=0.002, rtol=0.03
        )
        digest = hashlib.sha256()
        for output in outputs[0]:
            digest.update(output.contiguous().view(torch.uint8).numpy().tobytes())
        copy_evidence = {
            "mapping": "HiCache independent Host/device pages",
            "host_page_ids": host_table.cpu().tolist(),
            "device_page_ids": table.cpu().tolist(),
            "native_extension": os.environ.get("SGLANG_NPU_FSA_EXTENSION"),
        }
        return {
            "prefix": prefix,
            "chunk": chunk,
            "layers": 4,
            "modes": records,
            "output_sha256": digest.hexdigest(),
            "copy_evidence": copy_evidence,
        }
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        if runtime is not None:
            runtime.drain()
        if state is not None:
            for planner in state.planners.values():
                planner.close(drained=True)
        backing.pool.close(drained=True)


def test_nope_sfa_prefill_and_fused_decode_transition():
    prefix = int(os.environ.get("SGLANG_FSA_TEST_PREFIX", "8192"))
    chunk = int(os.environ.get("SGLANG_FSA_TEST_CHUNK", "128"))
    repeats = int(os.environ.get("SGLANG_FSA_TEST_REPEATS", "3"))
    result = exercise_sfa_pipeline(
        prefix, chunk, repeats, profile_dir=os.environ.get("SGLANG_FSA_TEST_TRACE")
    )
    destination = (
        Path(os.environ["SGLANG_FSA_WORK_DIR"]) / f"sfa_overlap_{prefix}_{chunk}.json"
    )
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
