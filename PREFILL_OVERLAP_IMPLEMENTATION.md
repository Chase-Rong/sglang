# Integrated Prefill copy overlap

The optional runtime in `python/sglang/srt/hardware_backend/npu/prefill_overlap.py`
overlaps submitted prefix restores and current-chunk saves with the preceding or
current DSA attention window. It preserves the two physical global-token HBM slots
and the existing registered Host backing. No additional per-layer full-KV staging
is allocated. Actual device overlap and speed depend on kernel resource contention.

## Enable

```bash
export SGLANG_NPU_PREFILL_H2D_PREFETCH=1
export SGLANG_NPU_PREFILL_D2H_ASYNC=1
```

Both switches default to false. With both false, the original
`IntegratedOffloadRuntime` remains selected. They can be enabled independently for
matched `00`, `10`, `01`, `11` experiments, without changing chunk size or Graph
coverage. Requires integrated layerwise offload, ordinary eager Prefill, MTP0,
CP1/DCP1 and no mixed Prefill/Decode batches. Prefill Graph is excluded; Decode
Graph remains supported. With only one physical slot, H2D lookahead is disabled.
The container must already provide the validated registered-Host copy extension
and A5 fused-attention OPP; this change requires no wheel or native rebuild.

## Ordering and ownership

1. Forward entry snapshots request prefix and new-row descriptors and orders them
   after the preceding Host-visible generation. First DSA prefix restore starts.
2. `prepare_layer` waits for its restore before scatter or slot reads.
3. After the actual latent and optional RoPE scatter, `on_kv_written` queues D2H
   from the physical slot at global row locations. Only the shared Host writer
   copies; other ranks still follow the publication protocol.
4. `on_compute_start(A)` queues restore B; compute proceeds with attention A.
   Reusing A's slot for C waits for both attention A and D2H A completion.
5. Forward tail joins saves and publishes each nonempty component in fixed layer
   order on the compute stream. Copy streams never submit HCCL collectives.
6. The resulting Host-visible event gates the next chunk and Decode. The runtime
   retains tensor/event references, with at most two retired generations before
   applying completion backpressure. Request release drains owned streams.

The hook begins after Indexer output preparation. It does not claim overlap with
the preceding Indexer. Empty prefix/write generations retain control events.
Missing/duplicate/out-of-order hooks fail explicitly; failures drain submitted
work and prohibit runtime reuse. An undrainable device error propagates rather
than returning memory for reuse.

## Verification as of 2026-09-17

- Focused CPU dependency, ownership, configuration and existing Host tests:116 passed.
- NPU single-card lifecycle:16 passed; profiler-only test skipped by default.
- TP2 and TP8:7 shared-Host tests passed per rank, including successive generations
  and Decode current-write Graph replay. A CPU test-phase race was repaired by
  joining checks before allowing the subsequent Decode test to mutate shared rows.
- Actual NoPE SFA:all four Prefill modes bitwise match; CPU oracle matches within
  BF16 tolerance; subsequent fused Decode eager and Graph results match SFA.
- Actual TP8 GLM-5.3 Flash service with both switches enabled:10-token and
  10,417-token prompts each generated64 tokens, no retractions or request abort.
  The latter used multiple Prefill chunks and then Decode Graph. First readiness
  was delayed by KDA Triton compilation, confirmed from live stacks. These cold
  smoke timings are not a matched E2E performance comparison. Only this test's
  PID/start-time-verified process group was stopped after results were saved.
- On idle141.61.55.34, 4 DSA layers, BF16,16 query heads, topk2048, prefix131072,
  new chunk8192, seven timed iterations (profiling excluded from timings):

| H2D prefetch | D2H async | Median component time | Reduction vs00 |
|---|---|---:|---:|
| off | off |116.8878 ms| baseline |
| on | off |114.9438 ms|1.66%|
| off | on |115.5642 ms|1.13%|
| on | on |115.2565 ms|1.40%|

This is a four-layer component measurement, not full-model TTFT/TPOT. Every query
uses the same sampled2048 history rows; production Indexer and other model layers
are absent. The first restore and final save remain exposed. A short8K/128 case
also passes, but its three-iteration timing is a smoke measurement only.

The trace shows task intervals overlap, yet H2D increases from about10.29 ms alone
to28.5 ms alongside an18.42 ms SFA; D2H increases from0.35 ms to18.7 ms.
These timings are consistent with substantial AIV scheduling/resource contention;
overlapping task intervals do not establish useful simultaneous copy progress.
Do not claim full H2D hiding or a corresponding E2E speedup. Keep opt-in until
matched model-level performance qualification; kernel scheduling/resource tuning is separate.

Artifacts: `results/prefill_overlap_20260917/` contains source hashes, device-idle
checks, CPU-independent hardware logs and kernel CSVs. Remote isolated source and
trace root:`/home/j00628475/prefill_overlap_20260917_v1` in
`jzx_sglang_5.3_flash`. Shared production source was not overwritten.

## Tests

```bash
python3 -m pytest test/registered/unit/npu/test_prefill_overlap.py -q
SGLANG_FSA_TEST_ATTENTION=1 SGLANG_FSA_TEST_DEVICE_ID=0 \
  python3 -m pytest test/registered/unit/npu/test_prefill_overlap_npu.py -q
SGLANG_FSA_TEST_SHARED_TP=1 SGLANG_FSA_TEST_DEVICES=0,1 \
  torchrun --nproc-per-node=2 --master-port=29847 -m pytest \
  test/registered/unit/npu/test_prefill_overlap_tp_npu.py -q
SGLANG_FSA_TEST_ATTENTION=1 SGLANG_FSA_TEST_DEVICE_ID=0 \
  SGLANG_FSA_TEST_PREFIX=131072 SGLANG_FSA_TEST_CHUNK=8192 \
  SGLANG_FSA_TEST_REPEATS=7 python3 -m pytest \
  test/registered/unit/npu/test_prefill_overlap_sfa_npu.py -q -s
```

Use the existing `scripts/npu/fsa_offload_env.sh integrated` setup and qualified
native/OPP paths. Verify selected cards are idle immediately before launching.
Never stop unrelated services. Profiling is opt-in through
`SGLANG_FSA_TEST_TRACE=/home/j00628475/<isolated-result-directory>`.
