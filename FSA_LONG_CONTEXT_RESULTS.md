# GLM-5.3 Flash single-service TP8 long-context validation

2026-09-15, host141.61.55.34, container `jzx_sglang_5.3_flash`.
All artifacts are under `/home/j00628475/fsa_glm53_pd_20260915`.
One service executes both Prefill and Decode, with TP8/DP1, MTP disabled,
FP8 model weights and BF16 KV, 8192-token Prefill chunks, one request,
1048576-token context and KV capacity, 4096-token resident selection cache.
Prefix/radix cache is disabled. Every performance request uses an exact
`input_ids` count and generates128 tokens at temperature0, with `ignore_eos`.
These are runtime/performance checks, not a900K model-accuracy benchmark.

## Measured offload performance

Additional completion test: exact1,000,000 input tokens plus1,024 output tokens
completed on the restored offload service, without restart or profiling.
`bench_1m_1024_v1` exited0; TTFT328.791797s, TPOT54.656021ms,
whole request384.705015s, streamed interval P50=53.386640ms. The first event
contained one output token; TPOT is `(last-first)/1023`. Server metadata confirms
prompt_tokens1000000, completion_tokens1024, finish_reason=length1024,
num_retractions0 and cached_tokens0. Decode logs show Graph replay throughout.
Sampling uses temperature0 and ignore_eos=True to force the requested output
length. This is a single synthetic completion/performance test, not model accuracy
qualification. The configured total capacity remains1048576; the request's total
token count is1001024. Service150593 on33195 remains running.

| Prompt tokens | Run | Profiling | TTFT (s) | TPOT (ms) | Request duration (s) |
| ---: | --- | --- | ---: | ---: | ---: |
| 500000 | first | off | 116.0382 | 51.2016 | 122.5410 |
| 500000 | repeat | off | 111.7450 | 49.2847 | 118.0043 |
| 900000 | diagnostic | on for two Prefill/two Decode steps | 293.7398 | 171.3612 | 315.5028 |
| 900000 | repeat1 | off | 272.1084 | 52.1008 | 278.7253 |
| 900000 | repeat2 | off | 272.4442 | 52.2300 | 279.0775 |

Both benchmark drivers exited0. Each request reports exactly the requested
prompt length,128 completion tokens, `num_retractions=0`, `cached_tokens=0`
and length completion. TTFT is client time until the first streamed token.
TPOT is `(last_token_time-first_token_time)/127`; every first event contains
one token. No health requests were issued during timed runs. Profiling's trace
export pauses contaminate diagnostic TTFT/TPOT; exclude that row from performance
comparisons. Memory sampling runs in a separate CPU thread, with npu-smi query
time plus5s between samples; its peaks are sampled observations, not allocator
high-water marks.

## Actual path evidence

`bench_900k_v2/profile` contains real NPU device task records from the actual
900K request. Rank0 summaries are in `trace_kernel_evidence.json`:

- Two Prefill chunks:22 `SparseFlashAttention` calls,33 `unidex_copy` calls.
  Eleven DSA layers each publish new KV twice; prefix restoration adds11 copies
  in the second chunk. Other TP lanes show11 restoration copies, while the TP0
  writer publishes the Host data. This matches the single-writer shared backing.
- Two Decode steps:22 `FusedSparseAttentionOverlap` device kernels on MIX_AIC,
  query shape `[1,8,512]`,2051 indices,32x128 resident selection slots. Rank0
  duration range147.632..351.409us, mean244.178us. These are diagnostic kernel
  durations, not all-layer TPOT.
- The service logs record `npu graph: True` during Decode at500K/900K.
- `offload_1m_v2/host_shared_segments.json`:11 logical Host KV segments of
  1073872896 bytes, each created by TP0 and attached by8 ranks, total11.0013GiB.
  This proves actual Host sharing at the1M capacity, rather than eight Host
  copies. Selection state, indexer/KPool and KDA remain per-rank NPU state.

## Observed offload memory

`server_info` reports the same1M token capacity and per-rank weights37.979GiB,
KV4.751GiB, Decode Graph0.865GiB. The KV figure includes two full-context Prefill
HBM slots plus all eleven resident indexer caches; it excludes shared Host KV.

| Workload | Sampled total HBM peak,8 devices (GiB) | Sampled scheduler PSS sum (GiB) |
| --- | ---: | ---: |
| 500K repeat | 430.267 | 29.781 |
| 900K repeat1 | 430.712 | 31.391 |
| 900K repeat2 | 430.710 | 31.394 |

Device0 includes extra runtime allocation; compare matched device IDs. HBM
values are whole-device usage including model/runtime/cache, not just KV.
PSS apportions shared pages; summing rank RSS would count shared KV repeatedly.
Host KV is offloaded, so HBM reductions may accompany Host-memory increases.

## Resident comparison

Matched1M-capacity resident measurement completed on the same machine. The same launch script's
`resident` role disables integrated offload, Prefill offload, fused attention and
Host KV budgets. It uses the same model, TP, context/capacity, chunk size and
Graph settings. Request JSON payload SHA256 matches between offload and resident
for each length, including exact input IDs and sampling settings. Resident ran
one measured request per length; offload numbers below use the500K repeat and
the mean of two unprofiled900K repeats. This is a limited sequential comparison,
not a statistically qualified performance benchmark. Compilation history differs
between first and repeated requests; no profiling timings are included.

| Prompt | Offload TTFT (s) | Resident TTFT (s) | Offload TPOT (ms) | Resident TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| 500K | 111.745 | 95.888 | 49.285 | 38.566 |
| 900K | 272.276 | 217.784 | 52.165 | 41.462 |

Offload TTFT was16.54%/25.02% higher and TPOT27.79%/25.82% higher at500K/900K.
The measured benefit is lower HBM occupancy; these tests show no speed benefit
over keeping full KV on NPU. Both paths admitted1M capacity and completed both
workloads, so these results do not establish an OOM/maximum-context advantage.

| Workload | Offload sampled HBM peak,8 cards (GiB) | Resident (GiB) | Saved (GiB) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 500K | 430.267 | 502.064 | 71.798 | 14.30% |
| 900K | 430.712 | 502.065 | 71.354 | 14.21% |

Actual scheduler KV allocation: resident13.752GiB/rank versus offload4.751GiB/rank,
approximately9.001GiB/rank or65.45% reduction in KV allocation. The full-device
sampled reduction is about8.92..8.97GiB per card on average. Resident weights
37.978GiB/rank and Graph0.966GiB/rank; offload weights37.979 and Graph0.865GiB.
Whole-device numbers also include runtime buffers and allocator retention.
Both services reserve1M capacity even for a500K request, explaining why the HBM
peak is almost flat between these two input lengths.

Host cost increases: sampled scheduler PSS sum500K18.635GiB resident versus
29.781GiB offload;900K18.631 versus31.394GiB. Shared full KV itself occupies
11.001GiB once per TP group. Offload trades Host memory and copy/coordination
time for HBM savings; it does not reduce all tiers of memory simultaneously.

`bench_resident_v1` exited0, both lengths returned128 tokens with no retractions.
After this comparison, only the owned resident service was stopped and a new
offload service was launched as `offload_1m_restored_v1`, HTTP33195/dist29625.
Restoration completed with driver exit0 and health200 at23:03 CST; parent150593
(start_ticks2721714), TP8/MTP0,1M capacity and offload/fused enabled. The service
remains running. Use `http://127.0.0.1:33195` inside the34 container.

Artifacts: `bench_500k_v1`, `bench_900k_v2`, `offload_1m_v2`, `resident_1m_v3`,
`bench_resident_v1` and `long_context_comparison.json`. Earlier resident900K attempt was stopped before qualification
to match capacity to1M; no900K-capacity result belongs in the memory comparison.
