# Integrated TP8 DSA memory allocation audit

The opening sections describe the initial audit snapshot. Dated sections below
record subsequent implementation and validation; they supersede the initial
service status and the statement that joint redistribution is not implemented.

Initial scope: local sglang_remote and34 isolated runtime under
`/home/j00628475/fsa_glm53_pd_20260915/sglang`. Five core allocation/budget/runtime
files were SHA256-compared and match. Live service33195 reports1M capacity,
max_running_requests1, disabled radix cache, enable_unified_memoryFalse,
and KV4.751GiB/rank. No service restart or code behavior change for this audit.

## DSA full KV is offloaded, but DSA HBM is not zero

NPUMLATokenToKVPool allocates logical full KV through DecodeHostKvBacking;
RegisteredHostRegion views alias Host storage, not copied HBM tensors. Integrated
mode also allocates physical Prefill slots with torch.zeros on NPU:
`physical_layers = prefill_layer_slots or layer_num`. It does not allocate one
full NPU KV per logical DSA layer, but it retains two full-capacity physical slots.
Each slot scales with `size + page_size`, not the8192-token Prefill chunk.
IntegratedOffloadRuntime switches which layer uses a slot and restores prefixes;
it never deletes those tensors after Prefill. Returning Host views during Decode
does not release the HBM referenced by pool.k_buffer/v_buffer.

For11 DSA layers,1048576 capacity,page128,latent512,RoPE0,BF16:

| Storage per rank unless noted | Resident | Integrated offload |
| --- | ---: | ---: |
| Full-context latent KV on NPU | 11.001343GiB | 2.000244GiB in two Prefill slots |
| Resident indexer cache | 2.750336GiB | 2.750336GiB |
| Reported KV total, rounded | 13.752GiB | 4.751GiB |
| Full latent KV on Host, once per TP group | none | 11.001343GiB |

Additional DSA HBM includes4096-token selection caches per layer/request row,
planner/status/maps, KPool tails and runtime/Graph workspace. These do not fit
inside the4.751GiB KV number; the selection/planner budget is accounted separately.
Thus the implementation removes persistent full KV replicas for all logical
layers but does not completely eliminate DSA HBM or all DSA preallocation.

## Budget accounting matches the reduced allocation

DecodeKvMemoryBudget separates:

- Host/token: `layers * (latent_dim + rope_dim) * element_bytes`.
- NPU/token: `(layers * index_dim + prefill_layer_slots * (latent_dim + rope_dim)) * element_bytes`.
- Fixed Host/NPU costs: selection, planner, maps, graph row capacity and optional staging.

DefaultPoolConfigurator uses this reduced NPU/token cost when Host offload is
enabled and solves capacity against both Host and NPU budgets. The old eleven
full-latent-KV cost is not still reserved as a shadow HBM buffer or charged a
second time to the NPU budget. Prior matched measurements agree with this:
approximately9GiB/rank less KV allocation and71..72GiB less sampled whole-device
HBM across8 cards, at equal1M capacity.

## Remaining HBM is reusable, but redistribution is not automatic

Avoided allocation leaves physical HBM available for subsequent allocations.
This is startup allocation avoidance, not runtime destruction of an already
allocated full cache. Existing live Prefill slots and selection buffers cannot
be reclaimed with empty_cache while tensors/Graph users still reference them.

In current GLM5.3 Flash, the other layer type is KDA, whose recurrent/conv states
are allocated through HybridReqToTokenPool/MambaPool. This is per-request state,
not a full-context K/V array growing linearly with sequence length.

Allocation sequence is `_profile_available_bytes -> _handle_max_mamba_cache ->
config_from_budget -> DSA capacity solver -> _apply_token_constraints -> pool
creation`. With disabled radix and max_running_requests1, the mamba state pool
uses the explicit request-count branch; actual buffers may include padding and
extra-state slots. Raising max_mamba_cache_size explicitly can reserve a larger
pool, while raising max_running_requests also raises state/selection demand.
The general ratio branch sizes KDA before the DSA capacity solve and does not
revisit it after DSA consumes less memory.

The1M max_total_tokens cap can leave DSA budget unused. Current code contains no
second solve that transfers this remainder into more KDA slots, no automatic
increase of the explicit request limit, and no runtime shared allocator between
the two pools. Unified-memory allocation is off in this service. Post-capture KV
resize is CUDA-gated (`current_platform.is_cuda()`), so it is not an NPU KDA
redistribution mechanism.

Therefore: other KV/state allocations can use the saved HBM when configured and
budgeted at startup, but the current service does not automatically devote the
savings to a larger KDA pool. To implement that policy, cap DSA capacity first,
subtract its reduced storage plus selection/Graph/activation reserves, then solve
KDA/request capacity against the remainder with target/draft/TP consistency.
This requires validation and is not already implemented or measured.

Removing the remaining full-context Prefill slots would require streaming/tiled
Prefill attention or another Host-aware Prefill implementation. Deleting them
or setting slot count0 is incompatible with the present full-cache SFA Prefill
path. Some NPU working memory and selection storage remain necessary even then.

## 2026-09-16 - Joint solver call-chain correction and validation

The initial reservation hook was ineffective: `_decode_budget` belongs to
`DefaultPoolConfigurator`, not `KVCacheConfigurator`. At profiling time the old
hook returned without reserving anything. A Mamba allocation log alone is not
evidence that the joint path executed.

The temporary fixed-only reservation is rejected. The resident Indexer and both
full-capacity Prefill slots are real HBM allocations. `required_bytes()[1]`
already excludes the Host full-latent backing; its token-linear term must remain.
Likewise selection storage really scales with query rows; removing that factor
would undercount the actual `FusedSparseAttentionCacheManager._allocate` buffers.

The corrected production entry creates the pool configurator before KDA sizing,
fixes the page-aligned DSA token target, and solves KDA slots and request rows
jointly. Each candidate charges DSA fixed and linear bytes, KDA padding, the
ReplaySSM ring and speculative intermediate state. Explicit request limits are
not enlarged; explicit state-slot limits must fit. Automatic requests grow only
when both DSA row storage and KDA state fit. The remaining DSA budget is returned
without subtracting the DSA reservation twice, and the downstream token solve is
pinned to the agreed target.

Local production-method/solver tests: 38 passed. The complete isolated snapshot
on 141.61.54.248 / jzx_sglang_5.3_flash additionally passed real runtime-config
tests: 40 passed, 14 dependency warnings, 16.11 seconds. These are control tests,
not full-model or NPU-capacity evidence.

The fresh E2E source is `/home/j00628475/joint_budget_20260916_v2/source` and the
run directory is `/home/j00628475/joint_budget_20260916_v2/run`. It uses the
validated FP8 model and existing native-build-v5/attention-opp-v8, TP8 integrated
P/D, 1048576 token capacity, eight explicit requests, disabled radix, port33226
and distributed port29726. Its runner records PID/start ticks and persistent
logs. At this checkpoint it is loading; success and capacity redistribution
remain unverified. Earlier ad-hoc BF16 launches with incomplete environment
variables and no 1M CLI cap are not matched evidence for this feature.

Live follow-up: service PID33661/start_ticks3915269 and runner PID33410 remain
alive on248. Py-spy verified the owned TP workers: TP1/5/6/7 are in
`torch_npu.npu.npu_config.set_compile_mode -> synchronize` during module import;
TP0 waits in the TCPStore rendezvous; TP2/3/4 wait in process-group creation.
The log also reports null NPU context errors. This is before model weight loading
and before the new joint solver. No restart was performed after this observation.
The runner is still observed via local exec session65791. An expired health
probe or missing local observation handle must not be treated as process exit;
revalidate the recorded PID/start ticks first.

## 2026-09-16 10:25 CST - Real class and explicit-eight E2E qualification

The v3 full-model launch on34 passed HCCL and exposed a missing dataclass slot:
assigning `_joint_capacity` failed because KVCacheConfigurator uses slots=True.
The field is now declared; the unused duplicate target/reservation fields were
removed. A new remote test imports the real class and executes its joint method.
Another test covers selection row reservation when requests exceed Graph max BS.
The solver now also charges the padded request-token table and per-row metadata.
FSA initialization reserves max(request capacity, Graph capacity) query rows,
including MTP expansion, before Graph locks cache allocation.

Local39 tests passed; remote34 actual-class/runtime tests44 passed in13.45s.
Full immutable source snapshot: `/home/j00628475/joint_budget_20260916_v4/source`.
Explicit8 E2E on34, run `explicit8`, port33246, reached ready and allocated:

- target tokens1048576, requests/state slots8;
- DSA budget5.102GiB/rank (KV4.751GiB plus fixed selection/planner);
- KDA0.155GiB/rank; request table0.035GiB/rank;
- unused static budget31.909GiB/rank, retained because request limit is explicit.

The matched budget-off run `baseline8`, port33247, also reached ready. The same
single request plus8 concurrent chat requests returned identical text in9/9
responses. Eight responses obeyed the number-only prompt; request7393 produced
the same reasoning/exclamation text and hit max_tokens64 in BOTH runs. Therefore
the semantic smoke as a whole failed, but this exact behavior is not introduced
by the joint-budget toggle. Preserve both outputs; do not label all semantic
cases passed. Server logs prove batch8 with npu graph=False and no memory error.
Both owned services were stopped using recorded PID/start_ticks and process-group
identity; no other services were stopped.

Collected logs/JSON are under local `results/joint_budget/141.61.55.34/`.
Automatic capacity tests are now in progress:34 `baseline_auto` with joint=0,
port33249/dist29749, and248 `automatic` with joint=1, port33248/dist29748.
248 passed HCCL this time but is loading checkpoint shards slowly. Live PIDs
must be refreshed from each manifest before any stop/restart. Auto capacity and
long-context results remain pending at this checkpoint.

## 2026-09-16 10:39 CST - Matched automatic-capacity allocation

On34 the same v4 source and FP8/TP8/MTP0 settings now have a matched toggle
comparison. `baseline_auto` (joint=0, HTTP33249) reached health200 but actual
capacity was only768 tokens,384 requests and1024 KDA slots. The configured
context length1048576 did not guarantee an actual1M token pool. Logs were
collected before stopping its owned PID177851/start_ticks6843729.

`automatic` (joint=1, HTTP33250, dist29750) reached health200 with1048576 actual
tokens,499 requests and499 KDA slots. PID182802/start_ticks6887610. All ranks
report DSA26.654GiB, KDA8.593GiB and request-table1.953GiB within the static
budget. DSA includes the real4.751GiB KV allocations plus499 rows of selection
and planner state. The solver prioritizes the token target while balancing
request and KDA costs; it does not merely maximize KDA at the expense of DSA.
This establishes allocation/capacity behavior, not499-request throughput.

On248 the same source computes498 requests/slots and1048576 tokens, with
DSA26.610GiB, KDA8.575GiB, request1.949GiB, unused0.057GiB. Initial health is
still pending. Live py-spy at10:38 shows KDA Prefill `recompute_w_u_fwd` Triton
compilation/autotuning across ranks, not FSA deadlock. Do not restart solely for
these health-probe timeouts. Concurrent-generation and long-context validation
are still pending for both automatic configurations.

## 2026-09-16 10:46 CST - Automatic capacity generation and first long case

Both automatic services reached health200 and completed one short request plus
eight concurrent requests. No OOM was observed. Strict number-only semantics
passed8/9 on each host:34 produced reasoning for7394 and248 for7393. Do not label
these smoke suites fully passed or claim exact automatic-vs-explicit equivalence.
The earlier explicit8 toggle comparison retains its separate9/9 text-equivalence
result, including a shared semantic failure.

On34 the unprofiled500000-input/1024-output case completed with0 retractions:
TTFT110.390752s, TPOT52.608852ms, total164.209735s. This is one measured run,
not a throughput comparison against the768-token old automatic configuration.
900K and1M cases are continuing sequentially in `automatic/long_context`.

On248 a separate16384-input/32-output diagnostic trace completed. Its timings
include profiling and must not enter the performance table. Device-kernel
evidence extraction is pending. Artifact path: `automatic/route_trace`.

Follow-up:248 TP0 trace extraction confirms33 `unidex_copy` and22
`SparseFlashAttention` calls across two Prefill chunks, then22
`FusedSparseAttentionOverlap` calls across two Decode steps on MIX_AIC.
The diagnostic fused durations range146.157..290.029us (mean212.718us);
these are not TPOT numbers. Collected evidence is in local
`results/joint_budget/141.61.54.248/joint_budget_20260916_v4/automatic/route_trace/trace_kernel_evidence.json`.

The automatic configuration intentionally consumes more HBM than explicit8:
34 server_info startup_available44.337GiB becomes12.438GiB, while reported KV
remains4.751GiB. `graph.decode` rises1.296 to22.854GiB because FSA selection
allocation occurs during Graph reservation; this field is not pure Graph
workspace. This change trades free HBM for499 admitted request rows while
preserving1M token capacity. It is not a new HBM-saving claim at unequal request
capacity, nor proof of499 simultaneous1M requests (the token pool is shared).

## 2026-09-16 10:51 CST - 900K completion and remaining1M case

34 unprofiled900000-input/1024-output completed with0 retractions:
TTFT263.804625s, TPOT55.883534ms. The1M case is running in the same service,
with the same automatic499-request capacity and no intervening restart.
All timing records, streaming events and sampled memory are collected under
`results/joint_budget/141.61.55.34/joint_budget_20260916_v4/automatic/long_context`.

248 diagnostics and outputs were collected, and its owned service37706 was
stopped after PID/start_ticks and process-group validation. A subsequent process
listing confirmed no remaining service/scheduler workers in that container.
Its manifest's `ready` value is historical and must not be read as live status.

## 2026-09-16 10:57 CST - Joint-budget long-context qualification completed

34 `automatic/long_context` exited0. Same integrated TP8/MTP0 service, FP8
checkpoint, BF16 KV, static fraction0.85, disabled radix, token pool1048576,
request/state capacity499. Each case ran once without profiling and returned
exactly1024 tokens with0 retractions. `ignore_eos=True` forced the requested
output length; this is execution/capacity qualification, not a long-form quality
benchmark or maximum-concurrency throughput test.

| Input tokens | Output tokens | TTFT seconds | TPOT milliseconds |
| --- | --- | --- | --- |
| 500000 | 1024 | 110.390752 | 52.608852 |
| 900000 | 1024 | 263.804625 | 55.883534 |
| 1000000 | 1024 | 313.558035 | 56.654331 |

The matched automatic-budget result on34 is now: old budget token capacity768,
request capacity384, KDA slots1024; joint budget token capacity1048576, request
capacity499, KDA slots499. This is a joint feasibility improvement, not an
increase in every individual pool. Fixed explicit8 still stays at8, and its
on/off response comparison is unchanged. Real concurrent generation covered8
requests;499 is the configured capacity, not a measured499-request workload.

44 control tests passed in the preceding v4 qualification, including the real
slotted KVCacheConfigurator. Remaining boundaries: automatic semantic smoke is
8/9 on both tested hosts, no all-pass model-quality claim; MTP/ring arithmetic
has unit coverage but MTP E2E is not validated here. 248 separately proves the
actual offload/FSA kernels for the same production snapshot. Profiling timings
are excluded from the table above. No NPU kernel or installed wheel rebuild was
needed for this Python allocation change.

Live at completion:34 HTTP33250, parent182802/start_ticks6887610, source
`/home/j00628475/joint_budget_20260916_v4/source`;248 owned service is stopped.
All artifacts stay under the assigned `/home/j00628475` task directory.
Local results: `results/joint_budget/<host>/joint_budget_20260916_v4/`.
`scripts/npu/summarize_joint_budget_results.py` regenerates each collected
automatic run's `qualification_summary.json`, including sampled memory peaks.

Enable through `SGLANG_NPU_ENABLE_JOINT_KV_BUDGET=1` (integrated launcher default;
the framework environment default remainsFalse). Use
`FSA_MAX_RUNNING_REQUESTS=auto` to solve request capacity, or an explicit limit
to preserve that cap. The solver accounts for physical Prefill slots, resident
Indexer, selection/planner/Graph row state, padded KDA state and request tables;
Host full latent KV is not charged as HBM. Runtime/activation headroom remains
outside the static fraction. Deploy the production snapshot as a unit; only
changing the environment variable does not install the new solver/runtime code.

## 2026-09-16 - Final requirement audit and semantic smoke completion

Fresh completion audit compared local and34 v4 SHA256 for four production files
and all three targeted UT modules. All seven match. The remote container rerun
passed44 tests in11.76s, with14 dependency deprecation warnings. Receipts:
`results/joint_budget/141.61.55.34/joint_budget_20260916_v4/control_source_sha256.json`
and `control_result.txt`.

The earlier semantic failures all ended with `finish_reason=length` and exactly
64 completion tokens. They were preserved. The same automatic499 service was
retested with the same prompts/temperature/concurrency/strict final-number
assertion and a512-token generation cap. No production code or service restart
was involved. All9 answers now match7391..7399 and all9 end with `stop`.
Separate `joint_extended_request_*`, `joint_extended_response_*` and
`joint_extended_generation_summary.json` retain this evidence. This closes the
finite-output-budget smoke issue, without claiming broad model accuracy.

| Requirement | Implementation and authoritative evidence | Result |
| --- | --- | --- |
| Charge real DSA HBM before KDA sizing | `DecodeKvMemoryBudget.required_bytes`; physical Prefill slots, resident Indexer and row-dependent selection included; Host full latent KV excluded; UT and actual4.751GiB KV allocation | Pass |
| Jointly solve KDA/request capacity within remaining budget | `solve_joint_dsa_kda_capacity` in the real `_handle_max_mamba_cache` call chain; padding, ring, intermediate state and request tables charged; eight TP rank budget logs agree | Pass |
| Preserve requested token capacity and explicit limits | Joint `config_from_budget` pins1048576; explicit8 stays8; insufficient-memory UT rejects instead of silently shrinking target | Pass |
| Use released HBM for usable state/request capacity | Matched34 automatic old768-token/384-request allocation versus new1048576-token/499-request allocation; real `server_info` and allocation logs | Pass |
| Retain old path when disabled | Disabled/non-Decode UT; explicit8 on/off outputs identical9/9; baseline_auto launch uses joint0 | Pass |
| Graph and eager request rows remain usable | Real-class and reservation UT; actual Graph BS1 and eager BS8 generations complete | Pass |
| Unit validation | Source-matched remote44 passed, fresh saved stdout | Pass |
| Integrated TP8 end-to-end validation | 500K/900K/1M each returns1024 tokens, no retractions; separate actual offload/FSA device trace;9/9 semantic smoke with adequate output cap | Pass |

The requested integrated-service joint-budget capability is qualified. The
published limits remain: MTP0 for full-model E2E, real concurrency tested8,
one measured run per long-context length, and no throughput claim for499
simultaneous requests. Ring/MTP accounting has UT coverage; additional model
quality benchmarks and MTP E2E are separate qualifications.

## 2026-09-16 - Explicit request priority with a shared token pool

User correction supersedes the prior explicit-limit policy: an explicitly
configured `max-running-requests=N` is fixed, not a reducible upper bound for
the joint solver. The user explicitly chose SHARED POOL semantics: a single
request may use the whole available token pool subject to the model context
limit; the pool is not divided byN into per-request quotas.

`solve_joint_dsa_kda_tokens` now reserves KDA state, the padded request table,
and DSA selection/planner storage for exactlyN requests, then solves the largest
page-aligned token pool fitting BOTH HBM and Host budgets. It respects an
explicit max-total-tokens cap. Without that cap, at mostN model contexts are
provisioned. Insufficient fixed-state memory or conflicting explicit KDA slots
raises an error; it does not reduceN. `resolve_max_num_reqs` preserves the result
through downstream sizing. The existing worker single-request/input limits use
the shared token capacity with existing safety headroom, without dividing byN.
When no request count is specified, the previous automatic-request fallback is
retained. Disabling joint budgeting retains the original framework path.

Local solver/production-method tests55 passed. Source-matched remote248 tests
including real slotted configurator and real worker-info method62 passed in
15.76s,14 dependency deprecation warnings. Receipt and hashes:
`results/joint_budget/141.61.54.248/joint_fixed_requests_20260916_v1/control_*`.

All eight248 devices were idle before launch. Independent E2E snapshot:
`/home/j00628475/joint_fixed_requests_20260916_v1/source`, run `fixed512`,
HTTP33258/dist29758, TP8/MTP0 integrated, explicit512 requests and1048576 token
cap. This intentionally tests a requested concurrency above the earlier
auto498/499 result, expecting fixed512 with reduced tokens. Launch is in
progress; current allocation/generation evidence is still pending.34's previous
service/source were not changed. Native extension and OPP remain unchanged.

User subsequently paused E2E until remaining changes are consolidated. No
generation or performance test is authorized for this new snapshot at this
checkpoint. The owned248 fixed512 launch was sent SIGTERM only after manifest
PID/start-ticks and process-group verification.62 remote control tests remain
the current qualification for the request-priority change; no new E2E pass is
claimed. The earlier v4 E2E evidence describes the previous allocation policy.

## 2026-09-16 - Request upper bound with capacity fallback

The latest user decision supersedes the strict fixed-request policy above.
Follow original SGLang's upper-bound semantics: prefer configured concurrency,
reduce it with a warning when the memory/capacity constraints cannot support it,
and fail only when even one request cannot fit the configured minimum state.
Shared-pool semantics remain unchanged; single-request length is NOT divided
by the configured or effective concurrency.

The explicit-request solver searches for the largest feasible count at or below
the user's limit. Feasibility charges minimum paged token storage, all per-request
DSA/KDA/planner/request-table costs, Graph row floor, Host backing, speculative
state and native query-row limits. Explicit KDA slot capacity may cap the request
count; explicitly reserved extra KDA slots remain charged. Once concurrency is
resolved, remaining memory maximizes shared tokens under the configured token
cap. Later PP/token/KDA caps may further lower concurrency with a warning instead
of raising the previous fixed-count exception. The requested CLI value is retained;
effective capacity is published by the existing pool/worker path.

Configured maximum context remains an upper bound. When the requested number
of requests fits, a smaller token pool is preferred over reducing concurrency.
This does not promise the maximum model context at every requested concurrency,
or automatically alter invalid configuration values. The unset-request fallback
and disabled-joint behavior remain unchanged.

Local solver/production-method tests67 passed. Remote control verification uses
an isolated snapshot `/home/j00628475/joint_request_cap_20260916_v2/source` on248,
including a real slotted configurator warning test and the real worker-info
shared-pool test. Its result is pending at this checkpoint. No E2E launch,
generation request, service restart or NPU benchmark is performed for this change,
per the user's instruction to consolidate changes before E2E.

Remote control verification completed:75 passed in15.95s,14 dependency
deprecation warnings. All seven checked production/UT files match local SHA256.
Receipts: `results/joint_budget/141.61.54.248/joint_request_cap_20260916_v2/control_result.txt`
and `control_source_sha256.json`. Tests include exact HBM boundary reductions
8->7/3/1, Host and token-cap reductions, explicit KDA capacity, Graph floor,
MTP/query-row limits, warning/no-warning behavior and preserved single-request
access to the shared token pool. New policy has UT qualification only; unified
E2E remains deferred by user instruction.

## 2026-09-16 12:14 CST - Planned eight-request 1M qualification configuration

User requested recording this specification for later consolidated E2E. It is
a CAPACITY ESTIMATE, not a completed eight-request benchmark or throughput claim.
Do not launch services or run E2E until the user resumes that phase.

Reference hardware/software:34 A5,8 NPUs of96GiB each, container
`jzx_sglang_5.3_flash`, single integrated TP8/DP1/PP1 P/D service, MTP disabled,
FP8 checkpoint `/mnt/share/w00936111/weights/GLM-5.3-Flash`, BF16 KV, radix cache
disabled.248/244 remain alternate hosts, subject to idle-resource checks.
Use the latest request-upper-bound implementation; old v4 E2E results do not
qualify the current allocation policy.

| Setting | Planned value |
| --- | --- |
| Exact input length per request | 1000000 tokens, decimal1M |
| Output length per request | 1024 tokens |
| Client concurrent requests | 8 |
| `--max-running-requests` | 8; assert effective capacity remains8 |
| `--context-length` | 1048576 |
| `--max-total-tokens` | Omit the CLI argument entirely |
| `--mem-fraction-static` | 0.85 |
| `--page-size` | 128 |
| `--chunked-prefill-size`, `--max-prefill-tokens` | 8192 |
| `--cuda-graph-bs-decode` | Keep existing1/2/4 initially; BS8 is eager |
| `SGLANG_NPU_ENABLE_JOINT_KV_BUDGET` | 1 |
| `SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD` | 1 |
| `SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD` | 1 |
| `SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION` | 1 |
| `SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE` | fused |
| `SGLANG_NPU_SHARED_HOST_KV` | 1 |
| `SGLANG_NPU_FSA_RESIDENT_CAPACITY` | 4096 |
| `SGLANG_NPU_PREFILL_LAYER_SLOTS` | 2 |
| `SGLANG_NPU_DECODE_HOST_KV_BYTES` | 137438953472 (128GiB);96GiB is an alternative |
| `SGLANG_NPU_LAYER_STAGING_BYTES` | 0 for integrated mode |

Full latent Host KV is TP-shared;128GiB is the configured backing/planner budget,
not128GiB of full-KV copies on each TP rank. Physical Host free memory must also
cover model/runtime/planner allocations. Do not add the Prefill Host budget as
a second full-KV store: integrated mode uses the Decode backing for both phases.

Estimated boundaries using the previously observed37.20GiB/rank static pool
budget after weights/runtime reserve, with current DSA/KDA/request formulas:

| Concurrent1000000-input +1024-output requests | Required pool HBM per rank | Required Host KV/planner budget |
| --- | --- | --- |
| 6 | about27.62GiB | about63.02GiB |
| 7 | about32.22GiB | about73.53GiB |
| 8 | about36.82GiB | about84.03GiB |
| 9 | about41.42GiB, exceeds37.20GiB | about94.53GiB |

For8 request slots, the estimated shared token pool is about8092400 tokens.
Eight requests require8008192 logical tokens, or8008704 after individually
rounding to128-token pages. The remaining static budget at that required size
is only about0.38GiB/rank; the solver may consume it by enlarging the pool.
This is not0.38GiB of total device free memory: runtime headroom was reserved
separately. More Host capacity alone cannot make9 such requests fit HBM.
Input1048576 + output1024 is a different workload and exceeds the planned
context limit; do not substitute it for the exact1000000-token input.

Launcher prerequisite, not yet changed: `scripts/npu/fsa_pd_role.sh` currently
always emits `--max-total-tokens ${FSA_MAX_TOTAL_TOKENS:-32768}`. Merely unsetting
the environment variable supplies32768 rather than omitting the CLI option.
`remote_joint_budget.py` launch also explicitly supplies1048576. Before the
planned E2E, add a supported omit/auto option or use a separate reviewed launch
command. Preserve normal defaults and verify the final command/server_info.
Do not use the previous launcher unchanged and assume the cap was removed.

Later acceptance checks: inspect final requested/effective concurrency and pool
capacity, verify the actual offload/FSA route, confirm eight long requests are
simultaneously resident rather than queued serially, complete1024 output tokens
per request with no OOM/retraction, and record TTFT/TPOT, aggregate output tokens/s,
queue time and sampled NPU/Host memory. If larger Graph batches are enabled later,
re-budget and validate them separately. Non-offload throughput requires a matched
comparison; this estimate establishes no performance benefit.

## 2026-09-16 - Remove legacy automatic concurrency and resume E2E

User explicitly resumed E2E and requested removal of the former auto499 path and
its switches, superseding the prior pause. Audit found the old path still present;
it has now been removed from executable Python, scripts and active tests:
`solve_joint_dsa_kda_capacity`, unused `JointKvBudget`/reservation helpers and
`SGLANG_NPU_ENABLE_JOINT_KV_BUDGET`. Offload budgeting uses only the request-priority
solver, selected by the existing offload pool configuration. Non-offload pool
behavior is retained. Offload without explicit max-running-requests now raises a
configuration error; the launcher's default is an explicit1. Soft upper-bound
fallback and shared token-pool semantics remain as described above.

Launcher now supports `FSA_MAX_TOTAL_TOKENS=auto` to OMIT --max-total-tokens,
while explicit numeric/default values retain their behavior. Request-count auto
mode and the remote helper's joint toggle were removed. The earlier eight-request
specification's JOINT_KV_BUDGET flag is obsolete and must not be set as a gate.

Local focused tests46 passed after removing tests belonging only to deleted
algorithms. No executable references to the removed solvers or switch remain.
34 is currently idle across all8 NPUs; its container has no active service.
248/244 are occupied by unrelated workloads and were left untouched. New isolated
snapshot `/home/j00628475/request_budget_only_20260916_v1/source` on34 is being
prepared for control tests and integrated TP8 E2E. Planned run: requests8,
context1048576, total-token cap omitted, Host budget128GiB, MTP0. A concurrent
driver will send8 exact1M input requests with1024 output tokens and require
observed simultaneous Decode concurrency, not just eight queued HTTP requests.
Native extension/wheel unchanged. Close the owned test service after testing,
using PID/start-ticks validation, and confirm resources released.

New snapshot control qualification completed:54 passed in13.96s,14 dependency
warnings; both launch shell scripts pass bash syntax checks. Seven checked
production/UT hashes match local. The lower test count reflects deletion of old
solver-only tests, not dropped coverage of the retained request-priority path.
34 Host available memory was1417GiB. The service is starting as `requests8`,
HTTP33318/dist29818; its manifest records ownership. No other services were
stopped to obtain the idle34 environment. The previous34 container/service had
already been replaced externally; old PID records must not be reused.

The initial `requests8` attempt was stopped by preflight because HTTP33318 was
already in use. Its manifest has no owned service PID and no server was started.
Do not stop the port owner. The replacement run is `requests8_v2`, HTTP19186,
dist29186, same source/configuration; it rechecks all devices and ports before
launch. Retain both run directories as evidence.

## 2026-09-16 12:46 CST - Recorded 1M maximum-concurrency target and qualification boundary

The requested specification is eight concurrent requests, each with exactly
1000000 input tokens and1024 output tokens, on a single A5 TP8 integrated P/D
service (8 x96GiB NPU, DP1/PP1, MTP0, FP8 model, BF16 KV, radix disabled).
This is the expected memory-capacity ceiling for the recorded configuration,
not a completed eight-request E2E result or a throughput guarantee.

Use max-running-requests8, context-length1048576, mem-fraction-static0.85,
page-size128, chunked-prefill-size8192 and max-prefill-tokens8192. Omit
max-total-tokens; the current launcher supports FSA_MAX_TOTAL_TOKENS=auto for
this purpose. The token pool is shared, without per-request partitioning.
Retain Graph decode sizes1/2/4 initially (batch8 eager).

Enable integrated offload, Prefill layerwise offload, fused attention in fused
mode and TP-shared Host KV. Set resident capacity4096, Prefill layer slots2,
integrated layer staging0, and Decode Host KV budget137438953472 bytes (128GiB).
The obsolete SGLANG_NPU_ENABLE_JOINT_KV_BUDGET switch is no longer required;
do not reuse that row or the obsolete launcher warning from the12:14 entry.
The full Host latent KV backing is shared across TP ranks, not replicated8 times.

At the required eight-request paged capacity8008704, the formula estimates
36.82GiB pool HBM per rank and84.03GiB Host KV/planner budget. Nine such requests
need about41.42GiB HBM per rank, exceeding the observed37.20GiB pool budget;
raising only the Host budget therefore does not enable nine requests.

New local evidence reviewed from requests8_v2/server.log: the current solver
actually selected8091392 shared tokens and8 request/state slots on all TP ranks,
with37.005GiB DSA,0.155GiB KDA and0.035GiB request-table budget per rank.
However, startup then failed during Graph capture with:
`dst_rows * block_bytes exceeds the kernel uint32 address range`.
The larger full-KV tensor exceeds the current copy helper's4GiB address limit.
Thus allocation is confirmed, but eight concurrent1M generations remain
UNQUALIFIED. Resolve this limit before claiming the expected configuration can
serve the workload. Preserve the failure receipts under
`results/joint_budget/141.61.55.34/request_budget_only_20260916_v1/requests8_v2/`.

Acceptance still requires eight simultaneously resident long requests,1024
outputs each, no OOM/retraction, actual offload/FSA route proof, TTFT/TPOT,
aggregate throughput and sampled HBM/Host peaks. This record-only update does
not launch or modify a service.

## 2026-09-16 - Large-pool copy addressing fixed and hardware-qualified

Continued the active E2E goal. Fresh inspection confirmed34 idle on all8 NPUs;
248/244 remain occupied by other jobs and were untouched. The old automatic
request solver and JOINT_KV_BUDGET switch are absent from executable source,
scripts and active unit tests. The new isolated source snapshot is
`/home/j00628475/request_budget_copy64_20260916_v2/source`.
Its source-matched budget/configuration UTs passed54 cases in16.10s.

The previous startup error came from unidex_copy's32-bit byte extents/offsets,
not the request-priority solver. Changed the sibling sgl-kernel-npu working tree
host/kernel implementation to64-bit global extents, computed row byte offsets,
queued destination offsets and CopyIn/CopyOut arguments. Retained existing row,
mapping, dtype and block-size validation. CANN's GlobalTensor size/index APIs
were checked and support uint64_t. Built only an isolated standalone extension
from donor539e7c9 plus these two file changes, leaving the installed wheel and
original native-build-v5 binary untouched.

New library: `request_budget_copy64_20260916_v2/native-build/libsgl_fsa_extension.so`;
SHA256 `5108d58dd8dc62dd20525bdcf597b55cd3929e843b9b3073ff033bfc7cacaf44`.
Hardware UT passed7 cases in18.67s on idle34/NPU0: existing registered roundtrip
and planner tests, plus >4GiB Host/NPU storage, FP16/BF16, both copy directions,
invalid index/mask handling and Graph replay after changing live indices/data.
Build logs, source hashes and UT stdout are under local
`results/joint_budget/141.61.55.34/request_budget_copy64_20260916_v2/`.

Starting an owned requests8 service with this extension, HTTP19187/dist29187,
otherwise retaining the recorded8-request/uncapped-token/128GiB Host specification.
The launcher checks idle devices and ports again immediately before launching.
Generation and simultaneous8x1M validation remain pending; component success
does not yet prove E2E completion. Stop the owned service after testing.

The new service reached readiness on19187, with parent6671/start_ticks7757324.
Graph startup passed; all ranks again allocated8091392 tokens/8 requests,
36.65GiB KV/rank. The startup freeze_gc helper logged a connection-refused race
before the listener became ready; subsequent health/server_info/generation worked.
The first capacity verifier incorrectly compared its8M pool minimum to the1M
per-sequence context. Corrected only the test helper to check these independently;
no model/source restart or allocation change was needed.

First generation smoke:8/9 correct;7393 request emitted repeated exclamation
marks after initial reasoning and reached512 outputs. This is NOT declared an
output-cap-only failure or a solved accuracy issue. Saved joint_v2 receipts.
Identical isolated7393 replay passed, and the full identical nine-request smoke
then passed9/9 with normal stop reasons (joint_repeat). Intermittent first-smoke
anomaly remains a qualification limitation. Proceeding with capacity stress:
8 simultaneous exact1M inputs,1024 outputs each; concurrent_1m artifacts pending.

Concurrent driver is live as container PID10859; local exec session55466.
At13:18 CST, the first1M prompt produced one output token after346.745s;
the scheduler then began the second Prefill with one prior request resident,
two KDA slots in use and six requests queued. This demonstrates progressing
shared-pool occupancy, not yet eight-request completion. Continue polling the
same service6671/start_ticks7757324 and benchmark; do not restart due only to
long Prefill time. Local watcher99295 reads live process state and result files.

## 2026-09-16 - Seven-request admission boundary found during capacity stress

The v2 stress successfully passed4GiB physical row offsets and accumulated seven
complete1M contexts, but the eighth remained queued while seven requests decoded.
This is not yet an8x1M concurrency pass. Actual server_info retains8 request/KDA
slots,8091392 tokens and PP microbatch8; Prefill delayers are disabled.

Source inspection found a request-slot double charge at the last chunk:
`_get_new_batch_prefill_raw` compares len(adder.can_run_list) with
get_num_allocatable_reqs(). The continuing chunk request is already allocated,
yet is counted against remaining free rows again. With6 running,1 continuing,
1 free, len1 >= available1 sets batch_is_full before admitting the eighth;
that flag survives the final Prefill merge until a Decode request finishes.

Local scheduler fix adds the already-owned Prefill rows back only to the row
availability side of the admission calculation, preserving PP running limits,
new-row charges and beam/pending-row reservations. Six extracted-production-method
regressions pass. The new keyword is used only by the Prefill can_run_list check.
No running service source was modified. Deploying independent v3 snapshot
`/home/j00628475/request_budget_admission_20260916_v3/source` for verification.
v2 still runs (parent6671, benchmark10859, local session55466); at13:57 CST seven
requests have generated about539 tokens and eighth is queued. Allow completion,
retain the7-concurrency failure result, then stop owned service before launching
v3 after a new idle gate. Watcher96701 is active; duplicate99295 was stopped
locally, not the remote service.

## 2026-09-16 14:18 CST - Stop v2 and gate v3 on 128K actual concurrency

User explicitly requested stopping the current service, finding an idle host,
and first proving actual concurrency with128K prompts before further long tests.
Stopped only v2 process group6671 after PID/start_ticks7757324 verification.
Fresh inspection confirms no running processes on any of34's8 NPUs and no
remaining service/benchmark workers in the container.248/244 remain occupied
by unrelated jobs and were not touched.

Collected v2 final concurrent_1m summary: all8 requests had already completed
1000000 inputs +1024 outputs, zero errors/retractions, elapsed3493.920s; maximum
Decode concurrency7 and peak logged full tokens7007616. Request5 was queued
until the first7 completed; its TTFT3432.765s and TPOT53.730ms contrast with
the first7's TPOT containing Prefill-priority stalls. The benchmark correctly
exited1 for failing actual8-concurrency. It must never be reported as8x1M pass.

v3 source was SHA256-checked against local production/UT files;60 remote UT
passed in16.46s. Starting v3 requests8 on idle34, HTTP19188/dist29188, using
the qualified64-bit copy extension from v2. Retain max-running8, context1048576,
uncapped total tokens, Host128GiB and static0.85. First E2E workload will be8
simultaneously submitted128000-token inputs with128-token outputs. Decimal128K
has a final partial8192-token Prefill chunk, exercising the fixed admission edge.
Require observed Decode batch8 and at least1024000 resident input tokens plus
successful completion of all8 outputs. Further long tests are gated on this
result; no new long-context stress has been submitted yet.
