# Offload versus resident performance matrix, 2026-09-16

Status: offload matrix running in v3; baseline will follow on the same host. No paired
performance conclusion is available yet.

User requirements: max-running-requests64, service context1048576, omit
max-total-tokens, output128, exact decimal inputs128000/256000/512000/1000000.
Use idle NPUs only and never stop unrelated jobs. Baseline cases that cannot
complete all128 outputs are recorded as no data, not extrapolated.

The current dirty SGLang source includes the request-priority memory solver,
removed auto499 solver/switch, and the chunked-Prefill admission fix. The fix
passed actual8x128K concurrency with1025024 resident tokens and all8 completions.
The64-bit row-copy extension also passed actual >4GiB Host/NPU Graph tests.

| Arm | Host | Run directory | HTTP / distributed port |
| --- | --- | --- | --- |
| Offload | 141.61.55.34 | /home/j00628475/offload_matrix_20260916_v3/offload64 | 19234 /29234 |
| Resident baseline (planned) | 141.61.55.34 | /home/j00628475/offload_matrix_20260916_v3/baseline64 | 19244 /29244 |

Container: jzx_sglang_5.3_flash. Host34 reported all8 A5 NPUs idle before
the v3 launch;248 was occupied and untouched. The isolated source snapshot is at
/home/j00628475/offload_matrix_20260916_v3/source. The runs use
FP8 GLM5.3 Flash weights at /mnt/share/w00936111/weights/GLM-5.3-Flash, BF16 KV,
single integrated TP8/DP1/PP1, MTP0, radix off, static0.85, page128, Prefill8192,
Graph sizes1/2/4 and no profiling. Both formal arms use the same host; record
their capacities, model metadata hashes and input hashes, not merely labels.

Offload uses TP-shared Host128GiB, Prefill slots2, selection4096 and the v2
64-bit copy extension. Resident role explicitly disables integrated/Prefill/
fused offload and resets Host budgets to0 inside fsa_offload_env.sh. Matrix
validation checks the live service environment flags before sending requests.

Run all four single-request lengths, then all four batches of64 concurrently
submitted distinct prompts, each128 outputs. The finite64-request batch puts
backpressure on the actual resident capacity; it is not an infinite steady-state
traffic test. Do not claim client64 means actual64 simultaneous Decode. Keep
actual Decode batch histogram, peak resident tokens, retractions and queue evidence.
One16K/128 global warmup precedes timed cases; shape-specific cold effects may
remain and must not be hidden. There is one measured trial per case.

Main Excel columns use arithmetic mean TTFT/ITL/TPOT in milliseconds.
Output throughput=sum completed output tokens / batch makespan; E2E throughput=
sum(input+output tokens) / makespan. Per-card throughput divides by8. Makespan
uses monotonic HTTP request start/end, excluding payload construction and
result writing. ITL is reconstructed from streaming increments with token-weighted
intervals; it can include Prefill-priority pauses. Failed/partial cases retain
raw evidence but get no aggregate performance metrics. Explicit failures are
no data; unrun/in-progress cases remain pending.

Workbook: D:/newCode/sglang/offload性能对比分析.xlsx. Preserve a backup and the
original14-column template. Separate single-request and batch sheets, plus
settings, evidence/status and paired gains. Only measured paired successful
cases may have computed gains.

DeepSeek external subagents are used for a read-only metric audit and the workbook
exporter implementation. Parent reviews their results and owns all remote actions.
The audit's claims that max_total_tokens and max_total_num_tokens must be the same
key, or that the offload shared-pool capacity gate should be bypassed, were rejected:
the former is the optional user cap, the latter the actual resolved pool capacity.
The resident launcher disables Host budgets internally. Valid audit concerns about
payload timing, failure evidence, process cleanup and workload labels were addressed.

## Startup and validation checkpoint

52 local budget/admission tests passed;12 metric/export tests passed. The workbook
worker's exporter has been parent-reviewed. Raw failed requests remain as evidence,
but aggregate cells are populated only from successful case metrics. Do not apply
the reviewer's proposed deletion of failed raw evidence or reinterpret retractions
as unmeasured: successful requests with retractions retain that count for analysis.

34 offload service ready: parent30356/start_ticks8634102, HTTP19234. Matrix runner
PID35075/start_ticks8671875 is live under the offload64 run directory.

244 resident startup failed before inference in aclrtGetMemInfo with107002:
Package_Error_Verify_Package(E30009), unsigned cann-hybm-compat.tar.gz rejected by
enabled device package signature verification. No active scheduler/launch_server
remained. This is an environment failure, NOT baseline inability to infer1M and
not performance data. The logs were retained locally under244/offload_matrix_v1.
No device reset, signature-policy change or other-job kill was performed. Since248
is occupied, use34 for the resident arm after offload finishes and cleanup confirms
idle devices. That also removes cross-host hardware differences from the formal
comparison. 244 remains an excluded failed preparation attempt.

Pending: all measured cases, paired input/config checks, workbook population and
final owned-service cleanup.

## 2026-09-16 16:12 CST - Interrupted v1, restarting in v2

Fresh process inspection after the user resumed confirms the v1 service30356 and
runner35075 no longer exist in the container. The server log ends15:51:29 during
Prefill with51 completed prefixes and12 waiting requests. No saturation request
completed128 outputs, so no batch throughput can be reported. The coordinator
correctly detected the missing runner and recorded blocked without launching a
baseline. NPU inspection confirms all8 devices idle. No process was killed by
this recovery. Retain v1 artifacts; deploy v2 to a separate root before restart.

Completed v1 offload single128K/128 result: TTFT25971.123ms, TPOT63.128ms,
output3.765985 tokens/s, total-token throughput3769.750502 tokens/s. This is one
unpaired sample, not an offload speedup. Actual64-slot token pool was7288064.
Raising concurrency from8 to64 consumes more fixed per-request memory, so do
not reuse the earlier8091392-token estimate for this configuration.

The original workbook is open/locked; a byte-identical backup was saved as
offload性能对比分析.backup-20260916_160812.xlsx. The exporter now saves atomically
and falls back to offload性能对比分析_测试结果.xlsx when the original cannot be
replaced. That results copy contains the completed single128K row; other cases
remain pending. Exporter regression suite remains9 passed.

v2 will collect single-request results for all four lengths first, then the64
request batch cases. The offered workload and metric definitions are unchanged.
Runtime harness files are uploaded at matrix launch; the source snapshot remains
isolated and the parent will verify the actual run-dir scripts. Use
/home/j00628475/offload_matrix_20260916_v2 on34. The baseline will run on the same
host after the offload arm closes and a new idle-resource gate passes.

For the formal v2 comparison, timed cases sample device usage via npu-smi but
do not repeatedly scan every worker's smaps_rollup. Scanning the much larger
offloaded Host mappings can add different monitoring work to the two arms;
Host backing configuration is recorded separately. The v1 single128K measurement
used the earlier sampler and remains historical, not a replacement for v2 data.

## 2026-09-16 - User-requested restart after accidental container stop

The user confirmed accidentally stopping the service and requested a restart.
Fresh inspection shows container restarted at 2026-09-16T08:21:05Z; all eight
devices were idle and the old service506, matrix4677 and coordinator242 were
absent. The v2 matrix was interrupted during16K warmup, before any measured
case completed. Its coordinator correctly blocked on missing runner receipt.
The stale ready manifest is not evidence of a live service. Keep v2 artifacts.

Restart in /home/j00628475/offload_matrix_20260916_v3 on141.61.55.34 with the
same TP8, max-running-requests64, context1048576, uncapped automatic token pool,
Host128GiB,64-bit copy extension, profiling disabled and19234/29234 ports.
Launch passes a fresh idle-resource gate. Run all four single-request cases,
then64-request batches, and finally the same-host resident baseline. No unrelated
process was stopped during recovery. The v2 interruption is not a performance
failure and must not be used as evidence that offload cannot support a length.

v3 service605/start_ticks8997697 became ready at16:30:07 CST (HTTP health200).
Matrix runner4844/start_ticks9045050 was submitted after readiness. The baseline
coordinator is queued for this v3 root only. Both run directories from earlier
interruptions remain intact; v2 offload logs were also collected locally.

At16:35:51 CST, live inspection confirms service605 and runner4844 running:
single128000 and256000 each completed with no errors; single512000 is in
Prefill. Coordinator5291/start_ticks9053669 is waiting for this offload arm.
Recovery is successful; the full paired experiment is still in progress.

## 2026-09-16 16:47 CST - All v3 offload single-request cases complete

| Input tokens | Output tokens | TTFT ms | TPOT ms | Output token/s |
| --- | --- | --- | --- | --- |
| 128000 | 128 | 21045.441660 | 46.808499 | 4.742452 |
| 256000 | 128 | 50526.033480 | 47.938271 | 2.260913 |
| 512000 | 128 | 115929.329760 | 49.760224 | 1.047044 |
| 1000000 | 128 | 322724.238230 | 53.635634 | 0.388425 |

All four have cached_tokens0, retractions0 and exactly128 increasing stream
events. Parent independently recomputed TTFT/TPOT and makespan throughput from
stream events and monotonic timestamps; all match. DeepSeek read-only audit
also verified the first three cases; its local snapshot predated1M completion.
Evidence: results/joint_budget/141.61.55.34/offload_matrix_20260916_v3/offload64/
matrix/single_*/request_0. These are measured offload-only results, no paired gain.

The live v3 resolved token pool is7289216 (slightly different from v1);64 request
limit,1048576 context, max_total_tokens null, TP8 and profiling0 were verified
in server_info/runtime_env. All integrated/Prefill/fused flags are1, TP-shared
Host KV1. Do not substitute the earlier v1 pool or measurements.

The original Excel is now writable. Export at16:46:45 updated its single-request
sheet with these four measurements and left batch/baseline cells pending. The
old _test-results fallback workbook remains a historical v1 artifact and is not
the current deliverable. The original byte backup is preserved.

Current workload is saturation128000:64 requests submitted together. At16:46:14
the Prefill log shows8 running prefixes and55 queued, not yet a Decode batch
measurement. Baseline coordinator5291 waits for the complete offload arm.

## 2026-09-16 17:18 CST - First offload batch complete; unattended collection active

Offload saturation128000 completed all64 requests, each128 outputs, zero
retractions and zero cached tokens. Parent verified every raw stream and
recomputed aggregate metrics from request start/end monotonic times.

| Measurement | Value |
| --- | --- |
| Whole batch elapsed seconds | 1493.259468 |
| Mean TTFT ms | 691989.173558 |
| Mean TPOT / ITL ms | 4493.455325 |
| Output token/s | 5.485986 |
| E2E token/s including input | 5491.471627 |
| Actual maximum logged Decode batch | 56 |
| Actual maximum logged full tokens | 7175168 |
| Decode batch log histogram | 56:3, 8:3 |

The64 offered requests are not64 simultaneous Decode. The histogram supports
a56-request wave followed by8; it is a sample of logged batches. The large
mean TPOT includes Prefill-priority scheduling delays:62 requests had at least
one >10-second token gap. The largest was1166.608908s between token1 and2 in
request56. Gaps >10s contribute4123.873906ms of4493.455325ms mean TPOT (91.8%).
Median ITL323.244455ms is also a client measurement, not kernel-only latency.
Retain the original end-to-end metrics; do not silently remove these pauses.

At17:17:38 CST the same owned service605/runner4844 is processing saturation256000,
with11 prefills completed and52 queued. No errors, no restart. Baseline waits.

Automatic local follower is now active: session34912, PID37100, started17:00:35,
24-hour timeout, polling every60s. It writes results/joint_budget/141.61.55.34/
offload_matrix_20260916_v3/follower_status.json, collects completed case evidence,
and refreshes Excel without modifying remote processes. Script:
scripts/npu/follow_fsa_matrix.py. Parent corrected the delegated export path to
use arm/matrix, added matched weight/config/input-hash checks, bounded log reads,
and tightened final cleanup/export qualification. Real collection and Excel
export have passed; full matrix completion is still pending. The follower does
not restart interrupted jobs and does not store the SSH password on disk.

The original Excel became locked again. Latest export17:09:29 is therefore at
D:/newCode/sglang/offload性能对比分析_测试结果.xlsx, now containing v3's four
single cases and the128K batch. This supersedes the earlier fallback's v1 data.
follower_status.json always records the actual latest workbook path. No paired
baseline result or throughput improvement is claimed yet.

## 2026-09-16 17:31 CST - User clarified request-level latency aggregation

TTFT, ITL and TPOT now explicitly mean: calculate each request's latency, then
take the equal-weight arithmetic mean over requests. Batch makespan is used
only for throughput. TTFT_i=first token arrival minus request_i send time;
ITL_i=mean of request_i's per-token intervals; TPOT_i=(last-first)/(outputs-1)
for the observed one-token stream. Queueing and Prefill interruptions within a
request remain included. No batch-duration division is used for latency.

TTFT and TPOT already followed this rule. fsa_matrix_metrics.py now also makes
ITL request-weighted, instead of flattening all intervals. ITL percentiles in
new summaries describe per-request ITL means; separate token_itl percentiles
retain token-interval diagnostics. Historical128K token ITL median323.244455ms
above describes the old interval distribution, not a request-level median.
Exporter reconstructs the three latency means from raw per-request records so
the currently running server/benchmark requires no restart. The remote active
benchmark retains its loaded metric helper; with128 outputs and127 intervals
per request the old/new means are exactly equal for all completed evidence.
Future launches use the updated local helper after source deployment.

Parent recalculated all five completed cases, including all64 requests in the
128K batch: maximum difference from stored TTFT/ITL/TPOT was0.0ms. The128K
request means remain TTFT691989.173558ms, ITL/TPOT4493.455325ms. The1493.259468s
batch duration is used only to obtain output5.485986tokens/s and the other
throughput rates. Do not divide those request latency means by64 again.

Artifacts: v3 local results/latency_per_request.csv (68 individual requests)
and latency_calibration.json, with the same result root as follower_status.json.
14 focused metric/export tests pass, including unequal request interval counts
and invariance of latencies when only batch duration changes. Original Excel
was writable at17:30:33 and its formulas/notes/results were refreshed.

Only local follower34912 was stopped to reload the exporter; replacement
session3031 uses the same remote v3 run. NPU service605, runner4844 and
coordinator5291 were not stopped or modified. Full paired matrix remains running.

## 2026-09-16 17:42 CST - Superseding latency definition: exclude initial queue

User explicitly requires inference-start TTFT, excluding initial request queueing.
This supersedes the previous17:31 TTFT definition. Historical TTFT values above
are client send-to-first-token latencies including queueing, NOT the requested
inference-start TTFT. In particular, random request55's990.741830s must not be
presented as inference-start TTFT. Its1881.744580ms TPOT already excludes initial
queue time because it starts at the first generated token. Decode scheduling
pauses after that token remain included; changing the initial timing origin
alone cannot remove those pauses. Active execution-only timing would require
additional per-forward instrumentation and must not be guessed from intervals.

Source inspection: scheduler.py sets set_forward_entry_time at first admission;
req_time_stats.py preserves this across chunked Prefill and exports forward_entry_time,
prefill_finished_time and queue_time. tokenizer_manager.py exposes these only
when enable_metrics is true. The active v3 run lacks these fields and has no
per-request ReqTimeStats log. No accurate historical inference-start TTFT can
be recovered from current client timing, aggregate batch logs or token counts.

Local metrics now calculate TTFT=(prefill_finished_time-forward_entry_time)*1000,
using server-side first-prefill completion as first-token production boundary.
Missing timestamps, invalid intervals or retractions produce no TTFT rather
than a guessed subtraction. Raw client_ttft remains separately available. ITL
and TPOT retain per-request averaging. Exported TTFT cells now read
"未采集推理起点" for existing cases;17:40:41 export updated the original workbook.
latency_per_request.csv and latency_calibration.json were also regenerated,
keeping client queue-inclusive timing in an explicitly named separate column.
16 focused tests pass, including queue-independent TTFT and missing timestamp handling.

Local fsa_pd_role.sh now enables native --enable-metrics for FUTURE matched
offload/resident launches (not profiling). This has NOT been copied into the
active remote v3 source: both active arms retain matching prior settings.
v3 throughput tests continue; inference-start TTFT needs a paired follow-up
with timestamps on both arms. Do not apply timestamp settings to baseline only.
Follower3031 was locally replaced by session21301; the remote service, matrix
runner and coordinator remain untouched. Follow follower_status.json for state.

## 2026-09-16 18:00 CST - Session handoff prepared

User requested a complete handoff including latest supplied SSH passwords.
Entry document is outside the Git repository:
D:/newCode/sglang/HANDOFF_OFFLOAD_PERF_20260916.md.
It contains remote access, source/OPP/native paths, exact launch/monitor/export
commands, workload and metric definitions, missing inference-start TTFT gate,
current results, capacity estimates distinguished from observations, and the
next-session sequence. Do not include that credential-bearing document in a PR.

Live remote refresh at17:55:54 confirms service605 and runner4844 still running,
saturation256000 completed56/64 requests with no errors, remaining8 processing,
and the same-host baseline coordinator still waiting. The handoff does not
stop or replace any remote process. The active v3 run still lacks server request
timing metadata; a new matched enable-metrics round needs a small-request field
check and a benchmark warmup guard before long requests. Current throughput
run remains valid for its original matched monitoring configuration.

## 2026-09-17 - AISBench follow-up bottleneck analysis supersedes cc interpretation

User provided AISBENCH_RESULTS_DATA_HANDOFF.md describing later AISBench runs,
different from v3. Full report: ../AISBENCH_BOTTLENECK_ANALYSIS_20260917.md.
Read-only collection downloaded both completed service logs and384 requests'
SQLite-backed timestamp arrays; no service was started/stopped. Source currently
at remote/home/j00628475/sglang_remote matches local scheduler/integrated_offload.

Actual Decode waves: offload128K56+8 versus resident23+23+18;256K28+28+8 versus
11x5+9;512K14x4+8 versus5x12+4. Capacity advantage IS exercised. AISBench cc is
sum(client E2EL)/batch span, including queued requests, not NPU concurrency;
the handoff's resident-not-capacity-bound conclusion fromcc42.7 is invalid.

Offload wall spans covered by consecutive Prefill logs represent90.0%,92.3%,94.2%
of total time at128K/256K/512K; resident92.8%,83.5%,84.8%. These are sampled
wall spans, not kernel timing. Both configs Prefill-first, mixed_chunkFalse,
prefill_decode_interval0; only128 outputs amortize each long Prefill. Dense
streaming-window diagnostic at512K saves about647s for offload but about955s
is added outside those windows, yielding307s longer batch and4% less throughput.
Ordinary Prefill skips fused Decode dispatch and restores whole existing prefix
rows each DSA layer/chunk. This is a code-supported source of increasing cost,
not a measured bandwidth attribution. Longer-output benefit still needs testing.

New AISBench convention intentionally uses client queue-inclusive TTFT per its
later user decision; do not mix it with v3's missing inference-start TTFT.
Installed AISBench concatenates ITL arrays across requests, so ITL percentile
is an interval distribution, not a distribution of per-request ITL means.
Artifacts and parent-reviewed independent DeepSeek audit are under
../aisbench_results/bottleneck_analysis_20260917/.

## 2026-09-16 18:12 CST - v3 ended by a manual container restart; environment audit

**User confirmed they restarted the 34 container by hand.** The v3
service/runner/coordinator are gone as a direct consequence. This is NOT a kernel,
capacity, OOM or performance failure and must not be recorded as one.
Container `jzx_sglang_5.3_flash` is `running` with
`StartedAt=2026-09-16T10:05:30Z`. The previous `StartedAt`
(`2026-09-16T08:21:05.77976802Z`) and PIDs605 / 4844 / 5291 are all invalid;
none are present in `/proc`. The `ready` value in `offload64/manifest.json` and
`waiting_for_offload` in `baseline_coordinator/status.json` are historical only.

### New standing rule (user, highest priority)

> Services running in the three environments' `jzx_sglang_5.3_flash` containers,
> and services in any other container, may belong to other people's tasks.
> **Never stop them. Run tests only on an environment verified idle.**

Consequences: no `pkill`/`kill` of unowned processes, no container restart or
device reset to free cards, and **idleness must be proven from the `npu-smi info`
Process section (`No running processes found in NPU 0..7`), not from util=0 or
unused HBM** -- utilisation is instantaneous and a zero sample is not idleness.

### Environment audit at 18:12 CST

| Host | HBM per die | Service processes | Idle | Occupant |
| --- | ---: | ---: | --- | --- |
| **141.61.55.34** | 4 749..4 753 MB (driver baseline) | 0 | yes | -- |
| 141.61.54.248 | 87 660..89 883 MB | 8 schedulers | no | other user |
| 141.61.54.244 | 90 600..92 825 MB | 8 schedulers | no | other user |

244's occupant is a large unrelated job: `GLM-5.3-Flash-0day-A5-convert`,
`--quantization modelslim`, `--speculative-algorithm NEXTN` (4 MTP steps),
`--max-running-requests 16`, `--disaggregation-mode decode`, port8811. It is not
ours and must not be disturbed. 248 is likewise occupied.

### v3 state at interruption (artifacts retained on disk)

- `warmup` (16K/128) and all four `single_*` cases: complete.
- `saturation_128000`: 64/64 complete.
- `saturation_256000`: **64/64 complete** (runner log ends `completed: 64`, 18:03).
- `saturation_512000`: **interrupted.** Every `request_*` directory holds only
  `input_metadata.json` + `stream_events.jsonl`; no `result.json` or
  `response.json`. `progress.json` had advanced to `input_tokens=512000` with an
  empty `completed` list, and `server.log` ends at18:05 with
  `#running-req: 0, #queue-req: 63, pending-token: 32 301 056`.
- `saturation_1000000`: not started. Resident baseline: not started.

Per this record's own rules the 512K batch is **no data**, not a failure: it
supplies no aggregate metric and must not be extrapolated or reported as a
failure. In particular the interruption says nothing about offload support at512K.

### Tooling

The three `scripts/npu/` remote helpers previously forced an interactive password
(`look_for_keys=False, allow_agent=False` plus an unconditional `getpass()`),
which defeated the already-installed key on all three hosts. Added
`scripts/npu/ssh_auth.py`; connection order is now explicit `password` argument ->
local private key -> interactive prompt. `remote_joint_budget.py`,
`follow_fsa_matrix.py` and `remote_fsa_native_capacity.py` all use it. Verified:
`--action status` no longer prompts. Local key naming is misleading --
`~/.ssh/id_ed25519.pub` is the private key, `~/.ssh/id_ed25519.pub.pub` the public one.

The codex long-term index is `C:\Users\j00628475\.codex\memories\MEMORY.md`
(SGLang group from line1156, Tasks1-4, plus ~200 notes under
`extensions/ad_hoc/notes/`). **It is stale**: last updated05:19Z (13:19 CST), so
the entire offload/resident matrix task is absent from it and lives only in this
file and the handoff.

Pending: the512K and1M saturation batches, the same-host resident baseline, and
the paired workbook. All require an idle environment; only34 qualifies today,
and248/244 must be re-checked later rather than assumed free.

## 2026-09-17 11:40 CST - Record wider Decode Graph coverage for next test

User explicitly asked to persist the Graph limitation for future testing.
The later AISBench runs capture only1/2/4 via --cuda-graph-bs-decode. Their
large Decode batches run eager; resident512K final batch4 is Graph. Alternating
Prefill and Decode phases does not itself preclude independent Decode replay.
No large-batch FSA Graph support/performance claim has been validated yet.

Candidate matched list:1,2,4,5,8,9,11,14,18,23,28,56,64. Verify capture correctness,
actual replay and planner/Host access in both arms, then measure. Re-record HBM,
Graph memory, automatic token capacity and actual Decode batch, because larger
Graph/planner reservations can change the capacity being compared. First use
common fitting fixed batches, then E2E; keep output length and scheduling unchanged
to isolate Graph effects. Do not silently enable wider capture on only one arm.

Full next-session instructions appended to ../AISBENCH_RESULTS_DATA_HANDOFF.md
section13 and ../AISBENCH_BOTTLENECK_ANALYSIS_20260917.md. No launch script or
remote service was changed for this documentation task. Earlier v3 status above
is historical; the later AISBench handoff describes the completed comparison.

## 2026-09-17 - Necessity of layerwise Prefill offload investigated

See ../PREFILL_RESIDENCY_FEASIBILITY_20260917.md. Remote current offload configuration,
pool, budget, fused backend and integrated runtime match local files. Current
integrated service rejects Prefill-off/Decode-Host-on; no independent switch
provides that combination. But Decode fusion requires correctly populated Host
KV, not necessarily two-slot eviction/restore during Prefill. Single1M resident
already completed; preserving about7 concurrent1M contexts with full global HBM
KV would not fit.64 concurrent1M is beyond both current Host and resident Indexer
budgets;64 queued requests are a different workload and were not tested at1M.

Promising new-mode design: keep all11 DSA layers for ONE active Prefill request
across chunks (~11GiB/rank for maximum1048576 context), mirror new KV to Host or
copy at transition, then recycle this independent workspace. Current two slots
scale with the7288576-token GLOBAL pool (~13.90GiB/rank), so an active-request-
sized workspace may remove prefix restores without increasing that budget term.
Must separate local/global mappings, enforce Prefill leases, validate TP Host
visibility and lifecycle, and recalculate global Indexer/selection/KDA/Graph budget.
This is a design feasibility finding, not implemented or hardware-tested. No
service/configuration was changed during the investigation.

## 2026-09-17 - User chooses current Prefill offload plus layerwise H2D prefetch

User keeps the existing two-slot Prefill offload mechanism and prioritizes
planning overlap between current-layer computation and next-DSA-layer prefix
restore. Plan: ../PREFILL_H2D_OVERLAP_PLAN_20260917.md. The alternative all-layer
resident active-request workspace is deferred, not selected for implementation.

Use a dedicated H2D stream, prefetch first two DSA layers at batch entry, compute
waits only its ready event, release event from the last KV user permits loading
layer+2 into the reused slot. Preserve current D2H and TP publication order;
bridge prior Host visibility into copy-stream dependencies, retain tensor/Host
lifetimes, and drain safely on batch transitions/abort. No per-layer global sync.
Map DSA execution order through the hybrid model pool, not model layer_id+1.

Implement behind an initially-off A/B switch, first CPU dependency tests then
actual NPU/TP multi-chunk correctness, then real layer profile and end-to-end.
Two streams alone do not prove overlap; AIV copy may contend with computation.
Keep Graph expansion and scheduling changes separate until isolated benefits
are verified. This checkpoint is a plan only; no runtime or service was modified.

## 2026-09-17 - vLLM donor overlap protocol incorporated into Prefill plan

Directly reviewed D:/newCode/vllm-ascend-dev/vllm-ascend-dev at HEAD788b17f06a3fffebfdf01dd76f905de7ff5d9b16
(working tree memcache_backend.py modified). AscendStore layerwise implementation
does have background H2D prefetch and scatter-ready early D2H save. It uses
separate send/receive worker threads calling Memcache batch_copy; do not describe
it as proven torch-stream or hardware overlap, nor as full masking without trace.

Evidence: pool_worker.py1131/1175 prefetch+load wait;1201 early save; kv_transfer.py
1155/1241 slot_free gates save+attention+PD;1385 load waits old owner and compute
start gate. sfa_v1.py1770 load-before-write,1962 KV-written,1974 attention gate.
memcache_comm_fence.py31 guards event recorded-before-wait; its reset docstring
is stale about own-layer binding, actual caller binds preceding compute window.

Updated ../PREFILL_H2D_OVERLAP_PLAN_20260917.md: first layer loads at entry; next
layer loads at preceding DSA compute_start, reused slots additionally wait prior
owner release. This supersedes initial entry-prefetch-A/B default. Early release-
only/entry lookahead remains a later A/B strategy. Add hooks for all DSA layers
including TopK reuse, explicit source lifetime/Host generation, empty-task gates,
and error propagation. Keep D2H async as a separately tested second phase with
TP publication ordering. No runtime code changes, no service launch, no donor
UT or profile executed. The previously mentioned extra rebase path was absent;
its features were not attributed to the inspected donor.

## 2026-09-17 - Detailed bidirectional Prefill overlap design v2

User requested complete H2D+D2H overlap design. Primary implementation spec:
../PREFILL_BIDIRECTIONAL_OVERLAP_DESIGN_20260917.md. Old H2D plan now points to v2.
Keep two global-pool physical slots and current Host/Decode semantics. Two initially
off switches give00/10/01/11. Separate compute,H2D,D2H streams. D2H reads only new
rows FROM THE PHYSICAL SLOT after scatter, so no per-layer temporary-KV staging;
slot reuse waits both compute readers_done and writer save_done. Runtime holds
indices/regions through actual completion, with bounded in-flight generations.

H2D A starts at entry; compute_start A triggers B; compute_start B triggers C
after A's readers/save finish. Never bind a restore to its own future start event.
D2H async mode separates copy-only from Host publication. Under audited ordinary
Prefill (no current-generation Host reader), defer per-layer/component TP publish
to forward tail on COMPUTE stream after copy completion; no copy-stream HCCL.
Host visible event gates next chunk/Decode. Slot-free and Host-visible are distinct:
requiring forward-tail visibility before slot reuse would deadlock two-slot reuse.

First scope ordinary integrated eager Prefill, MTP0,CP1/DCP1; unsupported combos
fail before execution. Plan includes scatter/readers hooks, zero-row/partial-page
control, TP failure handling, stream lifetime, copy>4GiB, multi-chunk/TP2/TP8 UT,
Decode Graph transition, profile and matched A/B. Native Host visibility and HCCL
completion-to-caller-stream guarantees require qualification, not assumptions.
No runtime code or remote service changed. Graph range expansion remains separate.

## 2026-09-17 - Prefill overlap implementation and first hardware gate

Implemented opt-in PipelinedIntegratedOffloadRuntime with dedicated H2D/D2H
streams, two physical slots, predecessor compute-start prefetch, scatter-complete
slot-backed new-row saves, reader+save slot reuse gates, forward-tail ordered TP
publication, bounded retained generations and failure drain. Existing runtime is
unchanged when both SGLANG_NPU_PREFILL_H2D_PREFETCH and
SGLANG_NPU_PREFILL_D2H_ASYNC are false (defaults). Unsupported integrated/MTP/CP/
mixed-chunk configurations are rejected. Empty writes still record completion.

Local focused control/config/Host tests:55 passed. Remote141.61.55.34 was checked
idle on all8 cards; existing container jzx_sglang_5.3_flash only had its shell.
Isolated deployment:/home/j00628475/prefill_overlap_20260917_v1/source; retained
validated copy64 extension and attention-opp-v8. No existing service/process killed.
Single NPU0:16 passed,1 skipped(profile opt-in),20.69s. Four switch combinations,
FP16/BF16,4 layers/2 slots/3 generations,partial pages,empty writes,physical-slot
checks,registered Host read/write,and subsequent Decode current-write Graph tested.
Attention oracle here is torch matmul/softmax; this is NOT full-model/SFA speed proof.
Logs/hashes:results/prefill_overlap_20260917/{single_v1.log,deployed_sha256.json}.
TP2/TP8,actual SFA overlap trace and matched performance remain pending at checkpoint.

## 2026-09-17 - TP publication gate and CPU regression

Focused CPU suites expanded:113 passed. TP2 passed7 tests per rank. Initial TP8
failed a CPU Host assertion on layer3: fast writer rank could enter the subsequent
Decode test and overwrite three rows before slow ranks finished CPU validation.
Added an all_gather_object after CPU Host checks, BEFORE starting the independent
Decode test. No synchronization was inserted in the Prefill layer/generation loops
or production runtime. TP8 rerun passed7 tests on each of8 ranks (22.63-22.67s).
Logs:results/prefill_overlap_20260917/tp2_v1.log,tp8_v1.log,tp8_v2.log.
Torchrun terminated only its own test-worker process group on the first test failure.

New actual NoPE SFA test verified four switch modes produce bitwise-equal Prefill
outputs and match a CPU attention oracle. Initial subsequent FSA Decode assertion
exposed invalid test request slot0 (reserved padding); corrected to real slot1 and
rerunning. No production change from that test correction. Full-model/E2E and
component speed qualification still pending; switches remain off by default.

## 2026-09-17 - Actual SFA/Decode transition and overlap trace

Actual NoPE SFA4-layer tests pass at prefix8192/chunk128 and prefix131072/chunk8192.
All00/10/01/11 Prefill outputs bitwise equal; CPU oracle passes BF16 tolerance.
Fused Decode immediately after queued Prefill and subsequent Graph replay pass.
128K/8K seven-iteration component medians(ms):00=116.887810,10=114.943756,
01=115.564178,11=115.256493; reductions1.66%,1.13%,1.40%. Timing excludes profiling.
This is real SFA+registered Host copy, but not full-model throughput/TTFT/TPOT.

Trace collected for all4 modes, saved under remote isolated root/trace_128k and
local results/prefill_overlap_20260917/kernels_{00,10,01,11}.csv. Intervals overlap,
but concurrent H2D duration rises from~10.29ms to~28.5ms alongside18.42ms SFA;
D2H from~0.35ms to~18.7ms. Consistent with AIV scheduling/resource contention,
not proof of useful full-rate copy overlap. Keep both new switches default-off.

CPU final additional capture/changed-stream/Decode visibility cases:116 passed.
Full TP8 model smoke launched on idle34 in isolated root/e2e_smoke, port33247,
dist29850,flags11,context32768,pool32768,request cap2,Host4GiB,MTP0,Graph1/2/4.
Owned server container PID104405/start_ticks17136482; launcher104067. At checkpoint
all ranks initialized HCCL and model weight loading started. No E2E success claim
yet. Do not restart/kill other jobs; only manifest-matched owned process group
may be stopped after smoke. Local implementation guide:PREFILL_OVERLAP_IMPLEMENTATION.md.

## 2026-09-17 - Final bidirectional overlap implementation smoke passed

TP8 integrated GLM-5.3 Flash service in isolated34 root completed both smoke cases:
10 prompt tokens ->64 output tokens,18.364s cold request elapsed;
10417 prompt tokens ->64 output tokens,37.254s cold request elapsed.
Both HTTP200,length completion,num_retractions0. Long prompt logs show Prefill
8192 then padded2304 new-token batches and subsequent Decode npu_graph=True.
Runtime process environment verified source PYTHONPATH is isolated deployment and
SGLANG_NPU_PREFILL_H2D_PREFETCH=1,SGLANG_NPU_PREFILL_D2H_ASYNC=1.
Initial health delays traced to progressing KDA Triton compiler kernels, not a
claimed FSA deadlock; first /freeze_gc connection-refused diagnostic was nonfatal.
Manifest reachedsmoke_passed. These are correctness/lifecycle smoke observations,
not matched TTFT/TPOT evidence; no claimed long-context E2E gain or new1M test.

Saved local e2e_manifest.json,e2e_response_0.json,e2e_response_1.json,e2e_server.log,
e2e_state.txt and e2e_smoke_launch.log in results/prefill_overlap_20260917.
Stopped ONLY owned process group104405 after checking /proc start_ticks17136482
against its manifest. Both new env defaults remainfalse in local code. Final116
CPU tests,16 single-NPU lifecycle tests,7 shared-Host tests per rank atTP2/TP8,
two actual-SFA shape/Decode-transition tests and full-model smoke qualified.
128K/8K4-layer component gain is only1.40% for11 (10alone1.66%);trace shows copy
duration inflation alongside SFA. Further useful masking needs kernel resource
contention work, not an unsupported promise from using separate streams.

Post-stop npu-smi confirmed all8 cards idle/no running processes. Final14-file
isolated sync hashes verified (production code,tests,implementation guide).

## 2026-09-17 - Source comparison with current vLLM Ascend donor

Read current D:/newCode/vllm-ascend-dev/vllm-ascend-dev HEAD788b17f06a3fffebfdf01dd76f905de7ff5d9b16.
Worktree now has multiple dirty files; conclusions refer to inspected worktree,
not pristine commit or a measured deployed donor. No remote job or runtime edit.
Core scatter-ready save,preceding compute gate and slot reader/save ownership
match conceptually. Important differences:
- Donor kv_transfer.py201/244 uses MemCache DistributedObjectStore.batch_copy from
  separate receive/send threads; current fsa_native.py212/243 uses registered-Host
  unidex AIV row copy(defaultblock_dim8) on NPU streams. Wrapper code does not
  establish donor's actual SDMA/RDMA/AIV selection or hardware overlap.
- Donor sfa_v1.py1962/1974 notifies save and opens prefetch BEFORE indexer post-
  processing. Current NPU forward_dsa_prepare_npu at465 runs indexer before core
  at505 invokes attention; actual scatter occurs inside ascend_backend.py1096,
  prefetch gate at1177. Thus current saves/prefetch miss this layer's Indexer
  window. Moving the hook alone is insufficient:prepare/scatter/written ordering
  and duplicate cache writes must be adapted together. Following MLP/KDA windows
  can still overlap while copies remain outstanding;not restricted to SFA alone.
- Donor layerwise_config.py94/124 supports configurable prefetch depth and default
  independent first/last layers; only independent-layer cache misses/tail need
  reload(pool_worker.py1045/1102). Current all DSA layers map modulo2 and restore
  all history rows each chunk. Donor shared-buffer default is num_layers,not2;
  compare same HBM budget before attributing speed to its pipeline.
- Donor task arrays are page/block address+size lists with background preparation,
  configurable byte/block split and optional rank stagger(default0). Current uses
  per-token index/mask once per forward and per-layer row-copy submission, no
  runtime-exposed packet/chunk or stagger tuning.
- Current slot release after SFA readers+save;donor after post-o_proj attention
  event+save+optional PD waiter. Current integrated async runtime has no PD reader
  gate and is intentionally not admitted to disaggregated service/MTP/mixed/CP.
- Current shared-Host writer publishes each component at forward tail with TP
  broadcasts;donor uses store/task completion and request leases. Do not infer
  that donor lacks all TP synchronization or copy either publication protocol.

Next candidate priority:qualify alternative copy backend/resource usage with same
component UT,then test earlier scatter/prefetch before Indexer. Resident edge
layers/deeper buffering require explicit new HBM budget. Current1.4% component
gain and trace inflation support contention investigation,not proof donor is faster.

## 2026-09-17 - Live container batch-copy API inventory

Read-only/import inspection on141.61.55.34 containerjzx_sglang_5.3_flash:
Python3.12.13,memfabric_hybrid1.2.0,memfabric_zbal1.1.3. memcache_hybrid
find_spec returnedNone;the donor DistributedObjectStore.batch_copy is not
directly importable in this interpreter. Existing /usr/local/memfabric_hybrid/
set_env.sh is absent now;MemFabric installed in site-packages imports successfully.

Available compiled Python binding:memfabric_hybrid.bm.BigMemory.copy_data_batch
(src_addrs,dst_addrs,sizes,count,type,flags=0,stream=0),plus copy_data_batch_partial_succeed,
register/unregister and wait. bm.create/create2 expose BmDataOpType.SDMA(default).
TransferEngine also exports batch_transfer_sync_read/write and
batch_transfer_async_read_submit/write_submit(stream argument);installed headers
document asynchronous stream submissions as SDMA-only. C header smem_bm.h197
declares smem_bm_copy_batch. Do not conflate these bindings with the missing
MemCache object-store wrapper or claim zero-change compatibility of Host pointers.

No engine/BigMemory object initialized,no device buffer allocated,no transfer
or performance test run,no jobs stopped. API import/signature availability is
confirmed;actual H2D/D2H correctness,Host-region registration/TP sharing and copy-
compute overlap still require idle-NPU UT. Probe and evidence:
results/prefill_overlap_20260917/inspect_batch_copy.py,batch_copy_probe.log;
remote /home/j00628475/prefill_overlap_20260917_v1/inspect_batch_copy_probe.py.

## 2026-09-17 - SDMA validation prototype and first real roundtrip

Initial occupancy34/248/244 all busy;no jobs stopped. Prepared validation-only
bridge using aclrtMemcpyAsync on existing RegisteredHostRegion.host_ptr (not its
device alias). MemFabric BigMemory API exists but its user registration contract
is HBM-specific;do not claim it accepts current Host allocation unchanged.
Isolation:/home/j00628475/sdma_validation_20260917_v1/source;native test library
libfsa_sdma_validation.so SHA982e357b7460326da77f98f1646c74bea3ed245d2bbcdd6c44f76ca00527a20f.
No production backend/default edits. Bridge requires TASK_QUEUE_ENABLE=0 before
process imports;matched AIV control also uses0. Native bridge submits contiguous
runs in a C++ loop via ACL;it is NOT MemCache.batch_copy or BigMemory.copy_data_batch.

14 CPU planner/ownership gates pass locally and in container;combined targeted
control62 passed. Review checked mapping validation outside timing,real save data
verification,retained owners,stream completion before descriptor release,64-bit
offsets,direction ABI assertions and nonzero submitted-copy counters. API/compile
success alone is not SDMA execution or resource contention evidence.

NPU0 on34 became idle;run_sdma_validation.py rechecked npu-smi immediately before
test. Real registered Host H2D->compute->D2H on separate streams:4 passed at
FP16/BF16 x64/512 widths,3 generations,noncontiguous rows,masked invalid row and
untouched-row checks. No new Host allocator needed for this roundtrip.
ABBA4-layer SFA128Kprefix/8Kchunk benchmark+trace now running;no performance claim
at this checkpoint. All tests/scripts/logs under results/prefill_overlap_20260917
and test/registered/unit/npu/{sdma_copy_lab.py,sdma_copy_bridge.cpp,test_sdma_copy*}.

## 2026-09-17 - First ACL-copy ABBA result rejects direct Host-pointer replacement

Same NPU0,TaskQueue0,both arms same4-layer128Kprefix/8Kchunk BF16 SFA;ABBA
orderAIV/ACL/ACL/AIV,7 samples per00/10/01/11,profiling outside timing. All outputs
bitwise identical SHA2234c2d0704f970c0e647c45e0fd910030c05ca02e587a9a7bc0b2ad95467703;
Host new rows poisoned then verified;Decode eager/Graph still pass.1 benchmark passed.
Combined14-sample medians(ms),AIV00/10/01/11=120.037/117.395/118.048/117.852;
ACL-host-pointer00/10/01/11=132.513/133.607/133.200/132.692. Candidate is slower.
Trace1_sdma/11 shows four SFA kernels but not the eight expected large copy tasks;
only one0.63us MEMCPY_ASYNC/SDMA_SQE,likely another framework operation. Therefore
do NOT call these figures proven SDMA offload/no-AIV contention. Investigating
whether original CPU pointer uses a synchronous fallback versus device alias or
Torch pinned Host control. Correct data alone does not qualify the async path.
Local raw artifacts:results/prefill_overlap_20260917/sdma_results/.

## 2026-09-17 - SDMA pointer contract counterexamples

128MiB H2D probe on idleNPU0:RegisteredHostRegion.host_ptr correct but host submit
median15.247ms,total~14-18ms;Torch pinned Host correct,submit median0.03896ms,
device event~1.4ms. This isolates original Host allocation registration as a key
problem for actual async SDMA. RegisteredHostRegion.device_ptr with D2D fails
ACL507899. Initial failure reporting crashed pytest when formatting tensor views
after shared region close;rerun with short trace captured explicit507899.

Tried additive aclrtHostRegisterV2(originalHost,size,ACL_HOST_REG_PINNED):register
returned0,subsequent ACL copy SIGSEGV inside bridge call. Independent test process
only;do not use additive registration in production. Reproduction now explicitly
gated by SGLANG_SDMA_REPRO_UNSAFE_PIN;no default test reruns it. Ordinary pointer
benchmark still pending separate trace classification. Production paths unchanged.

Next discriminating component arm uses extra Torch-pinned Host backing snapshot
while retaining same SFA/two-slot/events. It can demonstrate useful SDMA compute
overlap but is NOT drop-in Host sharing compatibility:extra full Host mirror and
export for assertions/Decode outside timing must be reported. ABBA+trace running.

## 2026-09-17 - DMA resource-contention hypothesis validated; allocator migration still needed

Pinned-staging ABBA completed on idle34NPU0. All4 arms x4 runtime modes match
output SHA2234c2d0704f970c0e647c45e0fd910030c05ca02e587a9a7bc0b2ad95467703;
Host writes and Decode eager/Graph after fixture export pass. Four-layer128K/8K,
BF16,16heads,topk2048,TaskQueue0,14 samples/backend/mode;profile outside timing.
Medians(ms) AIV00/10/01/11=121.013/118.070/118.546/118.273;
pinned-DMA00/10/01/11=80.941/76.284/80.465/75.786. Mode11 reduction35.92%;
DMA00->11 reduction6.37%. Improvement includes standalone transfer speed.

Actual A5 trace calls engineUBDMA:8 big MEMCPY_ASYNC tasks,4SFA tasks,no unidex
in candidate Prefill window. H2D~1.54-1.64ms/D2H~0.10-0.20ms overlap18.42msSFA.
Thus useful copy+SFA overlap and removal of AIV copy workload are demonstrated
for this component. No claim of same gain atTP8/full-model/1M or TaskQueue1.

Critical limit:positive arm uses extra545MiB Torch-pinned Host mirror and exports
mirror back to original registered Host OUTSIDE timing. Fixed contiguous plans
precomputed and metadata verified during warmup. It is NOT a production drop-in
replacement. Original registered host direct ACL path remains negative/slower;
device alias507899 and additive pin SIGSEGV are rejected. Need a unified qualified
shared pinned+FSA-visible Host allocator before production migration;MemFabric
shared pool remains a candidate,not tested as batch_copy replacement here.

Report:SDMA_PREFILL_VALIDATION_20260917.md. Raw positive artifacts:
results/prefill_overlap_20260917/sdma_pinned_staging_results/{timings.json,
timing_summary.json,sdma_11_trace.json,dma_timeline.json}. Existing negative
artifacts preserved in sdma_results.14 CPU checks,4 roundtrip cases,two ABBA
experiments completed. No service restarted or other process killed. Final
npu-smi reports all8 devices idle;all owned test processes exited.

## 2026-09-17 - Unified shared Host allocator probes; no qualified replacement yet

Continued when user reported34 idle;confirmed all8 idle and ran isolated probes.
Fresh SysV PINNED|MAPPED had previously registered successfully on248 but H2D
returned507899. Fresh PINNED-only on34 returned107000 during device-pointer lookup.
Separate248 probe then confirmed register0,lookup107000,and first DMA SIGSEGV,
without any earlier HAL registration. Thus failure is not limited to additive
registration;fresh PINNED DMA reproducer now requires explicit opt-in.

MemFabric LOCAL1GiB initialization on34 returned-1:missing launch extension
libmf_hybm_accoffload.so. Installed Python API alone is insufficient;no MemFabric
allocation was exercised. Runner adds package lib directory for libhcom but does
not replace installation. SHARED/TP attachment is not implemented in the lab;
adapter now explicitly rejects inherited SysV attach/export.

34 became occupied by Ray/VLLM jobs;248 subsequently started another TP8 service;
244 also occupied. Launch guards refused follow-up AIV probes. No jobs killed.
9 CPU ownership tests pass;no TP/FSA/Graph/no-mirror performance qualification.
Production allocation/defaults unchanged. Detailed report:
UNIFIED_HOST_VALIDATION_20260917.md;local results/unified_host_20260917;
isolated remote root /home/j00628475/unified_host_validation_20260917_v1.

## 2026-09-17 - MemFabric allocation dependency boundary clarified

Read-only audit of actual1.2.0 Python/native implementation confirms offload LOCAL
and SHARED eagerly require libmf_hybm_accoffload.so before creating/reserving the
pool. Loader resolves AccOffloadSparseCopy and AccOffloadGroupPackCopy even for
allocation-only clients. In contrast libmf_hybm_copy_extend.so loading is optional
for core initialization: its loader result is ignored; particular extended-copy
paths still require its function pointers. Do not claim all extend components
are universal allocation prerequisites or that all MemFabric allocation is broken.
Lower-level HYBM HOST allocation exists and is a possible adapter route;DMA/FSA/TP
compatibility remains unvalidated. Exact release20260904 pristine image inspected
on248 contains1.2.0 and lacks the offload launcher extension;existing pxc container
has1.2.1 after image creation. No NPU or service changes during this audit.
Details:MEMFABRIC_ALLOCATION_DEPENDENCIES_20260917.md and raw loader branch logs.

## 2026-09-17 - A5 offload branch built; extension loader passes, hardware pending

Reviewed clean local D:/newCode/memfabric/memfabric_hybrid br_feature_a5_offload
fd33448651a6bacc0b89ad2e9344cb2f0e6e069d. LOCAL/SHARED Host allocation and extension
sources present. SHARED supports rank0 physical memory/peer mappings;get_dva is
LOCAL-only,SHARED uses GVA. Default copy remains MTE/AIV64,not proven SDMA.
Built complete host/Python artifacts and both A5 extensions on34 in isolated root
/home/j00628475/memfabric_a5_fd334486_20260917_v1. Only remote shell CRLF normalized.
No local MemFabric source changes,global pip install,or .run installer execution.
CPU actual offload-extension loader now returns0;maps confirm candidate libraries;
NPU remains uninitialized. Original installed package and serving defaults intact.
9 existing CPU harness tests pass. All34/248/244 cards became occupied;hardware
test launch refused. Actual pool/DMA/FSA/Graph/TP/performance remain pending.
Detailed capability limits,artifacts and resume command:
MEMFABRIC_A5_BRANCH_VALIDATION_20260917.md.

## 2026-09-17 - MemFabric unified allocation passes LOCAL DMA/FSA and shared TP access

Real34 initialization found missing CANN whitelist symbol ra_socket_setwhite_list_status
orRsUpdate. Applied explicit2-file compatibility patch in local MemFabric repo:
optional symbol load;HostURMA OpenDevice fails-8 if unavailable. Native CPU probe
confirms load success and fail-closed HostURMA behavior. LOCAL MTE pool now init0.
TwoLOCAL pool cycles x3generations AIV read/write pass;8MiB same-allocation DMA
H2D/D2H+AIV passes3generations. Added external allocator registration to standalone
FSA binding,checking MemFabric range mapping endpoints and device;unregister before
free with users drained. New LOCAL fused cold/warm/reshuffle/Graph passes,registration
negative cases pass,and original SysV FSA with new binding regression passes.

TP2/TP8 shared GVA AIV read/write+Graph+realHCCL collective pass,one rank0 physical
pool and0bytes on peers. Initial peer CPU dereference test segfaulted:imported
GVA is device-only on peers,so corrected checks only read CPU on owner. This
boundary must carry into future SGLang allocator integration. TP2 shared DMA
reads using mapped-address D2D also pass Graph/HCCL;TP8 DMA subsequently passes
the same3generation Graph/HCCL and peer-write visibility test on all8ranks.

No-mirror4layer128Kprefix/8Kchunk BF16,16heads,topk2048,2slots,TaskQueue0,ABBA
7samples/arm/mode =>14samples/backend/mode. AIV00/10/01/11 ms:
110.059/107.949/108.232/107.952;DMA81.569/76.556/80.854/76.405.
Mode11 time reduction29.22%. Both arms use MemFabric,no extra pinned Host mirror,
no deferred export. All outputs share SHA2234c2d0704f970c0e647c45e0fd910030c05ca02e587a9a7bc0b2ad95467703.
Trace8UBDMA+4SFA,no unidex;7copies overlapcompute,firstH2D precedescompute.
H2D1.45-1.50ms,D2H0.086-0.180ms,SFA18.42-18.43ms/layer.
Single-device component benefit,not TP8performance/E2E/1M/TaskQueue1 proof.
Production installed package/defaults remain unchanged;oldwheel/.run are unpatched,
validated candidate uses staged compatibility core and separate FSA binding.
See MEMFABRIC_A5_BRANCH_VALIDATION_20260917.md,results/unified_host_20260917/
memfabric_validated_results;14CPU tests also pass local andremote.
