# GLM-5.3 Flash A5 offload migration

Target baseline: `23d5abcf9c403b6537150f4303f2cad7257e6770`.
The current deployment target is one TP8 service running both Prefill and Decode.
The separately implemented P/D services remain available as a different mode.
The baseline's NPU KPool, KDA, speculative execution and MemFabric Ascend
transport remain the integration source of truth.

## Latest qualification (2026-09-15)

Latest additional completion gate: `bench_1m_1024_v1` on the restored34 single
TP8 offload service completed exact1,000,000 input tokens +1,024 generated tokens.
No profiling/restart; TTFT328.792s, TPOT54.656ms, total384.705s, exit0,
no retraction and no prefix-cache hits. MTP0, Decode Graph on; sampling forces
1024 outputs with ignore_eos. This confirms execution/completion, not model
quality at1M. Results and timing definition are in `FSA_LONG_CONTEXT_RESULTS.md`.

### Latest:500K/900K performance and memory comparison

See `FSA_LONG_CONTEXT_RESULTS.md` for the complete result and limitations.
On34 a single TP8 service at1048576-token capacity completed exact500000 and
900000-token prompts with128 output tokens, no retraction, MTP0 and Decode Graph.
Unprofiled offload repeat results:500K TTFT111.745s/TPOT49.285ms;900K mean
TTFT272.276s/TPOT52.165ms.900K stage traces prove real `unidex_copy` Prefill
offload/restore and22 `FusedSparseAttentionOverlap` tasks over two Decode steps.
The1M Host KV has11 shared segments,8attachments each,11.001GiBtotal.

Matched resident service (offload/fuseddisabled) also completes both workloads:
500K TTFT95.888s/TPOT38.566ms;900K217.784s/41.462ms. Resident gets one sample per
length, so this is a limited comparison, not a statistical benchmark. Payload
hashes match exactly. Measured benefit is HBM, not speed: KV13.752->4.751GiB/rank;
8-card sampled HBM502.064->430.267GiB at500K and502.065->430.712GiB at900K.
Host PSS rises by approximately11..13GiB. Both paths fit1M, no OOM claim.

The32K service and benchmark offload/resident services were stopped only after
checking their ownership. Restoration uses `offload_1m_restored_v1` on33195 with
the same1M offload configuration; driver exit0 and health200 confirmed at23:03 CST.
Active parent150593/start_ticks2721714; service remains running.
Old service PIDs/ports in historical sections below are not the active endpoint.

### Single-service TP8 smoke qualified

The user clarified that mixed deployment means one non-disaggregated TP8 service,
not two P/D services on the same host. `SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=1`
now selects a local lifecycle: logical DSA Host KV is allocated once per TP
group, Prefill restores prefixes into two reusable HBM layer slots and writes
new KV to the same Host backing consumed by fused Decode. KPool/KDA stay on NPU.
The ordinary request allocator owns physical token rows across both phases;
no PD sender, receiver, transfer store or router is involved. Decode Graph
capture/replay stays enabled; bounded Prefill remains eager. Capacity accounting
includes both resident indexer and Prefill HBM slots without duplicating Host KV.

On34 at `/home/j00628475/fsa_glm53_pd_20260915`:

- `integrated_control_v1.log`:230 passed11.18s, exit0.
- `integrated_hardware_v1.log`:FP16/BF16,2 passed15.80s, exit0; four logical
  layers/two HBM slots, cross-chunk prefix restore, Host-visible new KV and
  Decode Graph write subsequently visible to Prefill restoration.
- `integrated_tp2_v1.log`:same lifecycle with actual TP2 shared Host backing,
  two cases per rank passed, about16s, exit0.
- `integrated_run_v1`: single-service TP8 startup/request gate launched with
  a fresh all-device occupancy check. All weights loaded (467s,37.98GiB/rank),
  then the budget builder failed on the removed `ServerArgs.cuda_graph_bs`
  field. Replaced old field reads with published
  `get_exec().graph.cuda_graph_config.decode` backend/bs/max_bs. The service
  exited and its own framework cleaned up its process tree; no other job stopped.
- Control UTs with module stubs remain230 passing. Real runtime configuration
  tests run in their own process to avoid those stubs, publishing an actual
  ServerArgs config first. `integrated_runtime_config_v4.log`:2 passed10.98s,
  covering Graph enabled/disabled with no old ServerArgs attributes. Earlier
  v2/v3 test failures were fixture isolation/publication defects, now resolved.
- `integrated_run_v2` is the subsequent full-model startup/request attempt.
  Weights loaded in approximately96s. KV allocation succeeded (32768 tokens,
  0.15GiB reported per rank) and all eight ranks completed Decode Graph capture
  in approximately28s. HTTP `/model_info` returns200. SysV inspection confirms
  eleven33685504-byte Host KV segments, each created by TP0 PID34693 with eight
  attached processes (`shared_host_segments.txt`). JIT cache paths now point
  inside the task root.
- The initial `/health` request generates a token, so it triggers cold Prefill
  KDA Triton compilation/autotune. Saved `stack_tp0_startup/second/third/fourth.txt`
  show progress through `chunk_kda_fwd_intra`, `recompute_w_u_fwd`, then
  `chunk_gla_fwd_o_gk`; compiler subprocesses continue changing. The20s health
  timeout is firing during this work. This is not evidence of an FSA kernel
  hang, and is not yet a successful generation result. Keep waiting/inspect
  current stacks before restarting; the launcher has not yet sent its smoke
  requests. Process34446 is the owned service parent; all logs remain in
  `integrated_run_v2`. No success claim until real responses arrive.

Final `integrated_run_v2` outcome: driver exit0, manifest `smoke_passed`;
service remains running at `http://127.0.0.1:33181` inside the34 container.
Parent PID34446, source is the dedicated task tree, TP8/DP1, disaggregation=null,
MTP0, fused mode, TP-shared Host enabled. `/health` returned200 after cold KDA
compilation completed. The saved compiler stacks explain the earlier health
timeouts; no restart or kernel workaround was needed for that stage.

| Actual request | Prompt tokens | Output tokens | Result |
| --- | ---: | ---: | --- |
| Raw short `/generate` | 10 | 64 | HTTP200, no retraction; starts with correct answer5 |
| Raw long `/generate` | 10417 | 64 | HTTP200, two Prefill chunks, Decode Graph; repetitive continuation |
| Chat short | 34 | 27 | Final answer exactly5 |
| Chat long | 10441 | 33 | Final answer exactly7391, two Prefill chunks |

Chat evidence is `integrated_run_v2/chat_v2/{chat_summary.json,chat_response_0.json,
chat_response_1.json,exit}`. Warm chat request wall times were1.541s and3.124s;
these are whole requests with different lengths, not TPOT/performance benchmarks.
Raw-completion behavior is not an instruction-following correctness oracle.
The service has no reasoning parser configured: the model emits reasoning plus
`</think>` in content even with the template option requesting no thinking.
The first chat checker incorrectly required the entire content to be just the
number; both actual final answers were correct. The corrected checker preserves
raw responses and compares the text after `</think>` exactly with the expected
number. Fresh requests passed both checks, exit0. This is basic semantic smoke,
not a model accuracy benchmark.

The actual long request crosses8192-token Prefill chunks, so the same-service
Host prefix restore and subsequent fused Decode were exercised end to end.
This qualification uses32K configured capacity and MTP0.1M admission/execution,
MTP acceptance, multi-request stress and performance comparisons remain untested.

Launch one service inside the assigned container:

```bash
export SGLANG_FSA_WORK_DIR=/home/j00628475/fsa_glm53_pd_20260915
python3 "$SGLANG_FSA_WORK_DIR/sglang/scripts/npu/run_fsa_integrated_e2e.py" --launch \
  --run-dir "$SGLANG_FSA_WORK_DIR/integrated_$(date +%Y%m%d_%H%M%S)"
```

Defaults: physical0..7, TP8/DP1, FP8 weights/BF16 KV,32K context/token capacity,
8K Prefill chunks, maxreq1, Graph1/2/4, MTP0,16GiB shared Host budget,
port33181 and distributed port29611. No disaggregation arguments are passed.
Scripts and source are synchronized to34; alternatives need the new integrated
changes before use. NPU voltage alarms are allowed per user instruction;
occupied devices still prevent launch. Historical same-host P/D planning below
does not describe this selected single-service topology.

The framework migration and pre-E2E control/native tests are implemented.
Final qualified pairing on34:

- Source: `/home/j00628475/fsa_glm53_a5_20260915/sglang`.
- Wrapper/helpers: `native-build-v5/libsgl_fsa_extension.so` under the task root.
- Vendor: `attention-opp-v8/install/packages/vendors/sgl_fsa_a5` under the task root.
- `logs/attention_ut_v22.log`:54 passed in20.41s, exit0.
- `logs/control_ut_v11.log`:226 passed in9.89s, exit0.
- `logs/final_env_gate_v2.log`: both role environments, selected source imports,
  installed kernel wheel path and standalone ABI passed, exit0.
- Actual framework import gate previously passed; installed sgl-kernel-npu
  remains in site-packages. No wheel replacement, production checkout overwrite,
  container restart or service launch was performed.

The54 hardware cases cover24 fused and24 gather/reference cases, FP16/BF16,
batch1/4, heads16/64/128,2051 columns with varying valid counts and0..3 tail
tokens, cold/warm/reshuffle, Graph and request-owner changes. Two MTP cases use
2 requests x3 verification tokens and rewrite mutable tails without changing
request identity. Four pool cases cover real Prefill4-layer/2-slot reuse with
prefix restore and resident KPool, plus Decode current-token Graph writes and
64KiB staging-to-Host copy split across native32KiB copy calls.

These are pre-E2E component gates. Real MemFabric P/D request exchange, router,
full model/MTP acceptance,1M context admission and TPOT remain end-to-end tests.
The Prefill component test injects transfer acknowledgements locally. It does
not prove transport or production request completion. Full-model execution on
the shared machine must be scheduled separately; the migration did not start it.

The historical failures below are retained for diagnosis; the final pairing
supersedes OPPv4-v7 and wrapperv4. The small UB separation fix inOPPv7 eliminated
the cross-call corruption in the subsequent full boundary matrix. No precision
threshold was relaxed.

## Running the migrated source

TP-shared Host backing is now enabled by default after real A5 TP2/TP4
qualification. `SGLANG_NPU_SHARED_HOST_KV=0` retains independent per-rank
allocations. Prefill and Decode each share within their own attention TP group;
they do not use one combined P/D allocation. Only DSA full latent KV is shared.
Selection caches, planner state, indexer KPool and KDA remain per-rank NPU state.
Ranks must be on the same host and IPC namespace, with replicated KV and DCP=1.
Attention TP spanning hosts must explicitly opt out; separate P/D groups on
different hosts are compatible with node-local sharing in each group, but real
cross-host transfer remains an E2E qualification gate.

Use the container's existing validated model/TP/MTP/network launch settings,
with separate `--disaggregation-mode prefill` / `decode` processes and disjoint
NPU assignments. Require `--attention-backend ascend`,
`--disaggregation-transfer-backend ascend`, `--disable-radix-cache`, and
FP16/BF16 KV storage. Disable optimistic Prefill and HiSparse. Bounded Prefill
supports replicated CP; DCP/layer-sharded KV is rejected. Model weights may use
the baseline FP8 quantization, but Host KV must be FP16/BF16.

Set explicitly in each service shell:

```bash
export SGLANG_FSA_WORK_DIR=/home/j00628475/fsa_glm53_a5_20260915
export SGLANG_NPU_FSA_EXTENSION="$SGLANG_FSA_WORK_DIR/native-build-v5/libsgl_fsa_extension.so"
export SGLANG_FSA_OPP_VENDOR="$SGLANG_FSA_WORK_DIR/attention-opp-v8/install/packages/vendors/sgl_fsa_a5"
```

For Prefill, set `SGLANG_NPU_PREFILL_HOST_KV_BYTES` to its Host backing budget,
then `source scripts/npu/fsa_offload_env.sh prefill`. For Decode, set
`SGLANG_NPU_DECODE_HOST_KV_BYTES` and positive32-byte-aligned
`SGLANG_NPU_LAYER_STAGING_BYTES`, then source the same file with `decode`.
These scripts select the repository containing the script via PYTHONPATH and
preserve existing vendor entries. They do not start processes. The original
`/home/j00628475/sglang` is unchanged. Offload and fused-attention defaults remain
off outside this explicit environment. Enabling shared Host backing by default
does not enable offload itself.

Size Host budgets jointly across P/D groups. Each rank supplies the same budget;
with sharing enabled, the group's backing is allocated once, not once per rank.
Explicit opt-out restores per-rank allocations and their multiplied Host cost.
Keep the baseline MemFabric protocol and store
configuration; the helper/attention tests do not qualify a new transport mode.
Alternate248/244 are authorized for UT; they need the final source/OPP/helper
pair synchronized before reuse. Both contain the MemFabric Python package and
its libraries; validate their transport configuration before PD tests.

## Execution plan

### Guarded single-host E2E launcher

`scripts/npu/run_fsa_pd_e2e.py` and `fsa_pd_role.sh` provide an isolated
same-host Prefill TP4 (physical0..3), Decode TP4 (physical4..7), and router.
They reference the existing FP8 GLM-5.3-Flash service's model and communication
settings. No existing process is stopped. Launch requires all eight NPUs to
be idle and all selected ports to be free. The user confirmed the observed
voltage-adjustment alarms do not prevent operation; those alarms no longer
block this launcher's resource gate. Check-only mode imports
the actual isolated source/native extension, parses role CLI arguments and
checks the router package without loading model weights or creating NPU work.

The dedicated deployment root on34/248/244 is
`/home/j00628475/fsa_glm53_pd_20260915`, containing its own source and the same
qualified wrapperv5/OPPv8. To check or launch inside the assigned container:

```bash
export SGLANG_FSA_WORK_DIR=/home/j00628475/fsa_glm53_pd_20260915
python3 "$SGLANG_FSA_WORK_DIR/sglang/scripts/npu/run_fsa_pd_e2e.py" \
  --run-dir "$SGLANG_FSA_WORK_DIR/check_$(date +%Y%m%d_%H%M%S)"
python3 "$SGLANG_FSA_WORK_DIR/sglang/scripts/npu/run_fsa_pd_e2e.py" --launch \
  --run-dir "$SGLANG_FSA_WORK_DIR/run_$(date +%Y%m%d_%H%M%S)"
```

Initial gate uses32K context/token capacity,8K Prefill chunks, BF16 KV,
FP8 weights, one active request and MTP disabled. Host budget is16GiB per
role TP group's shared backing; Decode staging is64MiB per rank. Prefill
uses two physical DSA layer slots; Decode uses fused mode and4096 resident
slots. P/D/router ports are31181/32181/6681, bootstrap8991, distributed
ports29581/29591 and MemFabric store24681. Graph batch sizes are1/2/4.
Automatic server warmup is skipped so initial requests are issued through the
router only after both roles are ready. Two smoke requests include a short
question and a long passage intended to cross an8K Prefill boundary; verify
the returned `prompt_tokens` before calling this a multi-chunk result.

After this gate passes, use a fresh run directory and set `FSA_MTP_STEPS=3`
for MTP, then raise `FSA_CONTEXT_LENGTH` and `FSA_MAX_TOTAL_TOKENS` for long
context admission. A32K smoke result does not qualify1M execution. Role command
lines, source paths, loader paths, responses, PID/start times, resource reports
and failure status are written below each run directory. A failed launch does
not automatically terminate any remaining processes; inspect recorded ownership
before cleanup. Use the guarded Python entrypoint rather than launching role
scripts directly.

The transport defaults to the reference environment's effective SDMA plus
`MF_HYBM_RDMA_FORCE_UNREGISTERED=1`, with a new isolated config-store port.
The existing reference shell's comments claiming DEVICE_RDMA disagree with its
actual export. Real transfer must still be qualified; environment import success
is not a MemFabric transfer result.

Historical 2026-09-15 20:24 CST preflight outcome on all three authorized hosts:

- Isolated source/helpers/vendor deployed and archive SHA256 verified on34,
  248 and244. Each `check_v1` passed Prefill/Decode CLI parsing, standalone
  extension loading and router import/help. Installed sgl-kernel-npu is still
  selected from site-packages. Both offload roles resolve shared Host toTrue.
- All three checks then exited1 at the resource gate, with an empty process
  manifest: no P/D/router service was launched.34 physical0/3 and248 physical1
  are idle but report `80E3A201`, voltage adjustment function abnormal.244 has
  all eight healthy cards occupied by another service. Do not count unused
  memory or zero instantaneous utilization as an idle device.
- Evidence is in each host's `check_v1/{manifest.json,npu_preflight.txt,
  prefill_check.log,decode_check.log,router_check.log}`, `check_v1_driver.log`
  and `check_v1.exit`. No unrelated service or container was stopped; no NPU
  reset, driver change or alarm clearing was attempted.
- Pending: rerun the guarded launcher on an idle healthy eight-card host,
  establish P/D readiness and real MemFabric transfer, then complete the smoke
  requests. MTP3 and1M remain subsequent gates. No background auto-launch watcher
  is left running; resource availability must be rechecked when work resumes.

Subsequent user correction: allow the observed NPU alarms and gate on process
occupancy. The updated gate is synchronized to all three dedicated E2E trees.
The next check found34 and244 idle,248 occupied. Launch topology needs explicit
resolution: the prior definition was two separate P/D services on one host,
whereas a single TP8 service serving both phases is a different execution mode.
Current `offload_config.validate_offload_service` rejects non-disaggregated roles;
`maybe_init_prefill_layerwise_offload` also requires a Prefill service and the
Prefill runtime requires layer-ready transfer senders. A single integrated
service therefore requires lifecycle adaptation, not just omission of PD flags.
No service has been launched pending this topology clarification.

1. Recover the prior offload runtime and its control tests; adapt NoPE,
   2051-wide indices and hybrid-layer ownership to this baseline.
2. Integrate bounded Host-backed KV allocation, current-token writes,
   selection-cache lifetime and fused/reference attention dispatch.
3. Connect Prefill layer staging and Decode receive/publication to the existing
   Ascend MemFabric transport; preserve KPool and KDA state transfers.
4. Supply an optional standalone native extension/OPP build and explicit loader
   without replacing the container's installed sgl-kernel-npu wheel.
5. Run control and integration UTs in the specified container, then focused
   hardware UTs only on a currently idle healthy device. Record exact gates and
   remaining end-to-end qualification.

## Environment and isolation

- Host: `141.61.55.34`, container: `jzx_sglang_5.3_flash`.
- All remote work and test artifacts belong under `/home/j00628475/`.
- Production source baseline: `/home/j00628475/sglang`; tests use an isolated
  task copy until synchronization can preserve concurrent session work.
- Use installed Torch, Torch NPU and sgl-kernel-npu. An OPP vendor alone does not
  register a new Torch operator: a small optional binding is also required.
- Do not restart services, remove caches globally, or terminate other jobs.
- Initial inspection found no service/NPU processes. Cards 0 and 3 reported
  Alarm; do not select them for UT. Recheck occupancy before every hardware run.

## Current evidence

- NoPE operator donor: sgl-kernel-npu commit `539e7c9`; previous real-device
  validation covers Ascend910_93 only, not A5.
- Full lifecycle donor: SGLang commit `f291e348ab`. Its GLM-5.2 assumptions and
  native-extension dependencies require adaptation; it is not an A5-tested port.
- This baseline already implements `IndexerKPool.forward_npu`, KPool cache
  updates, expanded token indices and MTP tail handling. Do not replace it with
  the older incomplete `kpool_reference.py` integration.
- No new E2E service has run. Hardware evidence below distinguishes helpers
  from attention qualification.

## 2026-09-15 framework and control-test stage

Migrated the `f291e348ab` lifecycle modules and adapted the baseline's actual
AscendDSA backend, request-generation tracking, NPU MLA pool, hybrid pool mapping,
model-runner entry points, and Ascend/Mooncake layer protocol. The feature remains
opt-in. No KPool Indexer source or KDA kernel was replaced.

- NoPE uses no registered RoPE allocation. Expanded indices retain all2051
  columns, including tail and padding. Per-layer validity remains separate from
  shared slot placement. Request-pool clear increments an epoch.
- Prefill latent storage reuses bounded layer slots. KPool index storage stays
  per logical layer and is excluded from Host save/restore, avoiding overwrite
  of newly compressed entries by a token-indexed restore.
- Layerwise transport handles latent/RoPE. The baseline's generic transport
  retains KPool index buffers and KDA/tail state; it skips already-published
  latent pointers. Whole-request completion still waits for layer ACKs.
- `fsa_native.py` loads an explicit `SGLANG_NPU_FSA_EXTENSION` in the separate
  `sgl_fsa` namespace. The installed wheel has no registered-Host/unidex/FSA APIs
  in this container, so the small extension includes these helpers too.
- Remote isolated checkout: `/home/j00628475/fsa_glm53_a5_20260915/sglang`.
  `/home/j00628475/sglang` and container-installed wheel remain unchanged.
- `logs/control_ut_v5.log`:222 CPU tests passed in12.49s.
- `logs/control_ut_v6.log` and `logs/import_gate.log`: processes exited137.
  Container start time changed; Docker reported OOMKilled=false and abundant
  Host memory. No service was stopped or container restarted by this migration.
- `logs/control_ut_v8.log`:225 CPU tests passed in10.04s, exit0, including
  production-method AST isolation and explicit KPool/KDA transfer preservation.
  These tests are control/CPU oracle evidence, not A5 execution or PD DMA proof.
- `native/fsa` is a separate CMake target. Helper sources were exported from
  sgl-kernel-npu539e7c9 to `kernel-source` under the isolated work directory.
  Configure/build logs are `logs/native_configure.log` and `logs/native_build.log`.

## 2026-09-15 standalone A5 build and pre-E2E qualification

- `logs/control_ut_v9.log`:225 passed in10.57s, exit0, including the latest
  strict-abort ownership and bounded copy changes.
- `logs/import_gate_v2.log`: actual AscendDSA, FSA backend, Ascend/Mooncake
  transport and model-runner imports all succeeded, exit0.
- `native-build-v4/libsgl_fsa_extension.so`: built for Ascend950PR_9599, loaded
  ABI1 with9 schemas; installed sgl-kernel-npu wheel unchanged.
- `logs/native_helpers_npu.log`:3 passed in12.50s on34/device1. Real FP16/BF16
  registered-Host roundtrip and graph replay, plus2051-entry native planner
  reshuffle and owner change. These tests do not execute attention.
- `native/fsa_attention` vendors the arch35 device sources and dependencies.
  NoPE adaptation removes RoPE DMA, uses512-wide MM1 with576 workspace row
  stride, and compiles both split-G selections. The local Buffer compatibility
  setter matches the installed CANN9.1 implementation.
- `attention-opp-v4`: host libraries, FP16/BF16 device binaries and isolated
  `install/packages/vendors/sgl_fsa_a5` built successfully, exit0. The build
  uses installed CANN templates, not a wheel or a full operator-repository build.
- Failed build stages: v1 omitted opbuild include paths; v2 used incompatible
  legacy logging; v3 lacked Buffer.SetCrossCoreID. Also, the CANN template's
  default target does not compile device binaries: build target `binary` before
  installation. These failures are resolved in v4; do not repeat these variants.
- Attention UT added as `test/registered/unit/npu/test_fsa_attention_a5.py`:
  FP16/BF16, batch1/4, heads16/64/128,2051 columns with0..3 tails; CPU attention
  oracle, cold/warm, poisoned Host source to prove selection reuse, reshuffle and
  graph. It has NOT passed hardware qualification yet.
- User authorized alternate hosts141.61.54.248 and141.61.54.244, same container
  and working directory. Both initially idle.248 received an isolated runtime
  copy at the same task-root path. Its first attention UT failed in setup:
  `invalid npu option name:ACL_OP_EXEC_TIMEOUT`, before attention invocation.
  Removed that option, set runner `ACL_DEVICE_SYNC_TIMEOUT=10` instead.
- At the subsequent check,248 and244 had other jobs on all8 NPUs. Do not rerun
  hardware tests until an idle healthy card is observed. The runner checks this
  immediately before pytest. No other process was stopped.
-248 has matching Torch2.10.0+cpu/Torch NPU2.10.0.post4/CANN9.1 but lacks
  `/usr/local/memfabric_hybrid`. This path check alone does not establish that
  the MemFabric Python package is missing; see final environment qualification.
- Local focused control regression:117 passed,8 skipped in3.28s. The8 skips
  are explicitly gated A5 attention cases and provide no hardware evidence.
- Reproducible op-only build and attention runner are under `scripts/npu/`.
  Both shell scripts passed `bash -n` in248. Runtime preflight refuses devices
  that are occupied, unhealthy or not unambiguously reported by npu-smi.

Remaining gates: actual A5 attention and external-plan precision/graph UT;
hybrid P/D/KPool state transfer and abort behavior with real components; final
runtime configuration and source synchronization preserving concurrent changes.
No end-to-end or performance claim follows from build/control/helper tests.

## Latest continuation: adapter contract and244 deployment

- `logs/control_ut_v10.log` on34:226 passed in11.50s, exit0. Added an actual
  `forward_fsa` adapter test with hybrid logical layer7 mapped to local Host
  layer1, two requests with two verification tokens each, per-query causal
  masks, original model scale and zero-width RoPE. The attention call itself
  is intercepted in this CPU control test; it is not kernel precision evidence.
-244 briefly became idle and received the isolated runtime archive at the
  same task root. Other services acquired its NPUs during the upload.
  `logs/attention_ut_v2.log` records preflight refusal before pytest:
  `NPU 1 is occupied, unhealthy, or could not be verified; UT not started`.
  A follow-up npu-smi confirmed scheduler processes on all8 devices. This is
  resource contention, not a kernel failure. Do not retry while occupied.
- Latest scripts/report/test were synchronized back to34's isolated checkout.
  All three production service checkouts and installed wheels remain untouched.

## A5 execution findings: invalid-index padding

-34 became idle. Attention UTv3 reached metadata materialization and failed
  because A5 aclnnArange does not support DT_INT16. Generate int32 row IDs and
  cast through copy_ into the int16 metadata instead. No planner ABI change.
- UTv4/v5 with the first attention binary returned output but failed FP32-oracle
  precision: max error0.0022846. Native SFA on identical inputs was bitwise equal
  to FSA. Selection-cache writes matched source KV exactly. CPU reconstruction
  adding duplicate tokens1664..1666 to the2048 valid tokens matched the output
  within0.0000171: the final3 invalid entries consumed stale triple-buffer data.
- `attention-opp-v5` fixes invalid-index handling: preserve column positions,
  zero invalid gathered rows, continue after -1 rather than stopping the lane,
  mask corresponding MM1 logits before softmax, and add required scalar/vector
  and vector/DMA dependencies. This also supports padding shuffled to the front.
- `logs/attention_ut_v6.log`:1 passed11.66s,FP16,batch1,heads16,TopK2051 with
  2048 valid entries and3 invalid pads. Cold/warm/reshuffle/Graph replay passed,
  including poisoned Host source after the first call. FSA max FP32 error
  0.0000198744; unchanged container SFA error0.0022846. Thresholds unchanged.
- `logs/attention_ut_v7.log` is the subsequent8-case matrix; inspect its actual
  exit file before rerunning or claiming all cases passed.

## A5 matrix follow-up: unresolved cross-call and split-G behavior

-v7 andv9:FP16 first case passed, then BF16 cold output failed on the upper
  half of heads (max error0.05864). BF16 alone(v8) passed, max error0.0001624.
  Two-case runsv10(with native-SFA diagnostics) andv11(with failure artifact
  capture) passed without kernel changes. This variability remains unresolved;
  do not conclude a dtype fix or use separate processes to hide the failure.
-v12:6 passed,1 failed in14.39s. Passed FP16/BF16 with(1,16,tail0),
  (1,64,tail3),(4,16,tail1), including warm/reshuffle/Graph. The(1,128,tail2)
  FP16 case hit507015; the final BF16 case was not executed due to fail-fast.
- Driver log forv12:`tmp/ascend/log/debug/plog/plog-14213_20260915175450305.log`.
  Reports `timeout or trap error`,subErrType0x4, multiple cores. Kernel dump
  exists under `tmp/ascend/extra-info/data-dump/1/`. No measured claim of an
  out-of-bounds access. Inspect split-G global mode0 flag15 participation and
  loop counts; the cause has not yet been established.
- Current code/kernel remains v5 (invalid-column fix), native helper v4.
  Matrix failures prevent declaring attention qualification complete.

## Final fixes and qualification

- OPPv6 vector Duplicate mask failed with explicit `VEC to access UB is not
  aligned` and was rejected. Do not repeat unaligned vector stores.
- OPPv7 separates `selectionPlanTBuf` from softmax `commonTBuf`. Previously
  plan DMA/scalar updates could overwrite the overlapping vector softmax
  workspace. The subsequent8-case matrix(v15) and24-case boundary matrix(v16)
  passed, including same-process dtype transitions and split-G128 heads.
- Reference mode now compacts valid slot indices and supplies per-query valid
  lengths to native SFA. This avoids the native SFA padding behavior already
  observed with the unmodified2051-column input. Reference24-case matrix(v18)
  passed with the same FP32 precision gate.
- Real pool tests:v17 Prefill2 passed15.53s;v19 Decode2 passed12.38s. The latter
  covers zero-width RoPE registration, current-token graph writes, skipped
  dummy location0 and a64KiB staging copy to Host split into32KiB native copies.
- MTPv20 exposed the donor wrapper replacing cumulative query lengths when
  request count differed from query rows. The local wrapper now preserves the
  provided cumulative lengths; missing lengths use arange(1,T+1), not ones.
- OPPv8 processes the full sparse-index column count even for short context;
  individual token IDs are checked against per-query causal bounds. This keeps
  KPool tail entries in columns2048..2050 when physical sequence length is256.
  MTPv21 passed both dtypes13.35s, including a mutable-tail rewrite during Graph
  replay without owner change. Finalv22 confirms all54 cases with wrapperv5.
- The wrapper is now source-controlled under `native/fsa`, while allocator,
  copy and planner helper sources still come from the pinned539e7c9 export.
- Local-vs34 isolated source manifest compared102 files: no mismatch. Final
  environment test selects isolated SGLang, site-packages sgl-kernel-npu, ABI1
  standalone extension and both role gates; it does not launch a server.
- MemFabric is installed as a Python wheel on34. Absence of the legacy
  `/usr/local/memfabric_hybrid/set_env.sh` is not a missing-dependency proof.
  The environment script sources it when present, then verifies TransferEngine
  import, allowing the container's wheel-only layout.

Final artifact SHA256:

- `native-build-v5/libsgl_fsa_extension.so`:
  `9f922d0327640e0b0b7ba896cf4f257cf59076cfc4773b089f8c49cd974d19da`.
- OPPv8 `op_api/lib/libcust_opapi.so`:
  `93a5606ca7b3241e17bd876b93b9431a73115ffdc805739bd6353b5ba27e0056`.

## TP-shared Host qualification and default (2026-09-15)

- Real tests ran on34, in the assigned container and isolated checkout, using
  idle physical NPUs. No unrelated services, installed wheels or production
  checkout were changed.
- `logs/shared_tp_v3.log`: TP2 on devices1,2, explicitly enabled;6 cases passed
  on each rank in17.27s, exit0. `logs/shared_tp_v4.log`: TP4 on devices1,2,4,5,
  environment variable unset;7 cases passed on each rank in approximately20.8s,
  exit0. Counts describe cases executed by each rank, not additional distinct
  scenarios for every rank.
- FP16/BF16 cases verify identical SysV segment identity across ranks, exactly
  one writer despite conflicting nonwriter inputs, real HCCL publication, Graph
  replay, and continued peer reads after the writer detaches. Prefill exercises
  four layers/two HBM slots and prefix restoration. Decode exercises fused
  attention with an independent FP32 CPU oracle and the real TP output all-reduce,
  including Host updates and selection-cache reloads across Graph replays.
- Default-True and explicit opt-out contracts and DCP rejection are covered.
  `environ.py` now defaults shared Host KV toTrue and `fsa_offload_env.sh` to1.
  Actual backing ownership now gates Decode's shared abort-drain collective;
  ordinary and explicitly unshared Decode do not acquire this collective merely
  because the environment default changed. Cross-rank drain remains mandatory
  when shared backing exists, including when a local deferred queue is empty.
- `logs/control_shared_default_v2.log`:228 control cases passed in10.51s, exit0,
  with the shared environment variable unset. Includes shared drain retention,
  unshared backing and non-offload regression cases.
- `logs/shared_tp_env_gate.log`: Prefill/Decode role scripts both select1 when
  unset and preserve explicit0. Both changed shell scripts pass `bash -n`.
- Rejected launcher settings: torchrun `--standalone` stalled on container
  hostname resolution; use explicit localhost IPv4 rendezvous. HCCL connect
  timeout90 is invalid on this runtime; use120 (supported range120..7200).
  These were launch/configuration failures, not failures of shared Host storage.
- This validates shared Host functionality at TP component/Graph level. It is
  not full-model, real PD transfer,1M-context, throughput or TPOT validation.
