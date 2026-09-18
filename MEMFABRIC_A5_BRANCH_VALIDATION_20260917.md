# A5 offload branch build and validation

## Source and capability assessment

Local repository: `D:/newCode/memfabric/memfabric_hybrid`.
Branch: `br_feature_a5_offload`.
Commit: `fd33448651a6bacc0b89ad2e9344cb2f0e6e069d` (`aicpu+mte`).
The checkout was clean before and after inspection; no source changes made there.

This branch provides the missing extension sources and the main Host KV pool
interfaces needed for an allocator candidate:

- LOCAL/SHARED allocation and free, with sizes aligned to GiB.
- SHARED can have rank0 contribute physical DRAM and all other ranks contribute
  zero bytes. Its example broadcasts rank0's allocated addresses for peer reads;
  peers must not independently allocate and assume the same slots.
- LOCAL provides `offload.get_dva`, resolving Host-to-device mapping through
  `hybm_hva_to_dva`. SHARED does not override GetDva: base class returns failure.
  SHARED instead maps a unified GVA and its example uses that GVA directly.
- `OFFLOAD_FLAG_URMA_POOL` selects a different LOCAL allocation path for URMA
  registration, with distinct Host/device addresses; it is not interchangeable
  with the default VMM-backed pool.
- `libmf_hybm_accoffload.so` and `libmf_hybm_copy_extend.so` can be built from this
  tree for `ascend950pr_9599` / `dav-c310`. A wheel-only installation does not
  automatically provide these separately built extensions.

Limits relevant to our integration:

- Default LOCAL/SHARED data operation is MTE, and the A5 `AccOffloadSparseCopy`
  launcher uses64 AIV blocks. This is not proof of DMA removing AIV contention.
- The LOCAL_URMA KV-exchange example documents synchronous host-driven H2D versus
  AICore D2H. It cannot be assumed to provide the desired stream overlap.
- SHARED chooses TCP port8500+device_id/world_size internally. Test for port
  conflicts; multi-service deployment needs an explicit isolation strategy.
- Unified DMA+AIV/FSA+TP qualification remains necessary; source availability
  alone does not establish correctness or the earlier pinned-mirror speedup.

## Completed build

Host34 container `jzx_sglang_5.3_flash` was stopped when work started. Inspection
showed its entrypoint only configures CANN and starts a shell. Started this
designated container; no inference service restarted and no other job stopped.

Isolated build root:
`/home/j00628475/memfabric_a5_fd334486_20260917_v1`.
Source exported by `git archive`; archive hash and exact revision are in
`source-revision.txt`. Tracked shell files contained CRLF; normalized shell-file
line endings only in the isolated remote snapshot after the first build failed.
No kernel/source behavior patched. Archive builds lack Git metadata inside the
tree; use the explicit revision manifest rather than the embedded empty git field.

Used the repository `build_and_pack_run.sh` wrapper, RELEASE, Python ON, NPU,
HCOM OFF, etcd OFF,12 host jobs. Built AscendC operators with8 jobs, then linked
the Torch NPU launcher with the installed Torch ABI. Built the copy extension
with bisheng dav-c310. Did not execute the .run installer or globally install pip
packages. Source-tree Python wrappers use staged newly built libraries; no wheel
provisioning marker and no change to the shared CANN AICPU installation.

Outputs:

- `source/output/memfabric_hybrid-1.2.1_linux_aarch64.run`
- `source/output/memfabric_hybrid/wheel/memfabric_hybrid-1.2.1-cp312-cp312-linux_aarch64.whl`
- `extend/libmf_hybm_accoffload.so`
- `extend/libmf_hybm_copy_extend.so`

Extension SHA256:

- accoffload: `605949d0c4abcae08893d423949378b6ab8bc8a882cf9b32f1b0fa5dc336b8a6`
- copy_extend: `edc8e506a3050b2a0f567d5a313ad14568dc76ff8ab5ee558572c8a7d72109df`

## Validation outcome

CPU import/load test passed: actual `AccOffloadLaunchApi::TryLoadLibrary` returns0.
`/proc/self/maps` confirms all three base libraries and the offload extension
come from the isolated build. `offload.get_dva` is exported and
`torch.npu.is_initialized()` isFalse. This resolves the earlier missing-extension
loader gate, but is not a complete memory-pool initialization test.

The existing CPU shared-region ownership suite passes9 tests. This checks the
test harness lifecycle only, not MemFabric hardware behavior.

Hardware launch was refused by occupancy check:34's eight cards became occupied
during compilation, and248/244 also have eight-card jobs. No new NPU test ran.
No claims yet for allocator initialization, DMA roundtrip, AIV access, Graph,
TP sharing, performance, or E2E from this build.

## Ready-to-run next step

The validation-only MemFabric adapter now uses `get_dva` when exported, instead
of assuming ACL can query MemFabric's mapping. Original production allocation
and serving defaults remain unchanged.

From local `D:/newCode/sglang/sglang_remote`, run after an idle window:

```powershell
python results/unified_host_20260917/remote.py run --host 141.61.55.34 --label branch_dma_aiv_graph --backend memfabric --memfabric-root /home/j00628475/memfabric_a5_fd334486_20260917_v1
```

The runner rechecks occupancy. Test sequence: LOCAL pool initialization,
three-generation DMA H2D/D2H + AIV write/read on the same allocation, followed by
FSA cold/warm/reshuffle/Graph. If that passes, qualify SHARED rank0-only physical
allocation onTP2 thenTP8, with its actual GVA contract and lifecycle. Then profile
UBDMA and rerun the no-mirror matched Prefill component benchmark.

Local evidence: `results/unified_host_20260917/branch_fd334486_artifacts/`,
`141.61.55.34_branch_cpu_load.log`, `*_branch_occupancy.log`,
and `*branch_dma_aiv_graph_occupancy.txt`.

## Follow-up: real LOCAL initialization and AIV access pass

After renewed occupancy checks,34 became free for actual testing. Unmodified
branch initialization failed before pool creation: the installed CANN libra.so
exports neither ra_socket_setwhite_list_status nor RsUpdate, which the branch
required globally even though only Host URMA calls it. Patched two local source
files to make that symbol optional at load and to reject Host URMA OpenDevice
with the real missing-API error. No fake success symbol or whitelist bypass.
The local MemFabric checkout now contains these explicit compatibility changes;
the earlier clean-source build remains the original unpatched artifact.

Patch: `results/unified_host_20260917/memfabric_host_urma_optional.patch`.
Incrementally rebuilt and staged only the candidate core library;current core
SHA256 `73a34fff8693ea6f0791df088572b8efc0b65c070e1bd21b0f17a886a38bf5e8`.
A compiled CPU probe linked to the actual library passes HCCP load and confirms
HostUrmaTransportManager::OpenDevice returns BM_UNDER_API_UNLOAD (-8).

Real NPU0 test `test_unified_host_pool_and_aiv`:1 passed. LOCAL1GiB pool initialized
and destroyed twice;each cycle tested3 generations of row-varying BF16 data with
AIV Host reads and writes,all bit-exact. Host pointer equals queried DVA in this
VMM pool. This qualifies small LOCAL mapping/lifetime only,not DMA or TP sharing.
14 adapter/ownership CPU tests pass locally and inside container.
One earlier post-patch invocation exited1 with an empty log;no surviving test
process was found. Fresh invocation captured the passing result;empty run is not
treated as a successful test or a diagnosed device failure.

## Unified LOCAL DMA and FSA, SHARED AIV and Graph

LOCAL8MiB same-allocation test passed3 DMA H2D/D2H + AIV read/write generations.
H2D CPU submissions0.055-0.132ms;all values matched,without a pinned mirror.
The old FSA wrapper initially rejected this valid mapping because its allowlist
only knew SysV allocations. Added `register_memfabric_region` and
`unregister_memfabric_region` in the isolated standalone binding. Registration
checks contiguous CPU metadata,current device,and actual MemFabric HVA->DVA for
both range endpoints. Unregister occurs before free,after users drain. Original
SysV validation is retained;no validation bypass. Candidate binding hash:
`bc2c277167ea05a1a25bafa9f66ba80e74eb2e24b8159d95016b1c341e97183d`.

Passed LOCAL fused cold/warm/reshuffle/Graph on the new allocation. Passed native
registration rejection for duplicate,forged DVA,wrong device,and double-unregister.
Also passed the original SysV FSA cold/warm/reshuffle/Graph using the new binding.

SHARED TP2 and TP8 passed:only rank0 allocates physical1GiB;other ranks allocate0.
All ranks read the same values through AIV,Graph replay changes3generations,
real HCCL all-reduce verifies the outputs,and each rank writes a distinct row
visible to rank0. Clean collective shutdown completes;driver emits a nonfatal
RemoveOneVaInfo warning for empty peer regions. This is a small mapping test,
not full-model or large-capacity TP qualification.

Important contract correction:imported shared GVA is device-mapped on peers,
not CPU-mapped there. The initial test wrongly called CPU torch.equal on a peer
view and rank1 segfaulted. Corrected it to validate CPU contents only on owner
and actual NPU reads on peers,following the branch's own example. Do not copy
the old SysV assumption that every rank may dereference Host tensors on CPU.
The external FSA registration added here validates LOCAL HVA mappings;SHARED FSA
binding adaptation itself has not been qualified by the shared copy tests.

## No-mirror Prefill component performance

Matched AIV/DMA/DMA/AIV order,7samples per arm per mode,14samples pooled per
backend/mode. Both backends use MemFabric LOCAL allocation with identical
4-layer BF16 SFA,prefix131072,current chunk8192,16heads,topk2048,two staging slots,
TASK_QUEUE_ENABLE=0. Profiling is outside timing. No pinned mirror or deferred
mirror export;the same Host bytes feed Decode FSA/Graph after Prefill writes.

| H2D prefetch / D2H async | AIV ms | DMA ms | Time reduction |
|---|---:|---:|---:|
|00|110.059|81.569|25.89%|
|10|107.949|76.556|29.08%|
|01|108.232|80.854|25.30%|
|11|107.952|76.405|29.22%|

All outputs are bit-exact across arms,modes and rounds,with SHA256
`2234c2d0704f970c0e647c45e0fd910030c05ca02e587a9a7bc0b2ad95467703`.
Mode11 per-arm medians:AIV110.673/105.125ms,DMA75.826/76.994ms;report the pooled
result with this observed run variation rather than promising exact E2E gains.

Trace verifies8UBDMA copies,4SFA kernels,no unidex copy kernels;7copies overlap
SFA,while the initial H2D precedes compute. H2D1.45-1.50ms,D2H0.086-0.180ms,
SFA18.42-18.43ms/layer. This qualifies the no-mirror single-device component
mechanism. TP8performance,production task-queue behavior,1M/E2E and full allocator
integration still remain. Raw evidence in local
`results/unified_host_20260917/memfabric_validated_results/` and remote
`unified_host_validation_20260917_v1/memfabric_prefill_abba/`.

Original built wheel/.run artifacts are the clean-branch build and do not contain
the later compatibility patch or FSA binding. The candidate uses its staged core
and separate binding explicitly;do not deploy the original wheel as the validated
candidate. Original container package and service defaults were not replaced.

## Final shared DMA gate and current resume commands

SHARED TP2 and TP8 additionally passed direct ACL DMA reads from the mapped GVA
using direction DEVICE_TO_DEVICE,including3generation Graph replays,HCCL all-reduce,
and AIV peer writes checked on owner. This is the correct device-address path for
peers without a CPU mapping. Each rank exited successfully. LOCAL performance
uses HOST_TO_DEVICE/DEVICE_TO_HOST on its CPU-visible allocation,as tested above.

Current candidate kernel binding must be selected explicitly;the original command
earlier in this report predates the external allocation registration adapter.

```powershell
python results/unified_host_20260917/remote.py run --host 141.61.55.34 --backend memfabric --memfabric-root /home/j00628475/memfabric_a5_fd334486_20260917_v1 --extension /home/j00628475/memfabric_a5_fd334486_20260917_v1/fsa-binding/libsgl_fsa_extension.so --label memfabric_local_checks --command 'env SGLANG_MEMFABRIC_REGISTER_BINDING=1 python3 -u -m pytest test/registered/unit/npu/test_unified_host_npu.py -q -s --tb=short'
```

Shared test launcher:
`/home/j00628475/unified_host_validation_20260917_v1/run_shared_tp2.sh`,with
`FSA_SHARED_TEST_RANKS=8 MEMFABRIC_SHARED_DMA=1 GLOO_SOCKET_IFNAME=lo` for TP8 DMA
and runner `--devices 0,1,2,3,4,5,6,7`. Omit MEMFABRIC_SHARED_DMA for AIV.
Launcher uses127.0.0.1:25971 to avoid container hostname resolution issues and
checks MemFabric internal port8500 before starting. All hardware launches gated
on actual device occupancy;do not infer availability from NPU utilization alone.

Remaining integration work:SHARED FSA binding must validate peer GVA mappings
without pretending they are CPU HVA mappings;framework Host reads/writes must be
owner-only. Qualify that combined attention path and the production task queue
before enabling a service. Existing TP8 copy/Graph tests and LOCAL FSA tests are
separate evidence. No full-model/1M TPOT claim or automatic default promotion.
