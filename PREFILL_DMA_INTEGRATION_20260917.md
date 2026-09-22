# Default integrated Prefill DMA

User authorized making the measured MemFabric/DMA path the production default.
The integrated-offload feature switch still selects offload versus resident
execution, but no separate DMA or copy-overlap environment switch is required.

## Implementation

- `model_runner.py` selects `PipelinedIntegratedOffloadRuntime` unconditionally
  for integrated offload, with H2D lookahead and D2H asynchronous writes enabled.
- `DecodeHostKvBacking(integrated=True)` owns a `MemFabricHostKvPool`. Prefill
  and Decode use the same bytes. No extra pinned-Host mirror or post-timing export.
- Rank zero owns physical Host storage in SHARED mode; peers own zero physical
  bytes. Peer CPU tensors carry addresses and shapes only. Native registration
  checks the GVA-to-device mapping before FSA or DMA can consume it.
- Each forward snapshots prefix/current row indices and builds immutable int64
  contiguous-run plans once. Padding is masked, fragmented rows are supported,
  and all DSA layers reuse these plans. DMA uses `aclrtMemcpyAsync` with mapped
  device addresses through `OpCommand::RunOpApiV2`, not raw Python/ctypes streams.
- Slot reuse waits for previous attention readers and D2H writers. The next
  chunk/Decode waits for Host publication. Only the owner writes shared Prefill
  KV; publication remains on the compute stream. Unregister/free is collective
  after drain. Native queued operations retain tensors and record NPU storage
  on the consuming stream.
- External-planner metadata and disaggregated PD allocation are unchanged.
  Decode current-token writes remain device-indexed and Graph-compatible.

Supported integrated configuration remains eager Prefill, MTP0, CP1/DCP1, no
mixed Prefill/Decode batch, and Decode Graph support. One slot serializes H2D
lookahead; two slots overlap adjacent DSA layers. MemFabric allocates physical
storage rounded up to GiB, reported separately from logical KV bytes. Its A5
branch currently supplies one pool per process and a same-host SHARED bootstrap.

## Build and environment

Remote isolated source:
`/home/j00628475/unified_host_validation_20260917_v1/source` in
`141.61.55.34 / jzx_sglang_5.3_flash`.

Candidate MemFabric root:
`/home/j00628475/memfabric_a5_fd334486_20260917_v1`.
This build contains the optional-Host-URMA-symbol compatibility repair committed
as `3ef081a7` in the local MemFabric repository. Original pre-repair wheel/run
artifacts must not be mistaken for the tested library set.

After sourcing the CANN and existing FSA environment, select:

```bash
mf_root=/home/j00628475/memfabric_a5_fd334486_20260917_v1
export PYTHONPATH="$mf_root/source/src/smem/python/memfabric_hybrid:$PYTHONPATH"
export LD_LIBRARY_PATH="$mf_root/source/src/smem/python/memfabric_hybrid/memfabric_hybrid/lib:$LD_LIBRARY_PATH"
export MEMFABRIC_HYBRID_EXTEND_LIB_PATH="$mf_root/extend"
export SGLANG_NPU_FSA_EXTENSION="$mf_root/fsa-binding-production/libsgl_fsa_extension.so"
```

The standalone binding builds with `native/fsa/CMakeLists.txt`; the existing A5
attention OPP remains selected. No installed wheel or service was replaced.

## Validation checkpoint

- Local CPU row-plan, allocator cleanup, runtime, backing, configuration and
  existing shared-Host/offload regression:110 passed.
- New native binding builds successfully; initial SHA256:
  `0db9ef1fbe67d3e8dd1a571b9d7fa18c02efd7e7786eed496a8f757c8589ffa7`.
- TaskQueue1 LOCAL DMA:two pool lifetimes, repeated writes, fragmented rows,
  cross-stream events, Graph replay and invalid-range rejection passed.
- First TaskQueue1 four-layer SFA component:128K prefix/8K chunk/16 heads/topk2048,
  seven samples, AIV serial111.688 ms versus production DMA+overlap86.690 ms
  (22.38% reduction). Full output SHA matches prior experiments. Decode fused
  eager and Graph transition also passed. This run preceded the NumPy row-plan
  refinement; final ABBA results are recorded separately after qualification.
- TP2 DMA/Graph/collective case passed. Combining two test modules then stalled
  while recreating the HCCL process group, confirmed by py-spy in
  `_patched_new_process_group_helper`, before attention execution. Only the two
  verified owned leftover workers were stopped; subsequent TP modules run in
  fresh processes. This failed combined run is not counted as a full-suite pass.
- TP2 fresh-process FSA Graph/HCCL:4 cases per rank passed, covering FP16/BF16
  and both SysV/non-integrated and MemFabric/integrated allocation.
- TP8 TaskQueue1:5 cases per rank passed, covering shared DMA/Graph/HCCL, FP16
  and BF16 three-chunk/four-layer/two-slot Prefill with fragmented mappings,
  subsequent Decode current-write Graph, and FP16/BF16 fused Graph+HCCL replay.
  Only rank zero allocates the physical1GiB test pool; seven peers allocate0.
- Above4GiB hardware H2D and D2H offsets passed, using a physical5GiB MemFabric
  pool and row offsets beyond4GiB in the real NPU slot.
- TaskQueue0 LOCAL allocator/DMA/Graph regression passed as well.
- Final production ABBA,7 samples per arm /14 pooled per backend, same128K/8K
  four-layer case, includes dynamic row planning inside each timed forward:

| TaskQueue | AIV serial median | Default DMA+overlap median | Reduction |
|---|---:|---:|---:|
| 1 |110.2044 ms|77.3108 ms|29.85%|
| 0 |107.3582 ms|76.3166 ms|28.91%|

TaskQueue1 AIV arm medians112.1707/108.2357 ms, DMA77.3473/77.0672 ms.
TaskQueue0 AIV106.8518/107.5831 ms, DMA76.3195/76.3137 ms. Report pooled
medians with the arm variation; do not interpret them as service TPOT.
All eight arms retain identical complete output SHA256
`2234c2d0704f970c0e647c45e0fd910030c05ca02e587a9a7bc0b2ad95467703`.
Both backends use the same MemFabric backing and no lab monkeypatch/extra mirror.
The AIV reference intentionally uses the previous serial runtime.

The initial production TaskQueue1 trace contains8 UBDMA KV copies,4 SFA tasks
and0 unidex tasks. Three next-layer H2D copies overlap preceding SFA;6/8 total
copy tasks overlap SFA in that trace. First H2D remains exposed; first D2H finishes
before the first SFA in this trace. It predates the CPU NumPy descriptor refinement.

Artifacts: `results/unified_host_20260917/production_dma_results/` holds the raw
ABBA samples, summary, trace evidence, native binary and matching14-file
local/remote source SHA manifest. Per-run logs/idle checks live in its parent.
Remote raw results are under `production_prefill_abba_queue{0,1}` and
`production_dma_trace` in the isolated test root. No owned test process remains;
the final check showed all eight NPUs idle. No unrelated process was stopped.

These are component and lifecycle results, not new 1M/full-model TTFT or TPOT
qualification. All NPU launches are gated by live idle checks.
