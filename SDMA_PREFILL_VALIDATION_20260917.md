# Prefill DMA replacement validation

## Result

DMA can remove the Prefill row-copy kernels from AIV and materially improve this
component workload. Direct replacement using the current shared Host allocation
is **not qualified**. A5 trace reports the successful engine as **UBDMA**, rather
than an AIV kernel; the prototype uses CANN `aclrtMemcpyAsync`, not MemCache
`batch_copy` or MemFabric BigMemory `copy_data_batch`.

Production code, native FSA binary and backend defaults were not replaced.

## Matched measurement

Host141.61.55.34,containerjzx_sglang_5.3_flash,NPU0 confirmed idle immediately
before each run. Other hosts248/244 and initially34 were occupied;no jobs stopped.
BF16,4 DSA layers,2 physical slots,16 query heads,topk2048,prefix131072,new8192.
Identical query/KV/indices;TASK_QUEUE_ENABLE=0 for both backends. ABBA order
AIV/candidate/candidate/AIV,7 iterations per mode per arm,14 samples per backend
and mode. Timings are device-event medians;profiling runs are outside timings.

| Runtime mode | AIV row copy | Pinned Host DMA fixture |
|---|---:|---:|
|00 serial|121.013 ms|80.941 ms|
|10 H2D prefetch|118.070 ms|76.284 ms|
|01 D2H async|118.546 ms|80.465 ms|
|11 bidirectional|118.273 ms|75.786 ms|

Mode11 reduction35.92%;within DMA alone,00->11 reduction6.37%. The overall
improvement includes faster standalone copies and useful overlap;it is not all
attributable to overlap alone. Individual AIV00 arm medians drifted across ABBA;
inspect raw samples before extrapolating this single component workload.

All four runtime modes and all four arms have identical Prefill output SHA256:
2234c2d0704f970c0e647c45e0fd910030c05ca02e587a9a7bc0b2ad95467703.
CPU attention oracle passed;new Host rows were poisoned and checked after copying.
Fused Decode eager/Graph transition passes after exporting the fixture data.

**Limits:**candidate uses an additional571473920-byte(545MiB) pinned Host mirror.
Its initial fill and mirror export back to original Host backing are outside the
timed interval. Fixed contiguous row plans are built ahead of timing and checked
against runtime indices during warmup. Thus results validate the copy/compute
mechanism,not arbitrary fragmented request mappings,production shared-Host
allocation,TP sharing,TaskQueue1,full-model TTFT/TPOT or1M capacity.

## Trace evidence

Pinned mode11 contains four SFA tasks (~18.42ms each) and eight UBDMA copy tasks.
Three next-layer H2D tasks (~1.54-1.64ms) start alongside preceding SFA;D2H tasks
(~0.10-0.20ms) overlap current SFA. The first H2D (~1.56ms) precedes first SFA.
No `unidex_copy` task remains in this profiled Prefill candidate window.

Example relative timestamps(ms):

| Work | Start | Duration |
|---|---:|---:|
|H2D layer0|0.441|1.560|
|SFA layer0|2.013|18.424|
|H2D layer1|2.013|1.636|
|D2H layer0|2.013|0.199|
|SFA layer1|20.448|18.424|
|H2D layer2|20.449|1.576|

## Why the existing allocation is not drop-in compatible

1. Original `RegisteredHostRegion.host_ptr` copies correctly but blocks the host
   call (~15.25ms median for128MiB);trace lacks the expected large asynchronous DMA
   tasks. First ABBA direct-host candidate mode11=132.692ms versus117.852ms AIV.
2. Original mapped `device_ptr` as D2D source returns ACL507899.
3. Adding `aclrtHostRegisterV2(...,ACL_HOST_REG_PINNED)` over the existing driver's
   registration returns0 but crashes the independent test process on the next
   ACL copy. This combination is rejected;never enable it in a service.
4. Ordinary Torch pinned Host copies correctly,submit~0.039ms and device~1.4ms
   per128MiB;trace identifies UBDMA. This is the positive control used above.

The unsafe registration reproducer requires `SGLANG_SDMA_REPRO_UNSAFE_PIN=1`;
normal test runs omit it. An earlier alias failure also caused pytest to crash
while formatting already-closed shared tensor views;short failure trace captured
the actual507899 return code on rerun. No service or third-party process killed.

## Tests and artifacts

-14 CPU address-plan/ownership checks pass locally and in target container.
-4 real original-Host roundtrip cases pass:FP16/BF16 x64/512 dimensions,3
  generations,noncontiguous row mappings,masked invalid and untouched rows.
-Two ABBA SFA validations pass:direct Host negative performance control and pinned
  Host positive mechanism control. Rejected pointer experiments are preserved.
-Compiled standalone bridge;no wheel or FSA OPP rebuild.

Local:`results/prefill_overlap_20260917/sdma_results`(direct-host trial) and
`sdma_pinned_staging_results`(positive trial) hold timings,CSV,trace and timeline.
Remote:`/home/j00628475/sdma_validation_20260917_v1`,isolated source subdirectory.
Entry script:`results/prefill_overlap_20260917/run_sdma_validation.py` refuses
occupied devices. Phases`roundtrip`,`pointers`,`benchmark`;positive benchmark adds
`--pinned-staging --profile`. No automatic retry or background resource claim.

## Next production gate

Design one Host allocation/registration lifecycle that simultaneously supports
real asynchronous DMA,existing FSA Host reads and TP-shared mappings. Candidate
is a qualified MemFabric shared Host pool or a supported unified pinned+mapped
allocator;this has not been selected or implemented. Avoid permanent full-KV
mirrors. Then adapt scheduler page/block metadata into contiguous run descriptors,
preserve stream ordering under the production task queue,and qualify TP2/TP8,
request reuse,fragmented pages and Decode Graph before E2E performance testing.
