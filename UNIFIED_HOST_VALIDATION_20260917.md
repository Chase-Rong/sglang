# Unified shared Host KV allocation validation

## Scope and current result

The goal is one physical Host KV allocation usable by asynchronous DMA, direct
AIV/FSA reads, and TP ranks. No extra pinned full-KV mirror is permitted in this
qualification. **No candidate has passed this complete contract yet.** Production
allocation and Prefill overlap defaults remain unchanged.

This is separate from the earlier pinned-mirror component benchmark in
`SDMA_PREFILL_VALIDATION_20260917.md`. That benchmark's 35.92% result does not
qualify the candidates tested here.

## Hardware evidence

| Candidate | Host | Observed result | What it establishes |
|---|---|---|---|
| Fresh SysV allocation, one PINNED + MAPPED registration | 248 | Registration and device pointer lookup succeed; first H2D returns ACL 507899 | Combined flags alone do not qualify DMA |
| Fresh SysV allocation, PINNED only | 34 | Wrapper returns 107000 during device-pointer lookup | No usable ACL device alias from this registration |
| Fresh SysV allocation, PINNED only, separate capability probe | 248 | Register returns 0; pointer lookup returns 107000; first H2D SIGSEGVs inside ACL-copy bridge | Failure also exists without prior HAL registration; not just additive-registration incompatibility |
| MemFabric offload LOCAL pool, 1 GiB | 34 | Initialize returns -1; cannot load offload launch library | Runtime dependency blocked; no allocation or DMA result obtained |

The fresh PINNED crash reproducer now requires `UNIFIED_REPRO_PIN_CRASH=1`.
Normal validation skips this already-rejected combination. Crashes occurred in
isolated test processes, not serving processes. SysV segments are marked
IPC_RMID immediately after attach for last-detach cleanup.

MemFabric installed Python API is present, but initialization requires
`libmf_hybm_accoffload.so`. It was not found under the inspected `/usr/local`,
`/home`, `/opt`, `/workspace`, and `/sgl-workspace` trees. The initial process also
lacked the package lib directory in LD_LIBRARY_PATH; the runner now adds it.
The missing extension is not solved by that search-path adjustment. No package,
wheel, shared CANN installation, or serving environment was replaced.

The installed wrapper's `fabric.empty(..., pin_memory=True)` ignores pin_memory
and calls `offload.malloc`; it cannot be treated as evidence of DMA pinning.
Its SHARED scene must be qualified separately from the LOCAL allocation probe.
The validation adapter explicitly rejects inherited SysV export/attach methods.

Driver headers warn against registering VMM addresses with halHostRegister and
against DMA registration of OS allocations on Linux below 5.19. Host34 runs
6.6.0-159.4.7.160.oe2403sp4.aarch64, so the old-kernel restriction does not explain
the observed failures. No exact driver/runtime root cause has been established.

## Occupancy and test boundaries

Every NPU launch checked live npu-smi. Initially34 had eight idle cards. After
the two34 probes, other Ray/VLLM jobs occupied all eight. Then248 briefly became
idle and accepted the single-device PINNED capability probe. Subsequent AIV
validation attempts were refused because another TP8 SGLang service was starting.
244 also became occupied. No other jobs were stopped; no tests were launched on
known occupied cards. The planned combined-flags AIV test has **not executed**.

There is no TP2/TP8 sharing, FSA/Graph, lifecycle stress, no-mirror benchmark, or
full-model/1M performance qualification from these new candidates.

## Artifacts and resume

Local scripts/logs: `results/unified_host_20260917/`.
Remote roots on34 and248:
`/home/j00628475/unified_host_validation_20260917_v1`.
Container: `jzx_sglang_5.3_flash`.
The isolated source and test libraries are below that root. No wheel rebuild.

Native FSA extension SHA256:
`5108d58dd8dc62dd20525bdcf597b55cd3929e843b9b3073ff033bfc7cacaf44`.
Validation bridge SHA256:
`42ed26ba3c2560ee066fe3b607fb14684b9e972cfc9eca241c672d45a04d17ec`.
CPU ownership tests: nine passed; these validate cleanup contracts, not DMA.

Next gates, in order:

1. Obtain a compatible MemFabric offload extension or a documented supported
   allocator/mapping contract for this A5 runtime. Do not retry known crashing
   PINNED registration or layer it on the production HAL mapping.
2. On an idle device, qualify the same allocation for repeated DMA H2D/D2H and
   AIV reads/writes. Record CPU submit time and a trace with actual UBDMA tasks.
3. Test cold/warm/reshuffled FSA and Decode Graph without exporting a mirror.
4. Implement and qualify the backend's real shared-handle protocol on TP2/TP8,
   including one writer, cross-rank consumption, and drained cleanup.
5. Run the matched four-layer A/B and then production task-queue and E2E tests.

## Exact release20260904 image inspection

User named image:
`swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-950-release20260904`.
It is present on248/244 with image ID
`sha256:d6c96b75d219a0c81d708a313415dd9687d4e9ef8e890084d1baaca2341ee82c`;
not present under this tag on34 at inspection time.

Verified the pristine image directly using an ephemeral read-only container on248,
network disabled, no NPU devices, entrypoint overridden to shell. It contains
MemFabric1.2.0 (git999034e9320abf27bbde7be83a32d3bf4d0f7940), with
`libmf_acc_offload.so` SHA256
`0c99b633b1a55d34ab3ff7d8d746193b7a30cfb162d63e7f1d47b3d58d316947`.
The missing `libmf_hybm_accoffload.so` and `libmf_hybm_copy_extend.so` are absent
from a complete root-filesystem filename scan. The former is explicitly named
inside the existing offload library, together with
`MEMFABRIC_HYBRID_EXTEND_LIB_PATH`, `AccOffloadSparseCopy`, and
`AccOffloadGroupPackCopy`. Setting a search path alone cannot supply this binary.

248's existing pxc_glm53_flash container uses this image but its installed package
is MemFabric1.2.1 (gitd369e2f779a0d4c98a07fd8ad182647132a7b1ce). That is a
container-level change, not the pristine image version; its root scan also found
no requested extension. All29 running-container root scans across34/248/244
completed without finding that library. Shared /mnt scans timed out on34/244;
do not interpret these as exhaustive absence from shared storage.

Evidence: `results/unified_host_20260917/141.61.54.248_pristine_release.log`,
`*_release_image.log`, `*_containers_scan.log`, and `*_abi_*.log`.
No NPU test, production install, or service restart performed for this inspection.
