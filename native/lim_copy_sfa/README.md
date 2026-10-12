# A5 LIM / Copy-SFA migration

This target provides two A5 native stages: fused Indexer/resident management,
and fused miss-copy/sparse attention/residency publication. The Indexer prepares
per-query pooled lengths, computes scores and TopK, expands tokens, protects
the request union, applies protected length-hashed replacement, and emits a signed-slot plan.
CopySFA consumes that plan and the deduplicated request union. On a new request
generation or rollback, or when any request has more than 512 union misses, it
fills the batch's miss unions before attention. Otherwise it overlaps missing
KV loads and per-query miss prefetch with the attention pipeline. Payload validity is
published inside the same kernel after the copy consumers finish.

CopySFA owns its kernel, Cube/Vector services, buffer utilities, vector functions
and tiling structures under `op_kernel/copy_sfa`. It does not include or link
the other fused-attention operator, generate types from that operator, invoke
the installed FSA operator, or launch a separate Triton commit kernel. The
independent NoPE implementation retains its source license notices and
manual-sync compiler options. Compilation and functional qualification do not
establish serving performance; component and full-model workloads are measured
separately.

CopySFA reserves workspace from the device Cube count and launches up to that
count according to query rows, with eight groups for small cases. Only actual
first-fill or above-threshold work runs the two-entry Host-to-UB-to-HBM queue and the leading global
barrier. Steady all-hit skips that stage and the trailing commit barrier.
Pure-hit rows skip owner scans. Mixed rows scan earlier queries' compact miss
prefixes and the three tail positions. Each query may prefetch its own copy of
a repeated miss into private workspace, while only the earliest query persists
it to resident KV. This avoids reading another AIV's unfinished write.
Other steady cases publish validity in parallel from the planner's 32-bit
mutable-tail presence masks instead of rescanning the full TopK for each miss.

For the 2048-token sparse body, LIM publishes an additional miss-prefix/hit-suffix
permutation without changing the public TopK/slot order. Within each segment,
the original TopK order is retained; this is not A3's source-ID sort. Pure hit and pure miss
use implicit identity order. CopySFA caches each row's IDs and signed plan, rotates
512-token groups using the official request-based permutation, and preserves the final three tail
positions. Attention remains tiled at 128 on A5; its read-ahead gather uses two UB
banks and 32 rows for at most 32 local heads, or 16 rows for larger head counts.
This retains UB headroom while issuing the next bank's reads before writing the
pending bank. A specialized hit-only gather avoids the mixed-source branch.
When attention still gathers Host misses, a BF16 bit-copy vector function
converts each gathered bank into padded NZ scratch and writes directly to L1.
The odd scratch pitch avoids UB bank conflicts. Input-ready events are recorded
before the next bank's H2D, so conversion does not wait for future reads.
When every payload is already resident, including after a batch union fill,
the kernel retains the faster GM-to-L1 layout conversion path. This decision
uses device miss/reset state on every replay, not batch/query specialization.

The membership ABI remains `(rows, 16400)` int16. Signed plans occupy
14333:16384; the optional mixed-body permutation occupies 4096:6144; control
16392:16394 stores the 32-bit mutable presence mask, and control 16394 stores
the sparse body's miss count. Consumers use identity order for miss counts 0/2048.
These fields are internal to the matched LIM/CopySFA binary; old/new binaries
must not be mixed within one producer-consumer chain.

The Indexer uses the CANN 9.1.0 Ascend950 RegBase implementation installed under
`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/lightning_indexer`.
`prepare_reference.py` verifies every imported header against pinned SHA256
values, stages headers in the build directory, and retains their license notices.
It publishes per-query score tiles, partitions the pooled sequence across AIC
groups, and defers per-query TopK until those tiles are complete. A mixed-core
`SyncAll<false>()` joins every participating AIC/AIV before TopK reads the shared
scores. This avoids serializing a long pooled sequence on one AIC for B1.
The dot product, weighted reduction and TopK implementation remain from the SDK;
the schedule and score storage are adapted. The installed SDK is not modified.

Scores use the SDK's sortable uint32 bit encoding in int32 storage. Only the
visible pooled prefix is valid; callers must mask the remainder, including
zero-length rows. Treating these bits as FP32 logits directly is incorrect.
The standalone Indexer takes one query per index row. The fused manager takes
request cumulative ends and constructs independent per-row pooled lengths.
Pooled page tables
are distinct from HiCache attention Host page tables. Runtime inputs must keep
lengths within the reserved score columns and physical page IDs within the
pooled key tensor, as required by the original Indexer.

The current bring-up wrapper supports BF16 query/key/weights, head dimension128,
32 or64 heads, page128, Top512 and up to the device's Cube-core count (at most64).
The wrapper reserves workspace using hardware capacity; each launch limits its
groups to the available row/sequence tasks. Per-query TopK and request-owner
management use the corresponding AIV groups. The UB layout supports at most12416 resident
tokens and six query routes. After scoring finishes, row-plan publication reuses
the dead matrix-result UB without changing the SDK TopK buffer layout. Each row
publishes signed slots and a compact list of payload misses. If every selected
token already owns a slot, the owner joins only those misses, or emits an empty
union for all-hit, without scanning or sorting the complete resident set.
The owner publishes its union count in request state; native CopySFA bounds
first-fill and validity publication by that count rather than reserved capacity.
Generation changes, rollback and newly selected unmapped sources retain full
management. Source unions use hardware-sorted 1024-entry runs and parallel
SIMT deduplication/rank merging. Every query's selected slots are protected before allocation.
Allocation consumes empty slots first, then scans 2048-slot resident chunks
from a length-hashed starting chunk, excluding every protected slot. It no
longer sorts resident scores or uses the scalar score heap. Union suffix bytes
are unspecified: every consumer must use the published request union count.
Payload validity still requires physical mapping and mutable-tail checks.
SIMT performs reset, stable miss compaction, victim selection and mapping
updates. One request-wide VF repairs up to six query rows; outputs remain
staged in aligned UB and published by DMA. Scratch storage is reused across
phases, keeping the largest manager allocation at 198144 bytes.

The BF16 histogram TopK now overlaps slot-map DMA with each histogram pass.
Its survivor sidecar carries four independent int32 slots per pooled candidate;
source IDs are not truncated and four adjacent tokens need not own adjacent
resident slots. The sidecar reuses dead matrix-result UB and the final classifier
still verifies Host physical mappings and mutable tails. Per-query pooled causal
lengths remain independent; the official C8 grouped-query clipping is not copied.

Queries in the same request share K loads in the BF16 score tiles when their
pooled page tables match. The kernel checks table equality on device during
each replay and falls back to independent rows if they differ. Grouped scoring
covers the maximum visible pooled length of the request; each row's Top512 is
still bounded by its own floor(length / 4). Variable route counts, empty
requests and changes between shared and independent tables retain that contract.

CopySFA prefetch workspace reserves 512 BF16 NoPE512 rows per query, shared by
layers within each runtime instance (0.5 MiB per query, 30 MiB for B10Q6).
The joint memory budget includes this workspace and the larger Indexer core
capacity; no additional full Host KV copy is allocated.
Outputs and workspace
are supplied by the caller and remain alive through Graph destruction. Native
launches use the Torch NPU queue and current stream without a Host callback.

```bash
cmake -S native/lim_copy_sfa -B "$WORK/lim-native-build" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION=Ascend950DT_9582 -DCMAKE_BUILD_TYPE=Release
cmake --build "$WORK/lim-native-build" -j2
export SGLANG_LIM_NATIVE_EXTENSION="$WORK/lim-native-build/libsgl_lim_copy_sfa.so"
export SGLANG_LIM_COMPONENT_TEST=1
python -m pytest -q test/manual/npu/test_lim_native_indexer.py
python -m pytest -q test/manual/npu/test_lim_native_plan.py test/manual/npu/test_lim_native_fused.py
```

This library uses its own `sgl_lim_native` namespace. Building or testing it does
not install a wheel or replace an OPP vendor. The serving selector is
`SGLANG_NPU_DSA_OFFLOAD_BACKEND=external_plan|lim_copy_sfa`; `lim_copy_sfa` is
the default on this branch. Set `external_plan` explicitly to use the original
LRU planner and FSA. The default does not enable offload by itself.
Selecting `lim_copy_sfa` requires the library path above,
`SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=1`, integrated HiCache offload,
BF16 NoPE512, KPool4, page128, at most six query routes and at most64 local
attention heads. Missing native support raises an error without silent fallback.

The KPool read phase calls the fused Indexer/manager once. Attention consumes
that plan after the existing current-token Host writeback. Shared-indexer layers
retain the supplied token set and invoke native management followed by CopySFA.
Seeded Draft chains run management on the first step, then publish a post-copy
plan for the remaining steps of that same iteration. Initial mutable-tail
misses remain misses until the chain ends. This supplied-TopK mode uses the same protected
length-hashed replacement policy;
it does not recompute a different selection from the current draft query.

Within a seeded one-query-per-request Draft chain, prefix-preserving rollback
invalidates speculative suffix payloads while retaining the immutable prefix.
Generation changes still reset the request; mapping, layer, request identity and
seed changes invalidate the reuse template. Ordinary supplied-TopK callers keep
the original full-reset behavior. Rebuild the native library with the Python
changes: `manage_out` and `copy_sfa_out` have new optional attributes.

Host page mappings are written directly into capture buffers. Draft preparation
shares the buffer only within one fixed-batch preparation and covers the last
draft position; every replay refreshes it. CopySFA keeps three task-local mask
banks, so Vec1 never consumes the next Vec0 task's mutable order state. Reordered
chunk plans and Fill-plan sign updates use vector operations. LIM reuses its
prepared classification only while the request generation and sequence remain
valid, preserving independent physical validity per attention layer.

Each physical attention layer has independent residency/payload state. Scratch
is shared within a worker; target and draft pools retain separate runtime state.
Request-generation uploads run outside capture, and native kernels also check
physical Host mappings and speculative mutable tails. Graph padding is masked
with inactive request slots. The memory solver accounts for the NPU source map,
request-shared union capacity, layer payloads, and native workspaces. HiCache
pages, MemFabric allocation, PD transfer and Prefill offload remain shared with
the existing backend; no second full Host KV payload is allocated.

The official A5 algorithm adaptation is detailed in `OFFICIAL_A5_MIGRATION.md`.
The fused and supplied-TopK SIMT kernels use configured Host-argument launchers with 216 KiB
dynamic UB and batch scheduling, reserving 32 KiB for DCache and 8 KiB for
runtime/stack. Its embedded ELF and launch ABI come from the same SDK build.

The mixed-core score join remains necessary for the current sequence partition.
The slot sidecar overlaps histogram computation, not the entire matrix/manager
pipeline. Candidate physical-address checks, per-request control decisions and
DMA publication remain explicit; metadata validity is not inferred from TopK
membership alone.

The device-management component currently limits the mutable tail to32 tokens
per query. It computes per-query tail membership and joins that information
when publishing resident validity after attention. A token selected only by a
later query must not be invalidated by an earlier query that did not select it.
Scores and selected IDs can be shared, but each physical attention layer must
retain independent payload validity; this remains a serving integration gate.

`indexer_manage_out` publishes plans, not completed KV payloads.
`copy_sfa_out` publishes physical validity only after completing attention.
The test-only `LegacyFsaComparator` and its Triton post-copy commit remain UT
comparators and are not used by the new serving backend. Run
`test/manual/npu/test_lim_native_pair.py` for the actual native pair and serving
adapter checks; full-model E2E and performance acceptance require their own runs.
`test_lim_production_geometry.py` adds 128K/32-head B1/B8/B16 fixed and changing
selection plus miss-union boundaries. `test_lim_draft_reuse.py` compares the
six-query extend and four-step draft management/attention chain; it does not
include the model projections, Indexer, FFN or serving TPOT.
