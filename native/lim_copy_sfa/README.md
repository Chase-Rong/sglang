# A5 LIM / Copy-SFA migration

This target provides two A5 native stages: fused Indexer/resident management,
and fused miss-copy/sparse attention/residency publication. The Indexer prepares
per-query pooled lengths, computes scores and TopK, expands tokens, protects
the request union, applies score-driven eviction, and emits a signed-slot plan.
CopySFA consumes that plan and the deduplicated request union. On a new request
generation or rollback it fills the union before attention; on steady requests
it overlaps missing KV loads with the attention pipeline. Payload validity is
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
first-fill work runs the two-entry Host-to-UB-to-HBM queue and the leading global
barrier. Steady all-hit skips that stage and the trailing commit barrier.
Other steady cases publish validity in parallel from the planner's 32-bit
mutable-tail presence masks instead of rescanning the full TopK for each miss.

For the 2048-token sparse body, LIM publishes an additional miss-prefix/hit-suffix
permutation without changing the public TopK/slot order. Within each segment,
the original TopK order is retained; this is not A3's source-ID sort. Pure hit and pure miss
use implicit identity order. CopySFA caches each row's IDs and signed plan, rotates
512-token groups across requests/query rows, and preserves the final three tail
positions. Attention remains tiled at 128 on A5; its read-ahead gather uses two UB
banks and 32 rows for at most 32 local heads, or 16 rows for larger head counts.
This retains UB headroom while issuing the next bank's reads before writing the
pending bank. A specialized hit-only gather avoids the mixed-source branch.

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
Query cumulative ends must describe one query per index row. Pooled page tables
are distinct from HiCache attention Host page tables. Runtime inputs must keep
lengths within the reserved score columns and physical page IDs within the
pooled key tensor, as required by the original Indexer.

The current bring-up wrapper supports BF16 query/key/weights, head dimension128,
32 or64 heads, page128, Top512 and one to eight AIC groups. The fused manager
uses eight AIC groups and all sixteen AIVs for the final per-row TopK and
request-owner management. Its current UB layout supports at most12416 resident
tokens and six query routes. After scoring finishes, row-plan publication reuses
the dead matrix-result UB without changing the SDK TopK buffer layout. Each row
publishes signed slots and a compact list of payload misses. If every selected
token already owns a slot, the owner joins only those misses, or emits an empty
union for all-hit, without scanning or sorting the complete resident set.
The owner publishes its union count in request state; native CopySFA bounds
first-fill and validity publication by that count rather than reserved capacity.
Generation changes, rollback and newly selected unmapped sources retain full
management. Source unions use hardware-sorted 1024-entry runs and an exact
ordered merge; empty-cache initialization assigns slots without victim sorting.
Occupied-cache eviction uses hardware-sorted score runs and slot-sorted equal
score groups, retaining deterministic ties. Nonfinite/very large score sums
outside the finite sentinel range keep the exact scalar heap fallback.
Payload validity still requires physical mapping and mutable-tail checks.
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
`SGLANG_NPU_DSA_OFFLOAD_BACKEND=external_plan|lim_copy_sfa`; `external_plan` stays
the default. Selecting `lim_copy_sfa` requires the library path above,
`SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=1`, integrated HiCache offload,
BF16 NoPE512, KPool4, page128, at most six query routes and at most64 local
attention heads. Missing native support raises an error without silent fallback.

The KPool read phase calls the fused Indexer/manager once. Attention consumes
that plan after the existing current-token Host writeback. Shared-indexer layers
and Draft seed reuse retain the supplied token set and invoke only native
management followed by CopySFA. Without fresh logits this supplied-TopK mode
uses equal eviction priorities and slot order as the deterministic tie-break;
it does not recompute a different selection from the current draft query.

Each physical attention layer has independent residency/payload state. Scratch
is shared within a worker; target and draft pools retain separate runtime state.
Request-generation uploads run outside capture, and native kernels also check
physical Host mappings and speculative mutable tails. Graph padding is masked
with inactive request slots. The memory solver accounts for the NPU source map,
request-shared union capacity, layer payloads, and native workspaces. HiCache
pages, MemFabric allocation, PD transfer and Prefill offload remain shared with
the existing backend; no second full Host KV payload is allocated.

These changes adopt the A3 miss-only union and all-hit early-exit design, but
do not reproduce its complete scoring pipeline: A3 packs slot/source payloads
while scoring tiles, whereas this A5 version consumes the SDK's completed
pooled TopK and performs slot/address lookup afterward. The mixed-core score
join remains necessary for the current sequence partition. Supplied-TopK
management, ordered run merges, and slot/address lookup still contain scalar loops.

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
