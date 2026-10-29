# external_plan execution contract

`external_plan` has one normal execution path:

```text
standard Indexer → physical-key/causal preparation
                 → protected-hash SIMT planner + final metadata (one kernel)
                 → SFA: unique Host loads, resident writes and Attention
                 → checked completion
```

`lim_copy_sfa` remains a separate, unchanged backend. Backend selection retains
its existing default; selecting external_plan uses only `shared_v3` / `FUSED`.
The historical `row_v0`, `shared_v1`, `shared_v2`, `UNION_FILL` and `OVERLAP`
execution selectors fail before allocation. No CLOCK/LRU or CPU callback planner
runs in external_plan. Ordinary Attention planning is native; CPU translation is
only a metadata validation helper. Duplicate selected occurrences retain their
Attention weight, while payload movement is unique per physical key.

```bash
export SGLANG_NPU_DSA_OFFLOAD_BACKEND=external_plan
# These are external_plan defaults; setting them explicitly is optional.
export SGLANG_NPU_FSA_EXECUTION_ABI=shared_v3
export SGLANG_NPU_FSA_LOAD_MODE=FUSED
```

The main operator keeps its existing name `npu_fused_sparse_attention_overlap`.
Its 30-argument interface is ABI3; `shared_v3` capability/configuration names
identify that interface version, not another Attention implementation.
The normal path requires exact planner and consumer ABI3 capabilities. Its
supported geometry is BF16 TND, H32, NoPE512/RoPE0, page128, K≤2051, routes≤6,
T≤6144, Qcap≤32767, B≤min(T,1024), C≤16384 and
W=max(16,next_power_of_two(routes×K))≤16384. C is page-aligned and covers routes×K.
The development device is Ascend950DT_9582 with CANN9.3; this does not qualify
another SoC or shape. Build the matching extension and sole SFA OPP from fresh
directories using [the native build guide](../../../../../../native/fsa/README.md).
The build_fused_sparse_attention_overlap.py packager copies the sole checked-in specialized source;
there are no V1/V2 source generators or execution registrations.

Optional committed-prefix warmup remains available through
`SGLANG_NPU_FSA_WARMUP_THRESHOLD` (0 disables it). It has its own prefix planner
and `external_plan_fill_only` helper ABI1; it does not require another Attention
OPP. Warmup validates a fully committed, published prefix, protects valid hits,
assigns misses to unprotected slots, and emits sorted unique physical keys with
the corresponding original logical prefix positions. It never computes
Attention. Exact generation, mapping/content epoch and completion identity are
required before a marker is published. Decode reuses completed resident payload;
Graph capture cannot run warmup. The independent helper capability is
`[1,1,512,0,128,16384,32767,1024,9,1]`.

Resident identity, allocation, payload validity, reusable eligibility and
execution leases remain separate. Generation and mapping-epoch changes revoke
reuse. Capture retains native inputs and leases until its owner drains; replay
records Host reader fences. Failed native submission drains before cancellation,
and failed drain quarantines the workspace/Host owner. Target and draft instances
own independent resident and fixed workspace allocations.

Current storage accounting uses Q=request-owner capacity, B=batch capacity,
T=token-row capacity, K=expanded topk and C=resident capacity. Per layer it includes
Q×C×512×2 payload bytes, queried native workspace, Q×(11C+12) resident metadata,
B×W×22+B×C×8+B×145 planner/identity buffers, T×K×20+T×12 plan buffers,
and T×(32800+4×(C/128)+4×status_stride) final SFA metadata.
Global device metadata is 34Q+12 and the CPU generation snapshot is 8Q.
Warmup adds 32C+96+96Q bytes per layer and 32Q coordinator bytes per owner.
Capture snapshots add 34Q+34TK+24T+80B+8 bytes per layer/capture bucket.
No staged partial Attention outputs, merge buffers, load streams or CLOCK/LRU
age/recent/hand state are allocated. These are owned-tensor budgets, not allocator
peak-HBM guarantees.

Historical source, old binaries, failed attempts and previous measurements are
preserved outside the production tree in review artifacts. The cleanup evidence
is `../review-artifacts/hisparse-external-plan-cleanup-20261028-213310/` relative
to the repository root. Its CPU/portable validation does not establish a new
native build or device pass. Previous component timing belongs to its original
source and binary hashes. Phase4 F06 direct internal copy/math overlap evidence
remains pending; this cleanup does not begin Phase5 or establish serving TPOT.
