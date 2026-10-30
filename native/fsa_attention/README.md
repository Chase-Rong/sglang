# external_plan sparse Attention

This directory contains one committed implementation: the ABI3 external_plan
SFA consumer, with request-owned unique Host loading and the existing
QK/softmax/PV pipeline. The standard Indexer and fused planner/metadata kernel
are independent. `lim_copy_sfa` remains in its separate unchanged backend.

`op_host` and `op_kernel` are the complete source of the sole Attention OPP.
They were materialized from the previously validated SharedV3 specialization;
legacy row/V1/V2 branches and their generators were removed. The established
`fused_sparse_attention_overlap` source and operator names are retained; its
wire contract remains ABI3.

`build_fused_sparse_attention_overlap.py --cann ... --source ... --output ...` packages this source
into a fresh CANN project. It copies source files and records their SHA256
manifest; it does not rewrite an older Attention source into another version.
See `../fsa/README.md` for complete build/load instructions and the independent
fill-only warmup helper, which requires no additional Attention OPP.

The mathematical methods and iteration order remain derived from the original
CANN implementation. Original CANN Open Software License Agreement Version2.0
headers remain attached to their source files. Protected-hash planning and
request state are independently owned by external_plan; no live LIM state or
execution path is shared.

The `abi` directory defines an application-owned ABI3 marker linked into the
actual ACL API library. The native adapter checks the exact marker contract and
same-library provenance before calling the restored same-name ACL entry. Legacy
row OPPs therefore fail closed rather than being invoked with a new signature.

## Control-header reuse

The A5 consumer reuses the validated eight-word control header while one AIV
processes successive S2 tiles of the same logical query row. The cache is local
to that kernel invocation and is explicitly reset when global buffers are
initialized. Query-row changes reload the header, even when the queries share
one request-owned resident cache. New launches and Graph replays cannot reuse a
previous invocation's header.

This adapts the header-cache optimization from `ed25fb51d8` to the ABI3 protocol.
The consumer still validates the membership marker, ready marker, shared layout,
ABI version, owner, positive resident stride and end-aligned plan offset. Invalid
headers retain the existing fail-closed path; no legacy row-plan fallback is
introduced. Every tile checks its range against the cached plan count and loads
its current signed plan slice. Shared ready state, union checks, unique Host
loads, resident writeback and the QK/softmax/PV pipeline are unchanged.

The portable parser regression compiles the actual `GetSelectionRow`,
`PrepareExternalSelectionPlan` and `InitGlobalBuffer` definitions with CPU
tensor/DMA stubs. It checks header-copy counts, current signed slices, ABI3
rejection and invocation resets without loading CANN or allocating NPU memory:

```bash
python -m pytest -q test/registered/unit/npu/test_external_plan_header_cache_cpu.py
```

A C++17 compiler is required (`CXX`, `c++`, `g++` or `clang++`). These checks do
not replace an A5 OPP build, attention precision checks or real Graph replay.
