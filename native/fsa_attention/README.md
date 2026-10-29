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
