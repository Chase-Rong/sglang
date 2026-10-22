# NPU HiSparse offload

This backend keeps full DSA KV in a TP-shared MemFabric Host allocation.
HiCache owns the pages, prefix entries and cache lifetime. Bounded HBM layer
slots serve Prefill; Decode uses resident KV and sparse attention. KDA state
and the Indexer retain their existing device execution paths.

Enable it with `--enable-hisparse` and an explicit Host byte budget:

```bash
python -m sglang.launch_server \
  --model-path /path/to/GLM-5.3-Flash --trust-remote-code \
  --device npu --tp-size 8 --attention-backend ascend \
  --kv-cache-dtype bfloat16 --max-running-requests 10 \
  --context-length 1001024 --disable-prefill-cuda-graph \
  --enable-hisparse \
  --hisparse-config '{"npu":{"decode_host_kv_bytes":137438953472}}'
```

The example budget is a configuration example, not a capacity guarantee.
Available Host RAM, cgroup limits, OS reserve, TP placement, private sidecars
and MemFabric allocation alignment are checked before allocation. HBM sizing
includes resident/planner buffers, Prefill slots, graph buckets, MTP and KDA
state. The configured request count is an upper limit; the solver reduces it
with a warning when necessary. Requests share the resulting token pool.

## Attention backends

`hisparse_config.npu.dsa_offload_backend` selects one implementation:

- `lim_copy_sfa` is the default: fused Indexer/resident management followed by
  CopySFA. It requires BF16, TopK 2048, KPool 4, NoPE 512 and page size 128.
- `external_plan` retains the external planner and FSA implementation.

Both use the same Host ownership, Prefill offload and PD protocol. Rebuild and
load the matching native libraries when changing their ABI; do not combine
unmatched LIM and CopySFA binaries. Build requirements and library loading are
described in `native/fsa/`, `native/fsa_attention/` and `native/lim_copy_sfa/`.
These standalone extensions do not require rebuilding the full
`sgl-kernel-npu` wheel.

Fixed-chain NEXTN/EAGLE with topk 1 is supported by the framework adapter.
For LIM, at most six query routes are accepted, including the target token.
Decode graph capture remains available; layerwise Prefill requires eager
execution. Adaptive, branching, windowed and multi-layer drafting are rejected.

## Separate Prefill and Decode services

Use `--disaggregation-transfer-backend ascend` on both services. Configure
`prefill_host_kv_bytes` on Prefill and `decode_host_kv_bytes` on Decode in the
`npu` configuration object. Prefill uses bounded layer slots and requires
disabled radix cache. Decode uses HiCache ownership even when retaining
completed request prefixes is disabled.

The transport path is:

```text
P HBM -> MemFabric transport -> D bounded HBM staging -> DMA -> D Host KV
```

Layer leases carry destination mappings and generation identity. Decode waits
for KV, Indexer/draft tails, KDA and sampling metadata, plus TP readiness, before
admitting the request. Cancellation retains pages and staging until every
writer confirms drain. A timeout alone does not release those resources.
Cross-host transport exchanges registered staging descriptors, not local Host
VMM pointers. Local TP sharing requires the same host/IPC domain.

## Boundaries and validation

This implementation uses the HiCache Host tier without an external Store or
linker. It rejects simultaneous use of the independent NPU sparse-offload
manager, generic PD Host receive, unsupported CP/DCP layouts and captured
Prefill. Existing CUDA/ROCm HiSparse execution keeps its own coordinator and
logical/scratch allocation paths.

CPU contracts are under `test/registered/unit/npu/`. They cover allocation and
ownership, descriptor layouts, budget limits, Prefill overlap ordering, MTP
dispatch, prefix restore, layer transport and cancellation. CPU tests do not
qualify device DMA, native graph replay, model precision, MTP acceptance or
serving throughput. Those require the corresponding NPU and service tests on
the current checkout and native binaries.
