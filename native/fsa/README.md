# Standalone FSA extension

This target builds a small `sgl_fsa` Torch extension against the installed Torch
and Torch NPU. It does not build or install an sgl-kernel-npu wheel and does not
replace the `npu` operator namespace. It contains the FSA ACLNN binding, native
external cache planner, Host registration and unidex copy.

The external planner defaults to CLOCK replacement with a persistent hash index.
Rebuild the extension with the Python adapter: it requires the new
`fsa_enqueue_clock_plan` binding. Set `SGLANG_NPU_FSA_PLANNER_POLICY=lru` before
constructing planners to select the legacy sorting implementation for comparisons
or rollback; other values are rejected. Do not switch policy on a live planner.

CLOCK protects all requested hits before selecting victims and retains distinct
slots for duplicate keys. It changes replacement order, not attention's selected
keys or mutable-tail validity. Each query row still has its own cache. Hash links,
reference bits and the CLOCK hand live in CPU tensors owned by the planner and
shared by all captured graphs and eager calls. They reset with owner changes,
length rollback and epoch rollover. At capacity 4096 the workspace adds 163848
CPU bytes per reserved row. The callback performs no sorting, device operation,
or scratch allocation; the existing metadata transfers remain.

The optional `FusedSparseAttentionState(mapped_host_plan=True)` uses registered
Host external plans. Production backend defaults are unchanged pending TP
qualification. The CPU cache
planner retains slot placement on Host, and `fsa_host_plan` encodes the per-layer
hit/miss plan using separate applied-key state. An ordered callback commits that
state after attention. Above 16 query rows, independent CPU rows use up to four
OpenMP workers. Indexer keys, request identities, lengths and mutable-tail
validity still travel from NPU to Host. The reference attention route retains
device plans for its explicit gather implementation.

Up to 16 query rows, attention reads the encoded Host plan through a validated
registered address. Larger batches stage the completed plan to HBM to avoid
repeated remote metadata reads. This removes the full int64 slots return and
device plan construction; it does not eliminate all metadata transfer. Captured
callback arguments remain alive until graph destruction. Plan state is ordered
on the execution stream; concurrent use of the same selection state on unrelated
streams is unsupported. Destroy graphs and drain streams before closing planners.

Rebuild this extension when updating the mapped-plan Python adapter; the new
`fsa_host_plan` and `npu_fused_sparse_attention_overlap_mapped` bindings are
required. The original FSA binding remains available. Current-token Host
writeback keeps its existing synchronous indexed-copy path; Decode DMA staging
is not enabled by this change.

For A5 Graph replay with TopK 2051, 16 query heads and 4096 resident slots,
single-device ABBA measured complete planner/attention/commit reductions of
2.5%, 12.7%, 2.2% and 0.4% at 1, 4, 16 and 64 query rows respectively. With
approximately 80% hits, reductions were 7.8%, 12.9%, 1.8% and 0.5%. The 64-row
result is effectively parity. These measurements do not establish TP8 or
end-to-end TPOT improvement. Hardware regression tests cover changing TopK
order, mutable tails, request generations, graph reservation and layer reuse.

The attention device binary and host tiling must be built separately as an
isolated OPP vendor. Both that vendor and this extension must match the target
SoC. An Ascend910_93 binary is not an A5 binary.

The source donor for the wrapper/helpers is sgl-kernel-npu commit `539e7c9`.
Export its `csrc` tree into the assigned work directory and pass that directory
as `FSA_SOURCE_ROOT`. For full-KV tensors larger than4GiB, also apply the current
sgl-kernel-npu `unidex_copy` host/kernel changes:64-bit global extents and byte
offsets, including queued destination offsets. The original donor rejects
these tensors; removing only its host check would leave overflowing offsets.
`scripts/npu/remote_fsa_native_capacity.py` exports the donor and overlays these
two files from the sibling checkout for an isolated build and hardware tests.
Device attention adaptation to arch35/NoPE/2051 external
plan is provided separately in `native/fsa_attention`; do not use the donor's
arch22 attention binary on A5. The wrapper in this directory overrides the
donor's one-query-per-request assumption and preserves MTP cumulative lengths.

```bash
cmake -S native/fsa -B "$SGLANG_FSA_WORK_DIR/native-build" \
  -DFSA_SOURCE_ROOT="$SGLANG_FSA_WORK_DIR/kernel-source" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION="$SOC_VERSION"
cmake --build "$SGLANG_FSA_WORK_DIR/native-build" -j2
export SGLANG_NPU_FSA_EXTENSION="$SGLANG_FSA_WORK_DIR/native-build/libsgl_fsa_extension.so"
```

Add the matching isolated vendor to `ASCEND_CUSTOM_OPP_PATH` and its `op_api/lib`
to `LD_LIBRARY_PATH`, preserving the container's existing vendor paths. The
Python source path remains the selected SGLang checkout. Compilation and
hardware qualification must be evaluated separately: the A5 helper build/load
and three registered-copy/planner tests passed. The separately built arch35
attention vendor and corrected wrapper passed54 attention/lifecycle cases. See
`FSA_OFFLOAD_MIGRATION.md` for exact logs and limitations.

## HiCache-managed integrated offload

Integrated offload uses HiCache page ownership over the existing MemFabric
regions. It does not allocate a second complete Host KV payload. The default
UnifiedRadix FULL/MAMBA stack owns prefix matching, cache references and
eviction; the offload runtime retains Prefill slot reuse and direct Host reads
in fused Decode attention. There is no switch selecting the previous private
integrated memory-management path.

Device KV page IDs and Host page IDs can differ. Prefill DMA accepts CPU int64
`[source_row, destination_row, count]` runs, coalescing only when both ends are
contiguous. Rebuild this extension before using the HiCache Python adapter.
The older two-column native copy contract remains available to other callers.
Decode keeps its current-token writeback operation and maps its destination to
the HiCache Host slot. Attention consumes a separate Host page table; the
compressed Indexer keeps its device addressing.

Backing allocation precedes graph capture. HiCache adopts completed pages
without repeating their D2H copy. Indexer and KDA checkpoint sidecars have
separate Host allocations and completion events. Page recycling waits for
outstanding readers; shared TP backing also requires agreement across ranks
before reuse. Closing the backing requires all graphs and streams to be drained.

This integration targets the single-service, non-speculative CP1/DCP1 path.
L3 storage, external linkers and speculative state restoration are not included.
Existing integrated-offload configuration enables the required HiCache/radix
settings automatically. Non-offload allocation remains unchanged. Compilation
and CPU contract tests alone do not qualify NPU, TP8, prefix-hit model accuracy,
or end-to-end performance; each requires its corresponding hardware validation.
