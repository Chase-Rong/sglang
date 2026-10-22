# A5 GLM-5.3 Flash attention vendor

This directory contains the standalone arch35 NoPE attention device and host
implementation. It requires absorbed512-dimensional Q/KV, zero RoPE, TND query,
paged KV, sparse block size1 and the external selection plan used by the
framework adapter. KPool-expanded2051-column indices include padding and tails.
Scale is supplied by the model (GLM-5.3 Flash uses1/sqrt(256)).

The device sources derive from the arch35 fused sparse attention implementation
in the vLLM-Ascend donor; all attention dependencies are vendored here. The
workspace row stride remains576, while attention uses only512 NoPE channels.
No other attention operator is called by this kernel.

Each AIV gathers up to16 rows at a time into two alternating UB buffers. In
the non-split-G path, once the final batch has been written to the attention workspace, `SignalKvReady`
publishes the existing L1 and workspace events before that batch's selection
miss writeback. AIC can then load the workspace and compute MM1 while AIV
finishes updating selection. Earlier batches still update selection before
reusing their UB buffer; this does not defer all selection writes in a tile.
The MTE3-to-MTE2 reuse event remains after selection writeback. Empty lanes
still publish once. Split-G execution retains its cross-core barrier and
publishes after selection writeback.
This ordering adds neither workspace reads nor UB/selection allocations.

The selection gather also adapts the windowed plan decoding and paired-copy
strategy from `vllm-ascend-poolingDSA-yq-opt-all0908`. Each 16-row window decodes
its plan once and calculates source offsets as needed. Contiguous source rows use one
strided copy; two scattered rows from the same source can be reordered by
address and copied together. Their selection update entries move with them,
so writeback still targets the original token's slot. Invalid or unsupported
windows fall back to the original row-wise path. Zero and out-of-range plan
entries are checked before issuing any fast-window copies. Fully hit windows skip the
selection writeback scan; miss rows retain the original direct UB-to-GM writeback.

The external-plan contract requires positive entries to identify valid cached
tokens for the current query. The framework establishes this in
`build_reference_sparse_plan` and `materialize_layer_plan`; the hit gather can
therefore avoid re-reading token IDs and full-KV block tables. Padding, causal
limits and mutable MTP tails remain part of the framework plan and attention
mask. No host synchronization or planner ABI change is introduced.

This is an A5 adaptation, not a replacement with the donor's A2 kernel. The
128-token attention tile, 16-row UB windows, three workspace buffers and A5
cross-core events are retained. The donor's workspace readback optimizations
are unnecessary because this implementation already writes selection directly
from UB. Its idle-core helper/credit-ring scheduler is not part of this port;
it requires a separate A5 scheduling and synchronization implementation. The
donor's disabled split-KV, removed wave scheduler and debug probes are not used.

Build only this OPP against the installed CANN9.1 template:

```bash
export SGLANG_FSA_WORK_DIR=/home/j00628475/fsa_glm53_a5_20260915
bash scripts/npu/build_fsa_attention_opp.sh "$SGLANG_FSA_WORK_DIR/attention-opp-new"
```

The build directory must be new. Nothing is installed into the global CANN
tree. The vendor is under `install/packages/vendors/sgl_fsa_a5`. The separate
`native/fsa` Torch extension is also required; the installed sgl-kernel-npu
wheel remains unchanged.

After verifying a healthy idle physical card, run:

```bash
export SGLANG_FSA_OPP_VENDOR="$SGLANG_FSA_WORK_DIR/attention-opp-v8/install/packages/vendors/sgl_fsa_a5"
export SGLANG_NPU_FSA_EXTENSION="$SGLANG_FSA_WORK_DIR/native-build-v5/libsgl_fsa_extension.so"
export SGLANG_FSA_TEST_DEVICE_ID=1
bash scripts/npu/run_fsa_attention_ut.sh -x -vv
```

The runner rechecks device occupancy and preserves existing OPP paths. There
is no atomic reservation against other sessions starting jobs concurrently.
Do not run while another task owns the selected NPU.

Build and54 hardware tests passed on34, including attention precision,
external-plan reuse, Graph, MTP and local Prefill/Decode Host-copy lifecycle.
See `FSA_OFFLOAD_MIGRATION.md` for coverage and logs. Whole-model1M execution,
real P/D transport and throughput are separate end-to-end gates.
