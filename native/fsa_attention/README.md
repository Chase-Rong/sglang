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
