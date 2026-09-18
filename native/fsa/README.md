# Standalone FSA extension

This target builds a small `sgl_fsa` Torch extension against the installed Torch
and Torch NPU. It does not build or install an sgl-kernel-npu wheel and does not
replace the `npu` operator namespace. It contains the FSA ACLNN binding, native
external LRU planner, Host registration and unidex copy.

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
