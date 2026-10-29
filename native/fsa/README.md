# external_plan native extension

`external_plan` has one decode path: the independent protected-hash SIMT planner
emits final metadata, then the existing SFA performs unique Host loads/resident
writes and Attention, followed by checked completion. `lim_copy_sfa` has its own
unchanged source and state. Old row, CPU CLOCK/LRU, union-fill and staged-overlap
implementations are archived outside the production tree.

The operator retains the established `fused_sparse_attention_overlap` name.
Its input contract is ABI3; versioned capability helper names describe that contract.
The normal path keeps its existing 30-argument schema, ABI 3, eight counters per
request and exact capability checks. The SFA OPP is built from the single
committed implementation in `native/fsa_attention/op_host` and `op_kernel`.

## Build

Use the exact SoC from the target runtime. The validated development platform
is Ascend950DT_9582 with CANN 9.3, BF16 TND H32, latent dimension512, RoPE0,
page128. This does not qualify other shapes or platforms.

The exported donor bundle must contain the shared Host allocator, unidex_copy,
ACL adapter helpers and `fused_sparse_attention_overlap/registered_acl_tensor.h`.
No donor LRU planner or row Attention wrapper is compiled.

```bash
source /usr/local/Ascend/cann-9.3.0/set_env.sh
FSA_BUILD=/tmp/external-plan-build
cmake -S native/fsa -B "$FSA_BUILD/extension" \
  -DFSA_SOURCE_ROOT=/path/to/exported/kernel-source \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION=Ascend950DT_9582
cmake --build "$FSA_BUILD/extension" -j4
python native/fsa_attention/build_fused_sparse_attention_overlap.py \
  --cann "$ASCEND_HOME_PATH" --source "$PWD/native/fsa_attention" \
  --output "$FSA_BUILD/opp"
cmake -S "$FSA_BUILD/opp" -B "$FSA_BUILD/opp/build" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DASCEND_COMPUTE_UNIT=ascend950 -Dvendor_name=sgl_fsa_overlap \
  -DENABLE_BINARY_PACKAGE=ON -DENABLE_SOURCE_PACKAGE=ON -DENABLE_TEST=OFF \
  -DCMAKE_INSTALL_PREFIX="$FSA_BUILD/install"
cmake --build "$FSA_BUILD/opp/build" -j4
cmake --build "$FSA_BUILD/opp/build" --target binary -j4
cmake --install "$FSA_BUILD/opp/build"
export SGLANG_NPU_FSA_EXTENSION="$FSA_BUILD/extension/libsgl_fsa_extension.so"
FSA_VENDOR="$FSA_BUILD/install/packages/vendors/sgl_fsa_overlap"
export ASCEND_CUSTOM_OPP_PATH="$FSA_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$FSA_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Build into fresh paths and load matching binaries. Device `.o` identity is
separate from the Host `op_api` library. The latter may remain byte-identical
when the device implementation changes without an ABI change.

## Exported execution operations

| Operation | Purpose |
|---|---|
| `external_plan_prepare` | One SIMT planner and final metadata kernel, ABI3 |
| `npu_fused_sparse_attention_overlap` | Unique loading, original Attention, then finish |
| `shared_v3_capabilities`, `shared_v3_workspace_bytes` | Exact normal-consumer contract and caller-owned workspace |
| `external_plan_fill_only` | Independent committed-prefix warmup helper, no query/output |
| `external_plan_warmup_capabilities` | Exact helper contract, independent ABI1 |

Host mapping/registration, shm allocator, prefill DMA and unidex_copy remain
shared utilities. `shared_last_workspace_bytes` reports the last actual ACL
workspace requirement; it is not an allocation or performance guarantee.

The warmup capability vector is
`[1,1,512,0,128,16384,32767,1024,9,1]`: helper ABI, BF16, latent dimensions,
RoPE dimensions, page size, maximum resident capacity, maximum request capacity,
maximum batch, identity columns, unique-copy support. The normal consumer's ABI3
and this helper's ABI1 are independent contracts. The helper is independent of the Attention OPP.
Warmup requires a complete committed prefix, sorted unique physical keys,
corresponding logical owners, unique destination slots and exact identity.
Its two counters count actual completed Host loads and resident writes.
`delay_cycles` and `fail_request` are bounded fault-injection parameters and
normally remain 0 and -1. An error never publishes successful completion.

## Ownership and error handling

`fused_sparse_attention_overlap.cpp` is the single adapter translation unit because the donor
ACL helpers contain non-inline definitions. `shared_runtime.cpp` contains only
common launch, validation, workspace and scoped descriptor/TLS cleanup; no old
Attention path is hidden there. Do not compile those included implementation
fragments independently.

Workspace is owned by the caller for the full eager/Graph lifetime. Queued
callbacks retain raw launch arguments, not the final Tensor owner. Descriptor
and HugeMem cleanup belongs to the active caller or queued-handler scope;
it must not be deferred to an unrelated final destructor. Enqueue errors
propagate before dependent work is submitted. Python retains the execution
lease and Host readers through completion, or drains before cancellation and
quarantines resources when completion cannot be established.

Portable lifecycle checks can be compiled with:

```bash
c++ -std=c++17 -I native/fsa native/fsa/tests/shared_lifetime_test.cpp \
  -o /tmp/external-plan-lifetime-test
/tmp/external-plan-lifetime-test
```

The source cleanup does not close Phase4 F06 overlap evidence or establish
serving/TPOT performance. Source, local tests, new native build and actual device
validation are reported separately in the cleanup review artifacts.

## Same-name legacy library rejection

The current OPP exports the application-owned
`sglangFusedSparseAttentionOverlapAbi` marker from the same `libcust_opapi` as
its ACL entry points. Capability checks and the execution entry validate the
marker value and require `GetWorkspaceSize`, launch and marker addresses to have
the same DSO base via `dladdr`. The verified function pair is the pair actually
used for descriptor/workspace preparation and launch. An older row OPP with the
same operator name, a missing/wrong marker, or mixed vendor symbols is rejected
before `GetWorkspaceSize` or device submission. This is a host contract check,
not a new CANN operator or a substitute for source/device-binary verification.
The marker is included in both normal shared-library and packed-object targets.
