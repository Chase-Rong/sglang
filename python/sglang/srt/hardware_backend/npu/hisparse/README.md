# Ascend HiSparse offload

The Ascend implementation uses SGLang's HiSparse lifecycle entry points while
retaining the existing NPU memory pools, HiCache ownership and fused attention
operators. `managers/hisparse_factory.py` selects the implementation at startup.
CUDA and ROCm continue to use the native coordinator.

## Configuration

Enable the NPU path with `--enable-hisparse` and provide an explicit Host budget
in `--hisparse-config`. For example, the configuration below requests a 32 GiB
Host budget for an integrated service; the actual capacity still depends on the
model, HBM budget, concurrency, MTP and Graph settings.

```json
{
  "device_buffer_size": 4096,
  "npu": {
    "decode_host_kv_bytes": 34359738368,
    "prefill_layer_slots": 2,
    "dsa_offload_backend": "lim_copy_sfa"
  }
}
```

`disaggregation_mode` determines the role. An independent Prefill service uses
`prefill_host_kv_bytes`; an independent Decode service uses
`decode_host_kv_bytes`. Separate P/D requires the Ascend MemFabric transport.
Integrated and Decode services retain the HiCache Host tier and Radix cache.
The independent Prefill service retains its request-owned pool and disabled
Radix cache.

Independent Prefill uses ordinary Prefill attention and never initializes a
Decode resident manager. Its fused sparse attention setting resolves to false;
the LIM geometry restrictions apply to integrated and Decode services only.

The existing `SGLANG_NPU_*` offload variables remain compatibility inputs to the
same HiSparse runtime. Explicit JSON and environment values must agree. The
parser rejects conflicts before publishing normalized inputs. Host budgets have
no implicit allocation default. `host_to_device_ratio` does not size NPU pools;
the existing joint DSA/KDA solver determines capacity.

`lim_copy_sfa` remains the default for its supported GLM-5.3-Flash geometry:
TopK2048, KPool4, NoPE512, page128 and at most six query routes.
`external_plan` retains the existing planner plus FSA path and its geometry
checks. Backend selection is fixed before allocating state.
LIM requires BF16 query and KV storage. An `auto` dtype is checked again after
the model and KV dtypes have been resolved; `external_plan` retains FP16 support.

## Resource and execution ownership

- HiCache retains Host page allocation, request/cache references and Prefix
  restore/eviction. The coordinator adopts the existing backing and never
  allocates or frees a second Host KV pool.
- Prefill retains layer slot reuse, H2D prefetch and D2H writeback overlap.
  Completed Prefill KV is already on Host, so the coordinator performs no
  additional all-layer backup.
- NPU requests use the ordinary Prefill-to-Decode batch merge. Native HiSparse's
  staging queue, swap-in kernel and current-token remapping are not executed.
- The fused paths retain their own resident manager. LIM runs at the Indexer
  call site and CopySFA consumes its plan at the attention call site.
- Target/draft payloads and resident state remain separate. They retain shared
  request page ownership and the existing MTP reuse and invalidation rules.
- Graph buffers retain stable addresses. Runtime publication dependencies also
  cover direct draft Graph replay. Request retirement updates CPU generation
  metadata; the existing metadata-preparation path uploads it before execution.
- Ordinary batch creation, merge and snapshots retain the coordinator so
  retraction follows the same lifecycle as request completion.
- Graceful shutdown drains target/draft execution and PD writers before
  detaching HiCache views and closing their physical Host allocation owners.
  TP ranks agree that draining succeeded before closing shared pools. A drain
  failure retains the allocations for a retry.

## PD transport

The transport remains:

```text
P HBM layer slot -> MemFabric -> bounded D HBM staging -> DMA -> D shared Host KV
```

The independent P Host backup supports later Prefill chunks; network transfer
reads the protected P HBM slot. Host visibility, transfer completion, ACK and
staging credit retain their existing separate lifetimes. The native HiSparse
Host-direct destination selection does not apply to Ascend.

TP sharing is local to one compatible host/IPC group. Cross-host transport uses
the same registered staging descriptors and network sessions, rather than
sharing local VMM addresses. The experimental double-buffer staging and direct
VMM transfer probes are not part of this implementation.

## Validation

CPU contracts are covered by the HiSparse configuration, coordinator, factory
and PD dispatch tests, together with the existing HiCache, Prefill, MTP and
budget tests. Real NPU operator correctness, TP visibility, Graph performance
and complete P/D model inference require separate hardware qualification.
Source migration alone does not establish those results.
