# MemFabric Host allocator dependency audit

Inspected actual MemFabric 1.2.0 binaries and Python wrappers in host34 container
`jzx_sglang_5.3_flash`. This is static installed-binary evidence, combined with
the earlier LOCAL initialization failure; no NPU workload launched in this audit.

## Offload API has an eager kernel dependency

`mf_acc_offload.py:empty` calls `offload.malloc(bytes)` and wraps its pointer using
ctypes/torch.frombuffer. It does not allocate or pin the memory itself.
The pool must first be established by `offload.initialize(config)`.

Installed `libmf_acc_offload.so` initializes as follows:

1. `AccOffloadEntryManager::Initialize` selects LOCAL or SHARED implementation.
2. Both implementations call `hybm_init`.
3. Both call `AccOffloadLaunchApi::TryLoadLibrary` and check the result before
   proceeding to create/reserve the memory entity.
4. The loader reads `MEMFABRIC_HYBRID_EXTEND_LIB_PATH`, opens
   `libmf_hybm_accoffload.so`, and resolves **both** `AccOffloadSparseCopy` and
   `AccOffloadGroupPackCopy`. Failure prevents successful pool initialization.
5. LOCAL then calls `hybm_create_entity`, `hybm_reserve_mem_space`,
   `hybm_alloc_local_memory(HYBM_MEM_TYPE_HOST)`, and `hybm_get_slice_va`.
   SHARED adds sharing/exchange/mapping before exposing the pool pointer.

Thus the missing extension implements copy launchers, but this particular
offload API requires it even for a caller intending to use allocation only.
Merely omitting `sparse_copy`/`group_pack_copy` calls does not avoid initialization.
No fake success shim, binary patch, or unsafe alias substitution was introduced.

## Other dependencies are not all mandatory extensions

| Component | Allocation relevance in the inspected build |
|---|---|
| `libmf_acc_offload.so` | Public offload facade; present |
| `libmf_smem.so`, `libmf_hybm_core.so` | Linked/imported foundations; present, including wheel-renamed dependencies |
| `libmf_hybm_accoffload.so` | Eager offload initialization dependency; absent |
| `libmf_hybm_copy_extend.so` | Core initialization attempts loading but ignores the loader result; needed for particular extended copy operations, not a universal Host-allocation prerequisite |
| ACL/HAL/HCCL and other selected runtime backends | Required as selected by the core loader; distinct from optional MemFabric extensions |

The evidence distinguishing the two extensions is the error handling:
`AccOffload*DramEntry::Initialize` branches on the offload-loader result.
`DlApi::LoadLibrary` calls `DlHybmExtendApi::TryLoadLibrary`, then continues to
`DlHcomApi::LoadLibrary` and returns the previously saved successful loader status.
It does not propagate either optional load result in that initialization path.
`HostDataOpSDMA::BatchCopyExtend` separately checks/calls its extended-copy function
pointer, so optional at initialization does not imply optional when that copy path
is requested. Python also explicitly preloads several optional libraries with
OSError ignored; this is not equivalent to the required offload launch library.

## Lower-level allocation option

Installed headers expose `hybm_create_entity`, `hybm_reserve_mem_space`, and
`hybm_alloc_local_memory(...,HYBM_MEM_TYPE_HOST,...)` in `libmf_hybm_core.so`.
The offload implementation itself uses these after the eager launcher check.
Using this lower layer is a possible allocator-only adapter, not an already
qualified replacement. It must implement pool ownership, TP handle exchange,
address mapping and cleanup, and independently validate DMA + FSA access to one
physical Host allocation. Missing offload launch libraries do not establish that
all MemFabric memory APIs are unusable.

## Evidence handles

Under `results/unified_host_20260917/`:

- `141.61.55.34_allocator_chain.log`: Python allocation wrapper and native calls.
- `141.61.55.34_allocator_branches_focus.log`: mandatory loader branches in LOCAL
  (call0xf11c) and SHARED (call0x1620c), subsequent HYBM allocation calls.
- `141.61.55.34_loader_branch_detail.log`: core optional load at0x22b280, continued
  load at0x22b284, returned saved status at0x22b288; Host copy API declarations.
- `141.61.55.34_copy_extend_dependencies.log`: ELF dependencies and Python import.

Addresses apply only to the inspected binary SHA256
`0c99b633b1a55d34ab3ff7d8d746193b7a30cfb162d63e7f1d47b3d58d316947`
for the offload facade; they are not stable source line references.

## Existing containers checked for usable initialization

Follow-up checked three live containers without touching NPUs or services:

| Host/container | Installed version | Offload extension loader result |
|---|---|---|
|34 / jzx_sglang_5.3_flash|1.2.0|-1|
|248 / pxc_glm53_flash|1.2.1|-1|
|244 / pxc_glm53_flash_v2|1.2.1|-1|

The CPU-only probe loads the actual library using ctypes and calls the exported
native `AccOffloadLaunchApi::TryLoadLibrary`; it does not call hybm_init,
aclrtSetDevice, offload.initialize, or import Torch/MemFabric Python provisioning.
Tested the inherited environment and a process-local override pointing
MEMFABRIC_HYBRID_EXTEND_LIB_PATH at the installed package lib directory. Both
failed in all three containers: unset variable in the first arm, `get lib path
failed` in the second. No extension was fabricated and no environment persisted.

Static inspection of1.2.1 confirms the same eager dependency in LOCAL(call0xcf34)
and SHARED(call0x149e4), including a required successful result before pool setup.
Therefore none of these inspected installations is ready for unmodified offload
Host-pool initialization. This does not rule out uninspected installations or
the lower-level HYBM adapter. Logs:`*_loader_cpu_*.log`,`*_loader_path_*.log`,
`141.61.54.248_allocator_branches_121.log`.
