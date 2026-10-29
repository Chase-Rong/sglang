#include <torch/library.h>
#include "torch_helper.h"
#include "external_host_mapping.h"
#include "prefill_dma.h"
#include "fused_sparse_attention_overlap.h"
#include "shared_runtime.h"
#include "external_plan_warmup.h"
#include "planner_prepare.h"

TORCH_LIBRARY(sgl_fsa, module)
{
    // Standalone loader handshake; distinct from normal ABI3 and warmup ABI1.
    module.def("abi_version() -> int", []() -> int64_t { return 1; });
    module.def("external_plan_prepare(Tensor(a!)[] inputs, Tensor(b!)[] outputs, int capacity, int max_routes, int host_tokens, int abi_version=3) -> ()");
    module.def("external_plan_prepare_capabilities() -> int[]", TORCH_FN(sglang::npu_kernel::PlannerPrepareCapabilities));
    module.def("npu_fused_sparse_attention_overlap(Tensor query, Tensor(a!) shared_kv, Tensor full_kv, int full_kv_address, Tensor full_block_table, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, Tensor logical_topk, Tensor request_slots, Tensor row_to_request, Tensor row_union_index, Tensor union_keys, Tensor union_slots, Tensor union_count, Tensor union_miss, Tensor union_owner, Tensor union_cacheable, Tensor identity, Tensor(b!) ready, Tensor(c!) status, Tensor(d!) completion_identity, Tensor(e!) counters, Tensor(f!) safe_topk, Tensor(g!) membership, Tensor(h!) selection_table, Tensor(i!) selection_status, Tensor(j!) workspace, int resident_capacity, float scale, int abi_version=3) -> Tensor");
    module.def("shared_v3_capabilities() -> int[]", TORCH_FN(sglang::npu_kernel::SharedV3Capabilities));
    module.def("shared_v3_workspace_bytes() -> int", TORCH_FN(sglang::npu_kernel::SharedWorkspaceBytes));
    module.def("external_plan_warmup_capabilities() -> int[]", TORCH_FN(sglang::npu_kernel::ExternalPlanWarmupCapabilities));
    module.def("external_plan_fill_only(Tensor(a!) shared_kv, Tensor full_kv, int full_kv_address, Tensor full_block_table, Tensor committed_lengths, Tensor request_slots, Tensor union_keys, Tensor union_slots, Tensor union_count, Tensor union_miss, Tensor union_owner, Tensor union_cacheable, Tensor identity, Tensor(b!) ready, Tensor(c!) status, Tensor(d!) completion_identity, Tensor(e!) counters, int resident_capacity, int delay_cycles=0, int fail_request=-1, int abi_version=1) -> ()");
    module.def("shared_last_workspace_bytes() -> int", TORCH_FN(sglang::npu_kernel::SharedLastWorkspaceBytes));
    module.def("memfabric_region(int address, int[] shape, Tensor dtype_template, int device_id, bool shared_gva) -> (Tensor, int)",
               TORCH_FN(sglang::npu_kernel::CreateMemFabricRegion));
    module.def("prefill_dma_copy(Tensor(a!) device_tensor, Tensor(b!) host_metadata, int device_address, Tensor runs, bool to_host) -> ()");
    module.def("register_memfabric_region(Tensor source, int device_address, int device_id) -> ()",
               TORCH_FN(sglang::npu_kernel::RegisterMemFabricRegion));
    module.def("unregister_memfabric_region(Tensor source, int device_id) -> ()",
               TORCH_FN(sglang::npu_kernel::UnregisterMemFabricRegion));
    module.def("shm_allocator_create_and_register(int size, int device_id, str name) -> (int, int)");
    module.def("shm_allocator_free_all(int device_id) -> ()");
    module.def("shm_allocator_free_by_name(str name, int device_id) -> ()");
    module.def("shm_allocator_export(str name, int device_id) -> (int, int, int, int)");
    module.def("shm_allocator_attach(int segment, int size, int creator, int created, int device_id, str name) -> (int, int)");
    module.def("unidex_copy(Tensor src, Tensor(a!) dst, Tensor src_index, Tensor dst_index, Tensor valid_mask, int src_rows, int dst_rows, int block_bytes, int max_copy, int block_dim=8, int? src_ptr=None, int? dst_ptr=None) -> ()");
    module.impl("shm_allocator_create_and_register", TORCH_FN(sglang::npu_kernel::shm_allocator_create_and_register));
    module.impl("shm_allocator_free_all", TORCH_FN(sglang::npu_kernel::shm_allocator_free_all));
    module.impl("shm_allocator_free_by_name", TORCH_FN(sglang::npu_kernel::shm_allocator_free_by_name));
    module.impl("shm_allocator_export", TORCH_FN(sglang::npu_kernel::shm_allocator_export));
    module.impl("shm_allocator_attach", TORCH_FN(sglang::npu_kernel::shm_allocator_attach));
}

TORCH_LIBRARY_IMPL(sgl_fsa, PrivateUse1, module)
{
    module.impl("external_plan_prepare", TORCH_FN(sglang::npu_kernel::PlannerPrepare));
    module.impl("external_plan_fill_only", TORCH_FN(sglang::npu_kernel::ExternalPlanFillOnly));
    module.impl("npu_fused_sparse_attention_overlap", TORCH_FN(sglang::npu_kernel::FusedSparseAttentionOverlap));
    module.impl("prefill_dma_copy", TORCH_FN(sglang::npu_kernel::PrefillDmaCopy));
    module.impl("unidex_copy", TORCH_FN(sglang::npu_kernel::unidex_copy));
}
