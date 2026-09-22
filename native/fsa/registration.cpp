#include <torch/library.h>
#include "torch_helper.h"
#include "fused_sparse_attention_overlap/fused_sparse_attention_overlap.h"
#include "fused_sparse_attention_overlap/fsa_lru_planner.h"
#include "external_host_mapping.h"
#include "prefill_dma.h"
#include "host_plan.h"
#include "clock_plan_op.h"
#include "mapped_attention.h"

TORCH_LIBRARY(sgl_fsa, module)
{
    module.def("fsa_host_plan(Tensor trigger, Tensor keys, Tensor slots, Tensor owners, Tensor lengths, Tensor cacheable, Tensor(a!) applied, Tensor(b!) state, Tensor(c!) membership, Tensor(d!) misses, int rows, bool commit) -> ()");
    module.def("abi_version() -> int", []() -> int64_t { return 1; });
    module.def("supports_clock_workspace_int16() -> bool", []() { return true; });
    module.def("npu_fused_sparse_attention_overlap_mapped(Tensor query, Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, Tensor(c!) selection_kv_block_table, Tensor(d!) selection_kv_block_status, Tensor(e!) selection_membership_map, Tensor selection_topk_indices, Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor? full_q_actual_seq, float scale_value, int sparse_block_size, int selection_topk_block_size, str layout_query, str layout_kv, int sparse_mode, int? full_rope_address=None, int? full_kv_address=None, int? membership_address=None) -> Tensor");
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
    module.def("fsa_enqueue_lru_plan(Tensor trigger, Tensor keys, Tensor owners, Tensor lengths, Tensor(a!) resident, Tensor(b!) ages, Tensor(c!) row_state, Tensor(d!) slots, int active_rows) -> ()");
    module.def("fsa_enqueue_clock_plan(Tensor trigger, Tensor keys, Tensor owners, Tensor lengths, Tensor(a!) resident, Tensor(b!) ages, Tensor(c!) row_state, Tensor(d!) slots, Tensor(e!) workspace, int active_rows) -> ()");
    module.def("npu_fused_sparse_attention_overlap(Tensor query, Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, Tensor(c!) selection_kv_block_table, Tensor(d!) selection_kv_block_status, Tensor(e!) selection_membership_map, Tensor selection_topk_indices, Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor? full_q_actual_seq, float scale_value, int sparse_block_size, int selection_topk_block_size, str layout_query, str layout_kv, int sparse_mode, int? full_rope_address=None, int? full_kv_address=None) -> Tensor");
    module.impl("shm_allocator_create_and_register", TORCH_FN(sglang::npu_kernel::shm_allocator_create_and_register));
    module.impl("shm_allocator_free_all", TORCH_FN(sglang::npu_kernel::shm_allocator_free_all));
    module.impl("shm_allocator_free_by_name", TORCH_FN(sglang::npu_kernel::shm_allocator_free_by_name));
    module.impl("shm_allocator_export", TORCH_FN(sglang::npu_kernel::shm_allocator_export));
    module.impl("shm_allocator_attach", TORCH_FN(sglang::npu_kernel::shm_allocator_attach));
}

TORCH_LIBRARY_IMPL(sgl_fsa, PrivateUse1, module)
{
    module.impl("fsa_host_plan", TORCH_FN(sglang::npu_kernel::MaterializeHostPlan));
    module.impl("npu_fused_sparse_attention_overlap_mapped", TORCH_FN(sglang::npu_kernel::fused_sparse_attention_overlap_mapped));
    module.impl("prefill_dma_copy", TORCH_FN(sglang::npu_kernel::PrefillDmaCopy));
    module.impl("unidex_copy", TORCH_FN(sglang::npu_kernel::unidex_copy));
    module.impl("fsa_enqueue_lru_plan", TORCH_FN(sglang::npu_kernel::enqueue_fsa_lru_plan));
    module.impl("fsa_enqueue_clock_plan", TORCH_FN(sglang::npu_kernel::enqueue_fsa_clock_plan));
    module.impl("npu_fused_sparse_attention_overlap", TORCH_FN(sglang::npu_kernel::fused_sparse_attention_overlap));
}
