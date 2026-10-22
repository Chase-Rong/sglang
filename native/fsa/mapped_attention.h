#pragma once

#include <torch/extension.h>

namespace sglang::npu_kernel {
at::Tensor fused_sparse_attention_overlap_mapped(
    const at::Tensor &query, const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache, const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status, const at::Tensor &selection_membership_map,
    const at::Tensor &selection_topk_indices, const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache, const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq, const c10::optional<at::Tensor> &full_q_actual_seq,
    double scale_value, int64_t sparse_block_size, int64_t selection_topk_block_size,
    c10::string_view layout_query, c10::string_view layout_kv, int64_t sparse_mode,
    c10::optional<int64_t> full_rope_address, c10::optional<int64_t> full_kv_address,
    c10::optional<int64_t> membership_address);
}
