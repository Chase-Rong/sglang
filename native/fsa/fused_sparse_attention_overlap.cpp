#include "fused_sparse_attention_overlap/fused_sparse_attention_overlap.h"

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "fused_sparse_attention_overlap_torch_adpt.h"
#include "sparsity_driven_kv_offload/shm_allocator/shm_allocator.h"
#include "external_host_mapping.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace sglang {
namespace npu_kernel {

namespace {

void ValidateFullSource(const at::Tensor &source, const at::Tensor &query,
                        c10::optional<int64_t> device_address)
{
    TORCH_CHECK(source.is_contiguous() && source.dim() == 3,
                "FSA full source must be contiguous [pages,page_size,dim]");
    TORCH_CHECK(source.scalar_type() == query.scalar_type(), "FSA source and query dtype differ");
    if (!device_address.has_value()) {
        TORCH_CHECK(source.device() == query.device(), "FSA Host source requires a registered address");
        return;
    }
    TORCH_CHECK(source.device().is_cpu() && source.storage_offset() == 0 && *device_address > 0,
                "FSA registered source requires base-aligned CPU metadata");
    TORCH_CHECK(shm_validate_mapping(source.data_ptr(), static_cast<uint64_t>(*device_address),
                                    source.nbytes(), query.device().index()) == 0 ||
                ValidateMemFabricRegion(source.data_ptr(), static_cast<uint64_t>(*device_address),
                                        source.nbytes(), query.device().index()),
                "FSA source does not match a live Host registration on this NPU");
}

}

at::Tensor fused_sparse_attention_overlap(
    const at::Tensor &query,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_membership_map,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const c10::optional<at::Tensor> &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    c10::optional<int64_t> full_rope_address,
    c10::optional<int64_t> full_kv_address) {
    TORCH_CHECK(query.device().type() == at::DeviceType::PrivateUse1,
                "FSA query must be an NPU tensor");
    TORCH_CHECK(c10_npu::getCurrentNPUStream().device_index() == query.device().index(),
                "FSA query must match the current NPU stream device");
    TORCH_CHECK(full_k_rope.dim() == 3 && selection_k_rope.dim() == 3,
                "FSA RoPE metadata must have shape [pages,page_size,dim]");
    TORCH_CHECK(full_k_rope.size(2) == selection_k_rope.size(2),
                "FSA full and selection RoPE dimensions must match");
    if (full_k_rope.size(2) > 0) {
        TORCH_CHECK(full_rope_address.has_value() == full_kv_address.has_value(),
                    "FSA latent and RoPE sources must use the same memory strategy");
        ValidateFullSource(full_k_rope, query, full_rope_address);
    } else {
        TORCH_CHECK(!full_rope_address.has_value(),
                    "FSA NoPE does not accept a registered RoPE address");
    }
    ValidateFullSource(full_kv_cache, query, full_kv_address);
    const auto q_actual = full_q_actual_seq.has_value()
        ? full_q_actual_seq.value()
        : at::Tensor();
    return vllm_ascend::npu_fused_sparse_attention_overlap(
        query, selection_k_rope, selection_kv_cache,
        selection_kv_block_table, selection_kv_block_status,
        selection_membership_map, selection_topk_indices, full_k_rope,
        full_kv_cache, full_kv_block_table, full_kv_actual_seq, q_actual,
        scale_value, sparse_block_size, selection_topk_block_size,
        layout_query, layout_kv, sparse_mode, full_rope_address, full_kv_address);
}

}  // namespace npu_kernel
}  // namespace sglang
