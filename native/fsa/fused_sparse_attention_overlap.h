#pragma once
#include <ATen/ATen.h>
#include <vector>
namespace sglang::npu_kernel {
at::Tensor FusedSparseAttentionOverlap(
    const at::Tensor & query,
    at::Tensor & shared_kv,
    const at::Tensor & full_kv,
    int64_t full_kv_address,
    const at::Tensor & full_block_table,
    const at::Tensor & full_kv_actual_seq,
    const at::Tensor & full_q_actual_seq,
    const at::Tensor & logical_topk,
    const at::Tensor & request_slots,
    const at::Tensor & row_to_request,
    const at::Tensor & row_union_index,
    const at::Tensor & union_keys,
    const at::Tensor & union_slots,
    const at::Tensor & union_count,
    const at::Tensor & union_miss,
    const at::Tensor & union_owner,
    const at::Tensor & union_cacheable,
    const at::Tensor & identity,
    at::Tensor & ready,
    at::Tensor & status,
    at::Tensor & completion_identity,
    at::Tensor & counters,
    at::Tensor & safe_topk,
    at::Tensor & membership,
    at::Tensor & selection_table,
    at::Tensor & selection_status,
    at::Tensor & workspace,
    int64_t resident_capacity,
    double scale,
    int64_t abi_version);
std::vector<int64_t> SharedV3Capabilities();
}
