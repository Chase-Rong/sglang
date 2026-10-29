#pragma once
#include <ATen/ATen.h>
#include <vector>
namespace sglang::npu_kernel {
std::vector<int64_t> ExternalPlanWarmupCapabilities();
void ExternalPlanFillOnly(at::Tensor &shared_kv, const at::Tensor &full_kv, int64_t full_kv_address,
                      const at::Tensor &full_block_table, const at::Tensor &committed_lengths,
                      const at::Tensor &request_slots, const at::Tensor &union_keys,
                      const at::Tensor &union_slots, const at::Tensor &union_count,
                      const at::Tensor &union_miss, const at::Tensor &union_owner,
                      const at::Tensor &union_cacheable, const at::Tensor &identity,
                      at::Tensor &ready, at::Tensor &status, at::Tensor &completion_identity,
                      at::Tensor &counters, int64_t resident_capacity, int64_t delay_cycles,
                      int64_t fail_request, int64_t abi_version);
} // namespace sglang::npu_kernel
