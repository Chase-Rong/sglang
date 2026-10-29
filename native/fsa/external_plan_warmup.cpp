// Standalone fill-only helper; it does not load an Attention OPP.
#include "external_plan_warmup.h"
#include "aclrtlaunch_external_plan_warmup.h"
namespace sglang::npu_kernel {
std::vector<int64_t> ExternalPlanWarmupCapabilities() {
    SharedWorkspaceBytes(); // Validate the supported platform, without resolving an OPP.
    return {1, 1, 512, 0, 128, 16384, 32767, 1024, 9, 1};
}
void ExternalPlanFillOnly(at::Tensor &shared_kv, const at::Tensor &full_kv, int64_t full_kv_address,
                      const at::Tensor &full_block_table, const at::Tensor &committed_lengths,
                      const at::Tensor &request_slots, const at::Tensor &union_keys,
                      const at::Tensor &union_slots, const at::Tensor &union_count,
                      const at::Tensor &union_miss, const at::Tensor &union_owner,
                      const at::Tensor &union_cacheable, const at::Tensor &identity,
                      at::Tensor &ready, at::Tensor &status, at::Tensor &completion_identity,
                      at::Tensor &counters, int64_t resident_capacity, int64_t delay_cycles,
                      int64_t fail_request, int64_t abi_version) {
    SharedWorkspaceBytes();
    TORCH_CHECK(abi_version == 1, "external_plan fill_only ABI must be 1");
    TORCH_CHECK(shared_kv.device().type() == at::DeviceType::PrivateUse1 && shared_kv.dim() == 3 &&
                    shared_kv.is_contiguous() && shared_kv.scalar_type() == at::kBFloat16,
                "external_plan fill_only BF16 NPU payload required");
    TORCH_CHECK(c10_npu::getCurrentNPUStream().device_index() == shared_kv.device().index(),
                "fill_only current device mismatch");
    auto C = resident_capacity;
    TORCH_CHECK(C > 0 && C <= 16384 && C % 128 == 0 && shared_kv.size(0) > 0 &&
                    shared_kv.size(0) % (C / 128) == 0,
                "fill_only invalid C");
    auto Q = shared_kv.size(0) / (C / 128);
    TORCH_CHECK(Q > 1 && Q <= 32767 && request_slots.dim() == 1 && request_slots.numel() > 0 &&
                    request_slots.numel() <= 1024,
                "fill_only invalid Q/B");
    auto B = request_slots.numel(), W = C;
    TORCH_CHECK(full_block_table.dim() == 2 && full_block_table.size(1) > 0,
                "fill_only requires page table");
    auto N = full_block_table.size(1);
    CheckTensor(shared_kv, shared_kv, at::kBFloat16, {Q * C / 128, 128, 512}, "shared_kv");
    TORCH_CHECK(full_kv.device().is_cpu() && full_kv.is_contiguous() &&
                    full_kv.storage_offset() == 0 && full_kv.scalar_type() == at::kBFloat16 &&
                    full_kv.dim() == 3 && full_kv.size(0) > 0 && full_kv.size(1) == 128 &&
                    full_kv.size(2) == 512 && full_kv_address > 0 &&
                    full_kv.size(0) <= INT64_MAX / (128 * 512 * 2) &&
                    full_kv_address <= INT64_MAX - full_kv.nbytes(),
                "fill_only registered Host payload invalid");
    TORCH_CHECK(shm_validate_mapping(full_kv.data_ptr(), full_kv_address, full_kv.nbytes(),
                                     shared_kv.device().index()) == 0 ||
                    ValidateMemFabricRegion(full_kv.data_ptr(), full_kv_address, full_kv.nbytes(),
                                            shared_kv.device().index()),
                "fill_only no live Host registration");
    CheckTensor(full_block_table, shared_kv, at::kInt, {B, N}, "full_block_table");
    CheckTensor(committed_lengths, shared_kv, at::kInt, {B}, "committed_lengths");
    CheckTensor(request_slots, shared_kv, at::kLong, {B}, "request_slots");
    CheckTensor(union_keys, shared_kv, at::kLong, {B, W}, "union_keys");
    CheckTensor(union_slots, shared_kv, at::kInt, {B, W}, "union_slots");
    CheckTensor(union_count, shared_kv, at::kInt, {B}, "union_count");
    CheckTensor(union_miss, shared_kv, at::kBool, {B, W}, "union_miss");
    CheckTensor(union_owner, shared_kv, at::kInt, {B, W}, "union_owner");
    CheckTensor(union_cacheable, shared_kv, at::kBool, {B, W}, "union_cacheable");
    CheckTensor(identity, shared_kv, at::kLong, {B, 9}, "identity");
    CheckTensor(ready, shared_kv, at::kInt, {B, W}, "ready");
    CheckTensor(status, shared_kv, at::kInt, {B}, "status");
    CheckTensor(completion_identity, shared_kv, at::kLong, {B, 9}, "completion_identity");
    CheckTensor(counters, shared_kv, at::kLong, {B, 2}, "counters");
    TORCH_CHECK(delay_cycles >= 0 && delay_cycles <= 100000000 && fail_request >= -1 &&
                    fail_request < B,
                "fill_only invalid injection");
    std::vector<at::Tensor> mutable_buffers = {shared_kv, ready, status, completion_identity,
                                               counters};
    std::vector<at::Tensor> inputs = {
        full_kv,     full_block_table, committed_lengths, request_slots,   union_keys, union_slots,
        union_count, union_miss,       union_owner,       union_cacheable, identity};
    for (size_t i = 0; i < mutable_buffers.size(); ++i) {
        for (auto &input : inputs)
            TORCH_CHECK(!Overlaps(mutable_buffers[i], input), "fill_only mutable/input alias");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(!Overlaps(mutable_buffers[i], mutable_buffers[j]),
                        "fill_only workspace alias");
    }
    auto source = reinterpret_cast<void *>(full_kv_address);
    auto host_tokens = full_kv.size(0) * 128;
    uint32_t blocks = 1;
    EXEC_SHARED_KERNEL_CMD(external_plan_warmup, blocks, source, shared_kv, full_block_table,
                           committed_lengths, request_slots, union_keys, union_slots, union_count,
                           union_miss, union_owner, union_cacheable, identity, ready, status,
                           completion_identity, counters, B, W, Q, C, N, host_tokens, delay_cycles,
                           fail_request);
}
} // namespace sglang::npu_kernel
