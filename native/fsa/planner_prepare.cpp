#include "planner_prepare.h"
#include "configured_planner.h"
#include <ATen/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <algorithm>
#include <array>
#include <climits>

namespace sglang::npu_kernel {
namespace {
void Check(const at::Tensor &t, const at::Tensor &reference,
           at::ScalarType dtype, at::IntArrayRef shape, const char *name) {
    TORCH_CHECK(t.device() == reference.device() && t.is_contiguous() &&
                t.scalar_type() == dtype && t.sizes().equals(shape),
                "external_plan_prepare invalid ", name);
}
bool Overlap(const at::Tensor &a, const at::Tensor &b) {
    auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
    auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
    return x < y + b.nbytes() && y < x + a.nbytes();
}
}
std::vector<int64_t> PlannerPrepareCapabilities() {
    // ABI, maxW, maxC, maxK, maxB, counterstride, configuredUB, maxUsedUB,
    // full-int64-physical-identity, protected_hash, first-occurrence-union.
    return {3, 16384, 16384, 2051, 1024, 8, 216 * 1024, 193 * 1024, 1, 1, 1};
}
void PlannerPrepare(at::TensorList i, at::TensorList o, int64_t C, int64_t R,
                    int64_t hostTokens, int64_t abi) {
    TORCH_CHECK(abi == 3 && i.size() == 14 && o.size() == 23,
                "external_plan_prepare ABI3 requires 14 inputs and 23 outputs");
    const auto &keys = i[0];
    TORCH_CHECK(keys.device().type() == at::DeviceType::PrivateUse1 && keys.dim() == 2 &&
                keys.size(0) > 0 && keys.size(1) > 0 && keys.size(1) <= 2051,
                "external_plan_prepare requires NPU physical keys [T,K]");
    c10::DeviceGuard guard(keys.device());
    auto stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(stream.device_index() == keys.device().index(), "planner current device mismatch");
    TORCH_CHECK(i[2].dim() == 1 && o[7].dim() == 2 && i[6].dim() == 2 && o[22].dim() == 3,
                "external_plan_prepare invalid ranks");
    int64_t T = keys.size(0), K = keys.size(1), B = i[2].numel(), W = o[7].size(1),
            Q = i[6].size(0), S = o[22].size(2);
    TORCH_CHECK(B > 0 && B <= 1024 && B <= T && R > 0 && R <= 6 &&
                C > 0 && C <= 16384 && C % 128 == 0 && C >= R * K &&
                W >= std::max<int64_t>(16, R * K) && W <= 16384 && !(W & (W - 1)) &&
                Q > 1 && Q <= 32767 && S > K && S % 8 == 0 && T <= 6144 && T <= INT32_MAX / K &&
                hostTokens > 0 && hostTokens < INT64_MAX,
                "external_plan_prepare unsupported geometry (C/W<=16384, routes<=6)");
    Check(keys, keys, at::kLong, {T, K}, "keys");
    Check(i[1], keys, at::kInt, {B}, "query ends");
    Check(i[2], keys, at::kLong, {B}, "request slots");
    Check(i[3], keys, at::kBool, {T, K}, "cacheability");
    Check(i[4], keys, at::kInt, {T, 1, K}, "logical topk");
    Check(i[5], keys, at::kInt, {B}, "KV lengths");
    Check(i[6], keys, at::kLong, {Q, C}, "identity");
    Check(i[7], keys, at::kBool, {Q, C}, "allocated");
    Check(i[8], keys, at::kBool, {Q, C}, "payload valid");
    Check(i[9], keys, at::kBool, {Q, C}, "reusable");
    Check(i[10], keys, at::kInt, {Q}, "inflight");
    Check(i[11], keys, at::kBool, {Q}, "blocked");
    Check(i[12], keys, at::kChar, {Q}, "release");
    TORCH_CHECK(i[13].numel() == 1 && i[13].scalar_type() == at::kInt &&
                i[13].device() == keys.device() && i[13].is_contiguous(), "invalid lifecycle error");
    Check(o[0], keys, at::kLong, {T, K}, "row slots");
    for (int n : {1, 2, 3}) Check(o[n], keys, at::kBool, {T, K}, "row flags");
    Check(o[4], keys, at::kLong, {T}, "row owners");
    Check(o[5], keys, at::kInt, {T}, "row requests");
    Check(o[6], keys, at::kInt, {T, K}, "row union");
    Check(o[7], keys, at::kLong, {B, W}, "union keys");
    for (int n : {8, 9}) Check(o[n], keys, at::kInt, {B, W}, "union slots/owners");
    for (int n : {10, 11}) Check(o[n], keys, at::kBool, {B, W}, "union flags");
    Check(o[12], keys, at::kInt, {B}, "union count");
    Check(o[13], keys, at::kBool, {B}, "overflow");
    Check(o[14], keys, at::kLong, {B, C}, "victims");
    Check(o[15], keys, at::kInt, {B, W}, "ready");
    Check(o[16], keys, at::kInt, {B}, "status");
    Check(o[17], keys, at::kLong, {B, 9}, "completion");
    Check(o[18], keys, at::kLong, {B, 8}, "counters");
    Check(o[19], keys, at::kInt, {T, 1, K}, "safe topk");
    Check(o[20], keys, at::kShort, {T, 16400}, "membership");
    Check(o[21], keys, at::kInt, {T, C / 128}, "selection table");
    Check(o[22], keys, at::kInt, {T, 1, S}, "selection status");
    for (int mutableInput : {9, 13})
        for (size_t n = 0; n < i.size(); ++n)
            if (n != mutableInput)
                TORCH_CHECK(!Overlap(i[mutableInput], i[n]), "planner mutable input alias");
    for (size_t n = 0; n < o.size(); ++n) {
        for (auto &input : i) TORCH_CHECK(!Overlap(o[n], input), "planner input/output alias");
        for (size_t p = 0; p < n; ++p) TORCH_CHECK(!Overlap(o[n], o[p]), "planner output alias");
    }
    // Capture only raw addresses and scalar geometry. Python/Graph owns every
    // reservation; allocator stream records cover an immediate eager caller.
    std::array<void *, 37> p{};
    for (size_t n = 0; n < i.size(); ++n) {
        p[n] = i[n].data_ptr();
        c10_npu::NPUCachingAllocator::recordStream(i[n].storage().data_ptr(), stream);
    }
    for (size_t n = 0; n < o.size(); ++n) {
        p[14 + n] = o[n].data_ptr();
        c10_npu::NPUCachingAllocator::recordStream(o[n].storage().data_ptr(), stream);
    }
    void *rawStream = stream.stream(false);
    uint32_t cores = std::min<int64_t>(8, (B + 1) / 2);
    auto launch = [=]() -> int {
        auto status = external_plan_configured(cores, rawStream,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],
            p[14], p[15], p[16], p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24], p[25],
            p[26], p[27], p[28], p[29], p[30], p[31], p[32], p[33], p[34], p[35], p[36],
            static_cast<uint32_t>(B), static_cast<uint32_t>(T), static_cast<uint32_t>(K),
            static_cast<uint32_t>(W), static_cast<uint32_t>(C), static_cast<uint32_t>(Q),
            static_cast<uint32_t>(R), static_cast<uint32_t>(S), hostTokens);
        TORCH_CHECK(status == 0, "external_plan_prepare enqueue failed: ", status);
        return status;
    };
    at_npu::native::OpCommand::RunOpApi("external_plan_prepare", launch);
}
}
