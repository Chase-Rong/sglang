#pragma once
#include <cstdint>

namespace sglang::npu_kernel {
// "FSA", shared unique-copy contract version3, Torch argument count30.
inline constexpr uint64_t kFusedSparseAttentionOverlapAbi = 0x465341000003001eULL;
inline constexpr const char *kFusedSparseAttentionOverlapAbiSymbol =
    "sglangFusedSparseAttentionOverlapAbi";
}
