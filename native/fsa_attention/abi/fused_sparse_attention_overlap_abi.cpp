#include "fused_sparse_attention_overlap_abi.h"

// This is an application-owned contract marker, not a CANN runtime API.
// It is linked into the same libcust_opapi that exports the ACL entry points.
extern "C" __attribute__((visibility("default"))) uint64_t
sglangFusedSparseAttentionOverlapAbi() noexcept {
    return sglang::npu_kernel::kFusedSparseAttentionOverlapAbi;
}
