#pragma once
#include <dlfcn.h>
#include "../fsa_attention/abi/fused_sparse_attention_overlap_abi.h"

namespace sglang::npu_kernel {
struct OverlapOppApi {
    void *get_workspace_size;
    void *launch;
};

// Kept independent of Torch/CANN so the exact production rejection logic can
// be exercised with real shared libraries in portable tests. The caller must
// pass the addresses returned by the resolver that will execute the call.
inline const char *ValidateOverlapOppAbi(const OverlapOppApi &api,
                                       void *marker) {
    if (!api.get_workspace_size || !api.launch || !marker)
        return "missing overlap entry point or ABI3 marker";
    Dl_info workspace_info{}, launch_info{}, marker_info{};
    if (dladdr(api.get_workspace_size, &workspace_info) == 0 ||
        dladdr(api.launch, &launch_info) == 0 ||
        dladdr(marker, &marker_info) == 0 ||
        workspace_info.dli_fbase == nullptr)
        return "cannot identify overlap symbol providers";
    if (workspace_info.dli_fbase != launch_info.dli_fbase ||
        workspace_info.dli_fbase != marker_info.dli_fbase)
        return "overlap entry points and ABI3 marker come from different libraries";
    using QueryContract = uint64_t (*)();
    if (reinterpret_cast<QueryContract>(marker)() != kFusedSparseAttentionOverlapAbi)
        return "overlap OPP contract is incompatible with ABI3";
    return nullptr;
}
}
