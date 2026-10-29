// Included once by fused_sparse_attention_overlap.cpp: donor ACL helpers have non-inline definitions.
#include "shared_runtime.h"
#include "shared_lifetime.h"
#include "torch_helper.h"
#include "external_host_mapping.h"
#include "sparsity_driven_kv_offload/shm_allocator/shm_allocator.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "fused_sparse_attention_overlap/registered_acl_tensor.h"
#include <dlfcn.h>
#include <cmath>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include <atomic>
#include <memory>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

// Shared launches must propagate enqueue failures: a failed reset must never
// allow Attention to consume the previous execution's READY/status values.
#define EXEC_SHARED_KERNEL_CMD(kernel_name, blockdim, ...) \
 do { \
   auto stream=c10_npu::getCurrentNPUStream().stream(false); \
   auto params=sglang::npu_kernel::FsaKernelArguments(__VA_ARGS__); \
   auto launch=[stream,blockdim,params]() -> int { \
     auto result=std::apply([&](auto &&...args) { \
       return ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim,stream,args...); \
     },params); \
     TORCH_CHECK(result==ACL_SUCCESS,"external_plan enqueue failed: ",#kernel_name," code ",result); \
     return static_cast<int>(result); \
   }; \
   at_npu::native::OpCommand::RunOpApi(#kernel_name,launch); \
 } while(false)

#define PREPARE_SHARED_NPU_CMD(aclnn_api, verified_api, ...)                                          \
    const auto getWorkspaceSizeFuncAddr =                              \
        (verified_api).get_workspace_size;                      \
    const auto opApiFuncAddr = (verified_api).launch;           \
    static const auto initMemAddr =                                           \
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           \
    static const auto unInitMemAddr =                                         \
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    TORCH_CHECK(                                                              \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,      \
        #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",        \
        GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);           \
    uint64_t workspace_size = 0;                                              \
    uint64_t *workspace_size_addr = &workspace_size;                          \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executor_addr = &executor;                                \
    InitHugeMemThreadLocal initMemFunc =                                      \
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
    UnInitHugeMemThreadLocal unInitMemFunc =                                  \
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
    auto thread_cleanup = sglang::npu_kernel::SharedScopeExit([unInitMemFunc]() { \
      if (unInitMemFunc) unInitMemFunc(nullptr,false); \
    }); \
    auto call_lifetime = std::make_shared<sglang::npu_kernel::SharedAclLifetime>([]() { \
      auto release = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr); \
      if (release) release(nullptr,false); \
    }); \
    if (initMemFunc) {                                                        \
      initMemFunc(nullptr, false);                                            \
    }                                                                         \
    auto converted_params =                                                   \
        ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);        \
    call_lifetime->SetCleanup([converted_params]() mutable { \
      ReleaseConvertTypes(converted_params); \
      auto release=reinterpret_cast<ReleaseHugeMem>(releaseMemAddr); \
      if(release)release(nullptr,false); \
    }); \
    static auto getWorkspaceSizeFunc =                                        \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);     \
    TORCH_CHECK(workspace_status == 0,                                        \
                "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
    TORCH_CHECK(workspace_size <= static_cast<uint64_t>(INT64_MAX), "shared workspace size overflow"); \
    TORCH_CHECK(workspace_size<=static_cast<uint64_t>(workspace_bound),"shared OPP workspace exceeds startup platform bound"); \
    sglang::npu_kernel::shared_workspace_bytes.store(static_cast<int64_t>(workspace_size)); \
    TORCH_CHECK(workspace_size<=static_cast<uint64_t>(workspace.numel()), \
                "external_plan pre-reserved workspace too small: ",workspace_size," > ",workspace.numel()); \
    void *workspace_addr=workspace.data_ptr(); \
    auto acl_call = [call_lifetime, workspace_addr, workspace_size,        \
                     acl_stream, executor, opApiFuncAddr]() -> int {                         \
      call_lifetime->EnterHandler(); \
      sglang::npu_kernel::SharedScopeExit acl_cleanup([call_lifetime]() {call_lifetime->Cleanup();}); \
      typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                               const aclrtStream);                            \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);       \
      auto api_ret =                                                          \
          opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",        \
                  aclGetRecentErrMsg());                                      \
      return api_ret;                                                         \
    };                                                                        \
    at_npu::native::OpCommand cmd;                                            \
    cmd.Name(#aclnn_api);                                                     \
    cmd.SetCustomHandler(acl_call);                                           \
    /* submission occurs only after host validation and workspace preflight */


namespace sglang::npu_kernel {
std::atomic<int64_t> shared_workspace_bytes{0};
int64_t SharedLastWorkspaceBytes() { return shared_workspace_bytes.load(); }
int64_t SharedWorkspaceBytes() {
    const char *soc=aclrtGetSocName();
    TORCH_CHECK(soc && std::strcmp(soc,"Ascend950DT_9582")==0,
                "external_plan currently supports Ascend950DT_9582; queried SoC is ",soc?soc:"<null>");
    auto *platform=platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    TORCH_CHECK(platform && platform->GetSocVersion()==platform_ascendc::SocVersion::ASCEND950,
                "external_plan CANN platform query failed for ",soc);
    auto cores=platform->GetCoreNumAic();
    TORCH_CHECK(cores>0,"external_plan CANN platform reported zero AIC cores");
    // Exact A5 branch of the shared OPP GetWorkspaceSize (H32, BF16).
    return static_cast<int64_t>(platform->GetLibApiWorkSpaceSize()) +
        int64_t(128)*576*2*3*cores;
}
namespace {
void *SharedSymbol(const char *name) {
    // Same resolver as EXEC_NPU_CMD: this checks the actually selected custom OPP.
    return GetOpApiFuncAddr(name);
}
void CheckTensor(const at::Tensor &t, const at::Tensor &q, at::ScalarType dtype,
                 std::initializer_list<int64_t> shape, const char *name) {
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == dtype && t.is_contiguous(),
                "external_plan ", name, " requires contiguous matching NPU/dtype");
    TORCH_CHECK(t.sizes().equals(at::IntArrayRef(shape.begin(),shape.size())),
                "external_plan ", name, " has incorrect shape");
}
bool Overlaps(const at::Tensor &a, const at::Tensor &b) {
    if (a.device() != b.device() || a.nbytes()==0 || b.nbytes()==0) return false;
    auto x = reinterpret_cast<uintptr_t>(a.data_ptr()), y = reinterpret_cast<uintptr_t>(b.data_ptr());
    return x <= y ? y-x < a.nbytes() : x-y < b.nbytes();
}
}
} // namespace sglang::npu_kernel
