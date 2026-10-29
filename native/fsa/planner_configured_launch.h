// Derived from native/lim_copy_sfa/configured_launch.h; independently owned
// binary/context registry. No LIM symbols, binary, or mutable state are reused.
#pragma once
#include <acl/acl_rt.h>
#include <map>
#include <mutex>
#include <string>
namespace ExternalPlanHost {
inline aclError Launch(const void *binary, size_t size, uint32_t blocks,
                       void *stream, const void *args, uint32_t argSize,
                       const char *functionName) {
    struct Entry { aclrtBinHandle binary; aclrtFuncHandle function; };
    static std::mutex mutex;
    static std::map<std::pair<aclrtContext, std::string>, Entry> entries;
    aclrtContext context = nullptr;
    auto status = aclrtGetCurrentContext(&context);
    if (status != ACL_SUCCESS) return status;
    aclrtFuncHandle function = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = std::make_pair(context, std::string(functionName));
        auto found = entries.find(key);
        if (found == entries.end()) {
            Entry entry{};
            status = aclrtBinaryLoadFromData(binary, size, nullptr, &entry.binary);
            if (status != ACL_SUCCESS) return status;
            status = aclrtBinaryGetFunction(entry.binary, functionName, &entry.function);
            if (status != ACL_SUCCESS) { aclrtBinaryUnLoad(entry.binary); return status; }
            found = entries.emplace(key, entry).first;
        }
        function = found->second.function;
    }
    aclrtLaunchKernelAttr attrs[2]{};
    attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attrs[0].value.dynUBufSize = 216U * 1024U;
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[1].value.schemMode = 1;
    aclrtLaunchKernelCfg config{attrs, 2};
    return aclrtLaunchKernelWithHostArgs(function, blocks, stream, &config,
        const_cast<void *>(args), argSize, nullptr, 0);
}
}
