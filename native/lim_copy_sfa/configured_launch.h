#pragma once

#include <acl/acl_rt.h>
#include <map>
#include <mutex>
#include <string>

namespace SglLimHost {
inline aclError LaunchConfiguredIndexer(const void *binary, size_t binarySize,
    uint32_t blocks, void *stream, const void *arguments, uint32_t size,
    const char *functionName = "sgl_lim_indexer_manage_2")
{
    struct Entry {
        aclrtBinHandle binary;
        aclrtFuncHandle function;
    };
    static std::mutex mutex;
    static std::map<std::pair<aclrtContext, std::string>, Entry> entries;
    aclrtContext context = nullptr;
    aclError status = aclrtGetCurrentContext(&context);
    if (status != ACL_SUCCESS) return status;
    aclrtFuncHandle function = nullptr;
    {
        std::lock_guard<std::mutex> guard(mutex);
        auto key = std::make_pair(context, std::string(functionName));
        auto found = entries.find(key);
        if (found == entries.end()) {
            Entry entry{};
            status = aclrtBinaryLoadFromData(binary, binarySize, nullptr, &entry.binary);
            if (status != ACL_SUCCESS) return status;
            status = aclrtBinaryGetFunction(entry.binary, functionName, &entry.function);
            if (status != ACL_SUCCESS) {
                aclrtBinaryUnLoad(entry.binary);
                return status;
            }
            found = entries.emplace(key, entry).first;
        }
        function = found->second.function;
    }
    aclrtLaunchKernelAttr attributes[2]{};
    attributes[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attributes[0].value.dynUBufSize = 216U * 1024U;
    attributes[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attributes[1].value.schemMode = 1;
    aclrtLaunchKernelCfg config{attributes, 2};
    return aclrtLaunchKernelWithHostArgs(function, blocks, stream, &config,
        const_cast<void *>(arguments), size, nullptr, 0);
}
}
