#pragma once

#include <ATen/ATen.h>
#include <c10/util/string_view.h>
#include <tuple>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace sglang::npu_kernel {

std::tuple<int64_t, int64_t> shm_allocator_create_and_register(int64_t, int64_t, c10::string_view);
void shm_allocator_free_all(int64_t);
void shm_allocator_free_by_name(c10::string_view, int64_t);
std::tuple<int64_t, int64_t, int64_t, int64_t> shm_allocator_export(c10::string_view, int64_t);
std::tuple<int64_t, int64_t> shm_allocator_attach(int64_t, int64_t, int64_t, int64_t, int64_t, c10::string_view);
void unidex_copy(const at::Tensor &, at::Tensor &, const at::Tensor &, const at::Tensor &,
                 const at::Tensor &, int64_t, int64_t, int64_t, int64_t, int64_t,
                 c10::optional<int64_t>, c10::optional<int64_t>);

inline void *FsaKernelArgument(const at::Tensor &tensor)
{
    return const_cast<void *>(tensor.data_ptr());
}

template<typename Value>
inline Value FsaKernelArgument(Value value)
{
    return value;
}

template<typename... Values>
inline auto FsaKernelArguments(Values &...values)
{
    return std::make_tuple(FsaKernelArgument(values)...);
}

}

#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                                   \
    do {                                                                            \
        auto stream = c10_npu::getCurrentNPUStream().stream(false);                   \
        auto params = sglang::npu_kernel::FsaKernelArguments(__VA_ARGS__);             \
        auto launch = [stream, blockdim, params]() -> int {                           \
            std::apply([&](auto &&...args) {                                         \
                ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, stream, args...);          \
            }, params);                                                             \
            return 0;                                                               \
        };                                                                          \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, launch);                     \
    } while (false)
