#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "aclrtlaunch_sgl_lim_copy_sfa.h"

namespace {
void CopyOut(const at::Tensor &query, const at::Tensor &host, at::Tensor resident,
    const at::Tensor &ids, const at::Tensor &ends, const at::Tensor &lengths,
    const at::Tensor &kvLengths, const at::Tensor &table, const at::Tensor &requests,
    at::Tensor membership, const at::Tensor &unionSources, const at::Tensor &unionSlots,
    const at::Tensor &unionPhysical, at::Tensor physical, const at::Tensor &state,
    at::Tensor output, at::Tensor workspace, int64_t hostAddress, int64_t mutableTail, double scale,
    int64_t cores)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1 && query.dim() == 3 &&
        query.size(0) > 0 && query.size(1) > 0 && query.size(1) <= 64 && query.size(2) == 512,
        "CopySFA requires NPU TND queries, 1..64 heads and NoPE512");
    for (const auto &tensor : {query, resident, output}) {
        TORCH_CHECK(tensor.device() == query.device() && tensor.scalar_type() == at::kBFloat16 &&
            tensor.is_contiguous(), "CopySFA query/resident/output must be contiguous BF16 on the same NPU");
    }
    for (const auto &tensor : {ids, ends, lengths, kvLengths, table, unionSources, unionSlots}) {
        TORCH_CHECK(tensor.device() == query.device() && tensor.scalar_type() == at::kInt &&
            tensor.is_contiguous(), "CopySFA requires contiguous int32 metadata on the query device");
    }
    for (const auto &tensor : {requests, unionPhysical, physical, state}) {
        TORCH_CHECK(tensor.device() == query.device() && tensor.scalar_type() == at::kLong &&
            tensor.is_contiguous(), "CopySFA requires contiguous int64 state on the query device");
    }
    TORCH_CHECK(host.scalar_type() == at::kBFloat16 && host.is_contiguous() && host.numel() > 0 &&
        host.numel() % 512 == 0 && hostAddress > 0 && hostAddress % 32 == 0,
        "CopySFA requires a registered aligned Host KV address");
    TORCH_CHECK(output.sizes() == query.sizes() && ids.dim() == 2 && ids.size(0) == query.size(0) &&
        ids.size(1) == 2051 && lengths.dim() == 1 && lengths.numel() == query.size(0), "Invalid CopySFA rows");
    TORCH_CHECK(ends.dim() == 1 && ends.numel() > 0 && requests.sizes() == ends.sizes() &&
        kvLengths.sizes() == ends.sizes() && table.dim() == 2 && table.size(0) == ends.numel() && table.size(1) > 0,
        "Invalid CopySFA request metadata");
    TORCH_CHECK(physical.dim() == 2 && physical.size(0) > 1 && physical.size(1) > 0 &&
        physical.size(1) <= 12416 && physical.size(1) % 128 == 0 &&
        resident.numel() == physical.numel() * 512 && state.dim() == 2 &&
        state.size(0) == physical.size(0) && state.size(1) == 16, "Invalid CopySFA resident state");
    TORCH_CHECK(membership.device() == query.device() && membership.scalar_type() == at::kShort &&
        membership.is_contiguous() && membership.dim() == 2 && membership.size(0) == query.size(0) &&
        membership.size(1) == 16400, "Invalid CopySFA signed-slot plan");
    TORCH_CHECK(unionSources.dim() == 2 && unionSources.size(0) == ends.numel() &&
        (unionSources.size(1) == 4096 || unionSources.size(1) == 16384) &&
        unionSlots.sizes() == unionSources.sizes() && unionPhysical.sizes() == unionSources.sizes(),
        "Invalid CopySFA union");
    TORCH_CHECK(cores > 0 && cores <= 64, "CopySFA core count exceeds the qualified A5 range");
    TORCH_CHECK(workspace.device() == query.device() && workspace.scalar_type() == at::kByte &&
        workspace.is_contiguous() && workspace.numel() >= cores * 3 * 128 * 576 * 2 +
            (query.size(0) * 4 + 31) / 32 * 32 + query.size(0) * 512 * 512 * 2,
        "CopySFA workspace is too small");
    TORCH_CHECK(mutableTail >= 0 && mutableTail <= 32 && std::isfinite(scale), "Invalid CopySFA attributes");
    c10::DeviceGuard guard(query.device());
    auto npuStream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : {query, resident, ids, ends, lengths, kvLengths, table, requests,
            membership, unionSources, unionSlots, unionPhysical, physical, state, output, workspace})
        tensor.record_stream(npuStream);
    const auto stream = npuStream.stream(false);
    auto launch = [=]() -> int {
        return ACLRT_LAUNCH_KERNEL(sgl_lim_copy_sfa)(cores, stream, query.data_ptr(),
            reinterpret_cast<void *>(hostAddress), resident.data_ptr(), ids.data_ptr(), ends.data_ptr(),
            lengths.data_ptr(), kvLengths.data_ptr(), table.data_ptr(), requests.data_ptr(), membership.data_ptr(),
            unionSources.data_ptr(), unionSlots.data_ptr(), unionPhysical.data_ptr(), physical.data_ptr(),
            state.data_ptr(), output.data_ptr(), workspace.data_ptr(), static_cast<uint32_t>(query.size(0)),
            static_cast<uint32_t>(ends.numel()), static_cast<uint32_t>(query.size(1)),
            static_cast<uint32_t>(physical.size(1)), static_cast<uint32_t>(table.size(1)),
            static_cast<uint32_t>(unionSources.size(1)), static_cast<uint32_t>(mutableTail),
            static_cast<uint32_t>(physical.size(0)), static_cast<float>(scale));
    };
    at_npu::native::OpCommand::RunOpApi("sgl_lim_copy_sfa", launch);
}

void CopyMeta(const at::Tensor &, const at::Tensor &, at::Tensor, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, at::Tensor, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    at::Tensor, const at::Tensor &, at::Tensor, at::Tensor, int64_t, int64_t, double, int64_t) {}
}

TORCH_LIBRARY_FRAGMENT(sgl_lim_native, module)
{
    module.def("copy_sfa_out(Tensor query, Tensor host, Tensor(a!) resident, Tensor ids, Tensor ends, "
        "Tensor lengths, Tensor kv_lengths, Tensor table, Tensor requests, Tensor(b!) membership, "
        "Tensor union_sources, Tensor union_slots, Tensor union_physical, Tensor(c!) physical, Tensor state, "
        "Tensor(d!) output, Tensor(e!) workspace, int host_address, int mutable_tail, float scale, int cores=8) -> ()");
}
TORCH_LIBRARY_IMPL(sgl_lim_native, PrivateUse1, module) { module.impl("copy_sfa_out", TORCH_FN(CopyOut)); }
TORCH_LIBRARY_IMPL(sgl_lim_native, Meta, module) { module.impl("copy_sfa_out", TORCH_FN(CopyMeta)); }
