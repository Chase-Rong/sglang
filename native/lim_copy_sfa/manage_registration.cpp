#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "configured_indexer.h"
#include "validation.h"

namespace SglLimHost {
void CheckTensor(const at::Tensor &tensor, const at::Tensor &reference, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.device() == reference.device() && tensor.scalar_type() == dtype && tensor.is_contiguous(),
        "Native resident inputs must have matching device, dtype and contiguous storage");
}

void ValidateManager(const at::Tensor &ids, const at::Tensor &scores, const at::Tensor &ends,
    const at::Tensor &lengths, const at::Tensor &table, const at::Tensor &requests,
    const at::Tensor &generations, at::Tensor mapping, at::Tensor sources, at::Tensor physical,
    at::Tensor state, at::Tensor outputIds, at::Tensor outputSlots, at::Tensor membership,
    at::Tensor unionSources, at::Tensor unionSlots, at::Tensor unionPhysical,
    int64_t hostTokens, int64_t mutableTail, bool pooled, bool encoded)
{
    TORCH_CHECK(ids.device().type() == c10::DeviceType::PrivateUse1, "Native resident manager requires NPU");
    for (const auto &tensor : {ids, ends, lengths, table, mapping, sources, outputIds, outputSlots, unionSources, unionSlots})
        CheckTensor(tensor, ids, at::kInt);
    for (const auto &tensor : {requests, generations, physical, state, unionPhysical}) CheckTensor(tensor, ids, at::kLong);
    CheckTensor(membership, ids, at::kShort);
    CheckTensor(scores, ids, encoded ? at::kInt : at::kFloat);
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) == (pooled ? 512 : 2051), "Invalid selected IDs");
    TORCH_CHECK(ends.dim() == 1 && ends.numel() > 0 && requests.sizes() == ends.sizes(), "Invalid request rows");
    TORCH_CHECK(lengths.dim() == 1 && lengths.numel() == ids.size(0), "Invalid causal lengths");
    TORCH_CHECK(mapping.dim() == 2 && mapping.size(0) > 1 && mapping.size(1) > 0 && mapping.size(1) <= (1 << 21),
        "Invalid source map");
    TORCH_CHECK(sources.dim() == 2 && sources.size(0) == mapping.size(0) && sources.size(1) % 128 == 0 &&
        sources.size(1) >= 128 && sources.size(1) <= 12416, "Native resident capacity must be page aligned and <=12416");
    TORCH_CHECK(physical.sizes() == sources.sizes() && state.dim() == 2 && state.size(0) == mapping.size(0) &&
        state.size(1) == 16 && generations.dim() == 1 && generations.numel() == mapping.size(0), "Invalid resident state");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == ends.numel() && table.size(1) > 0, "Invalid Host page table");
    TORCH_CHECK(scores.dim() == 2 && scores.size(0) == ids.size(0) && scores.size(1) * 4 >= mapping.size(1),
        "Scores do not cover the source map");
    TORCH_CHECK(outputIds.dim() == 2 && outputIds.size(0) == ids.size(0) && outputIds.size(1) == 2051 &&
        outputSlots.sizes() == outputIds.sizes(), "Invalid output rows");
    TORCH_CHECK(membership.dim() == 2 && membership.size(0) == ids.size(0) && membership.size(1) == 16400,
        "Invalid encoded plan storage");
    TORCH_CHECK(unionSources.dim() == 2 && unionSources.size(0) == ends.numel() &&
        (unionSources.size(1) == 4096 || unionSources.size(1) == 16384) &&
        unionSlots.sizes() == unionSources.sizes() && unionPhysical.sizes() == unionSources.sizes(), "Invalid union storage");
    TORCH_CHECK(hostTokens > 0 && mutableTail >= 0 && mutableTail <= 32, "Invalid Host bounds or mutable tail");
}

void ManageOut(const at::Tensor &ids, const at::Tensor &scores, const at::Tensor &ends,
    const at::Tensor &lengths, const at::Tensor &table, const at::Tensor &requests,
    const at::Tensor &generations, at::Tensor mapping, at::Tensor sources, at::Tensor physical,
    at::Tensor state, at::Tensor outputIds, at::Tensor outputSlots, at::Tensor membership,
    at::Tensor unionSources, at::Tensor unionSlots, at::Tensor unionPhysical,
    int64_t hostTokens, int64_t mutableTail, bool pooled, bool encoded)
{
    ValidateManager(ids, scores, ends, lengths, table, requests, generations, mapping, sources,
        physical, state, outputIds, outputSlots, membership, unionSources, unionSlots,
        unionPhysical, hostTokens, mutableTail, pooled, encoded);
    c10::DeviceGuard guard(ids.device());
    auto npuStream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : {ids, scores, ends, lengths, table, requests, generations, mapping, sources,
            physical, state, outputIds, outputSlots, membership, unionSources, unionSlots, unionPhysical})
        tensor.record_stream(npuStream);
    const auto stream = npuStream.stream(false);
    const uint32_t cores = std::min<int64_t>(8, ends.numel());
    auto launch = [=]() -> int {
        return sgl_lim_configured_manage(cores, stream,
            ids.data_ptr(), scores.data_ptr(), ends.data_ptr(), lengths.data_ptr(), table.data_ptr(),
            requests.data_ptr(), generations.data_ptr(), mapping.data_ptr(), sources.data_ptr(), physical.data_ptr(),
            state.data_ptr(), outputIds.data_ptr(), outputSlots.data_ptr(), membership.data_ptr(),
            unionSources.data_ptr(), unionSlots.data_ptr(), unionPhysical.data_ptr(),
            static_cast<uint32_t>(ids.size(0)), static_cast<uint32_t>(ends.numel()),
            static_cast<uint32_t>(mapping.size(0)), static_cast<uint32_t>(mapping.size(1)),
            static_cast<uint32_t>(sources.size(1)), static_cast<uint32_t>(table.size(1)),
            static_cast<uint32_t>(unionSources.size(1)), static_cast<uint32_t>(scores.size(1)),
            static_cast<uint64_t>(hostTokens), static_cast<uint32_t>(mutableTail), pooled, encoded);
    };
    at_npu::native::OpCommand::RunOpApi("sgl_lim_manage", launch);
}

void ManageMeta(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    int64_t, int64_t, bool, bool)
{}
}

TORCH_LIBRARY_FRAGMENT(sgl_lim_native, module)
{
    module.def("manage_out(Tensor ids, Tensor scores, Tensor ends, Tensor lengths, Tensor table, "
        "Tensor requests, Tensor generations, Tensor(a!) mapping, Tensor(b!) sources, Tensor(c!) physical, "
        "Tensor(d!) state, Tensor(e!) output_ids, Tensor(f!) output_slots, Tensor(g!) membership, "
        "Tensor(h!) union_sources, Tensor(i!) union_slots, Tensor(j!) union_physical, "
        "int host_tokens, int mutable_tail, bool pooled=False, bool encoded=False) -> ()");
}
TORCH_LIBRARY_IMPL(sgl_lim_native, PrivateUse1, module)
{
    module.impl("manage_out", TORCH_FN(SglLimHost::ManageOut));
}
TORCH_LIBRARY_IMPL(sgl_lim_native, Meta, module)
{
    module.impl("manage_out", TORCH_FN(SglLimHost::ManageMeta));
}
