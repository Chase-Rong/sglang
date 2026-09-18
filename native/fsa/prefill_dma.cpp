#include "prefill_dma.h"
#include "external_host_mapping.h"
#include "dma_segments.h"

#include <acl/acl.h>
#include <utility>
#include <vector>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace sglang::npu_kernel {
void PrefillDmaCopy(const at::Tensor &device_tensor, const at::Tensor &host_metadata,
                   int64_t device_address, const at::Tensor &runs, bool to_host)
{
    const auto stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(device_tensor.device().type() == at::DeviceType::PrivateUse1 &&
                device_tensor.device().index() == stream.device_index(), "DMA stream device mismatch");
    TORCH_CHECK(device_tensor.is_contiguous() && host_metadata.device().is_cpu() &&
                host_metadata.is_contiguous() && device_tensor.dim() == 2 && host_metadata.dim() == 2 &&
                device_tensor.scalar_type() == host_metadata.scalar_type() &&
                device_tensor.size(1) == host_metadata.size(1), "DMA row layout mismatch");
    const auto segment_ends = MemFabricDmaSegmentEnds(host_metadata.data_ptr(), device_address,
                                                     host_metadata.nbytes(), stream.device_index());
    TORCH_CHECK(runs.device().is_cpu() && runs.is_contiguous() && runs.scalar_type() == at::kLong &&
                runs.dim() == 2 && runs.size(1) == 2, "DMA runs must be contiguous CPU int64 [N,2]");
    const auto *entries = runs.const_data_ptr<int64_t>();
    const uint64_t row_bytes = device_tensor.size(1) * device_tensor.element_size();
    std::vector<std::pair<uint64_t, uint64_t>> copies;
    copies.reserve(runs.size(0));
    for (int64_t index = 0; index < runs.size(0); ++index) {
        const auto start = entries[index * 2];
        const auto count = entries[index * 2 + 1];
        TORCH_CHECK(start > 0 && count > 0 && count <= device_tensor.size(0) &&
                    count <= host_metadata.size(0) && start <= device_tensor.size(0) - count &&
                    start <= host_metadata.size(0) - count, "DMA run is outside the KV allocation");
        const auto segments = SplitDmaRun(static_cast<uint64_t>(start) * row_bytes,
                                         static_cast<uint64_t>(count) * row_bytes, segment_ends);
        copies.insert(copies.end(), segments.begin(), segments.end());
    }
    c10_npu::NPUCachingAllocator::recordStream(device_tensor.storage().data_ptr(), stream);
    const auto raw_stream = stream.stream(false);
    auto callback = [device_tensor, host_metadata, device_address, copies = std::move(copies), to_host, raw_stream]() {
        const auto device_base = reinterpret_cast<uint64_t>(device_tensor.data_ptr());
        for (const auto &[offset, bytes] : copies) {
            const auto source = (to_host ? device_base : device_address) + offset;
            const auto destination = (to_host ? device_address : device_base) + offset;
            const auto result = aclrtMemcpyAsync(reinterpret_cast<void *>(destination), bytes,
                reinterpret_cast<const void *>(source), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, raw_stream);
            TORCH_CHECK(result == ACL_SUCCESS, "Prefill DMA submission failed: ", result);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("fsa_prefill_dma", callback);
}
}
