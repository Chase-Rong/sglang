#include "prefill_dma.h"
#include "external_host_mapping.h"
#include "dma_segments.h"

#include <acl/acl.h>
#include <algorithm>
#include <array>
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
                device_tensor.size(1) == host_metadata.size(1) &&
                device_tensor.size(1) > 0, "DMA row layout mismatch");
    TORCH_CHECK(runs.device().is_cpu() && runs.is_contiguous() && runs.scalar_type() == at::kLong &&
                runs.dim() == 2 && (runs.size(1) == 2 || runs.size(1) == 3),
                "DMA runs must be contiguous CPU int64 [N,2] or [N,3] (source, destination, count)");
    if (runs.size(0) == 0) {
        return;
    }
    const auto *entries = runs.const_data_ptr<int64_t>();
    const auto columns = runs.size(1);
    const auto source_capacity = to_host ? device_tensor.size(0) : host_metadata.size(0);
    const auto destination_capacity = to_host ? host_metadata.size(0) : device_tensor.size(0);
    std::vector<std::array<int64_t, 3>> row_runs;
    std::vector<std::pair<int64_t, int64_t>> destinations;
    row_runs.reserve(runs.size(0));
    destinations.reserve(runs.size(0));
    for (int64_t index = 0; index < runs.size(0); ++index) {
        const auto source = entries[index * columns];
        const auto destination = columns == 3 ? entries[index * columns + 1] : source;
        const auto count = entries[index * columns + columns - 1];
        const auto minimum_row = columns == 2 ? 1 : 0;
        TORCH_CHECK(source >= minimum_row && destination >= minimum_row && count > 0 &&
                    count <= source_capacity && count <= destination_capacity &&
                    source <= source_capacity - count && destination <= destination_capacity - count,
                    "DMA run is outside the KV allocation");
        destinations.emplace_back(destination, destination + count);
        if (!row_runs.empty() && row_runs.back()[0] + row_runs.back()[2] == source &&
            row_runs.back()[1] + row_runs.back()[2] == destination) {
            row_runs.back()[2] += count;
        } else {
            row_runs.push_back({source, destination, count});
        }
    }
    std::sort(destinations.begin(), destinations.end());
    for (size_t index = 1; index < destinations.size(); ++index) {
        TORCH_CHECK(destinations[index].first >= destinations[index - 1].second,
                    "DMA runs contain duplicate destination rows");
    }
    const auto segment_ends = MemFabricDmaSegmentEnds(host_metadata.data_ptr(), device_address,
                                                     host_metadata.nbytes(), stream.device_index());
    const uint64_t row_bytes = static_cast<uint64_t>(device_tensor.size(1)) * device_tensor.element_size();
    std::vector<std::array<uint64_t, 3>> copies;
    copies.reserve(row_runs.size());
    for (const auto &[source, destination, count] : row_runs) {
        const auto host_offset = static_cast<uint64_t>(to_host ? destination : source) * row_bytes;
        const auto device_offset = static_cast<uint64_t>(to_host ? source : destination) * row_bytes;
        const auto segments = SplitDmaRun(host_offset, static_cast<uint64_t>(count) * row_bytes,
                                         segment_ends);
        for (const auto &[offset, bytes] : segments) {
            copies.push_back({device_offset + (offset - host_offset), offset, bytes});
        }
    }
    c10_npu::NPUCachingAllocator::recordStream(device_tensor.storage().data_ptr(), stream);
    const auto raw_stream = stream.stream(false);
    auto callback = [device_tensor, host_metadata, device_address, copies = std::move(copies), to_host, raw_stream]() {
        const auto device_base = reinterpret_cast<uint64_t>(device_tensor.data_ptr());
        for (const auto &[device_offset, host_offset, bytes] : copies) {
            const auto device_pointer = device_base + device_offset;
            const auto host_pointer = static_cast<uint64_t>(device_address) + host_offset;
            const auto source = to_host ? device_pointer : host_pointer;
            const auto destination = to_host ? host_pointer : device_pointer;
            const auto result = aclrtMemcpyAsync(reinterpret_cast<void *>(destination), bytes,
                reinterpret_cast<const void *>(source), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, raw_stream);
            TORCH_CHECK(result == ACL_SUCCESS, "Prefill DMA submission failed: ", result);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("fsa_prefill_dma", callback);
}
}
