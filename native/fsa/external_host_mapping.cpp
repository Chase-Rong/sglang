#include "external_host_mapping.h"

#include <acl/acl.h>
#include <algorithm>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace sglang::npu_kernel {
namespace {
struct Mapping {
    uint64_t device_address;
    uint64_t bytes;
    std::vector<uint64_t> dma_segment_ends;
};
std::mutex mapping_mutex;
std::map<std::pair<uint64_t, int64_t>, Mapping> mappings;

std::vector<uint64_t> QueryDmaSegmentEnds(uint64_t device, uint64_t bytes)
{
    std::vector<uint64_t> ends;
    uint64_t offset = 0;
    while (offset < bytes) {
        void *base = nullptr;
        size_t size = 0;
        const auto address = device + offset;
        const auto result = aclrtMemGetAddressRange(reinterpret_cast<void *>(address), &base, &size);
        const auto start = reinterpret_cast<uint64_t>(base);
        TORCH_CHECK(result == ACL_SUCCESS && start <= address && address - start < size,
                    "Cannot query MemFabric DMA mapping range: ", result);
        offset += std::min<uint64_t>(bytes - offset, size - (address - start));
        ends.push_back(offset);
    }
    return ends;
}

uint64_t QueryDeviceAddress(uint64_t address, bool shared_gva)
{
    uint64_t device = 0;
    int32_t result;
    if (shared_gva) {
        using Query = int32_t (*)(uint64_t, uint32_t, uint64_t *);
        auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "hybm_gva_to_va"));
        TORCH_CHECK(query != nullptr, "MemFabric shared mapping query is not loaded");
        result = query(address, 1U, &device);
    } else {
        using Query = int32_t (*)(uint64_t, uint64_t *);
        auto query = reinterpret_cast<Query>(dlsym(RTLD_DEFAULT, "hybm_hva_to_dva"));
        TORCH_CHECK(query != nullptr, "MemFabric mapping query is not loaded");
        result = query(address, &device);
    }
    TORCH_CHECK(result == 0 && device != 0, "MemFabric address has no device mapping: ", result);
    return device;
}
}

std::tuple<at::Tensor, int64_t> CreateMemFabricRegion(int64_t address, at::IntArrayRef shape,
    const at::Tensor &dtype_template, int64_t device_id, bool shared_gva)
{
    TORCH_CHECK(address > 0 && !shape.empty() && dtype_template.device().is_cpu(),
                "MemFabric metadata requires an address, shape and CPU dtype template");
    TORCH_CHECK(device_id == c10_npu::getCurrentNPUStream().device_index(),
                "MemFabric metadata must match the current NPU device");
    uint64_t bytes = dtype_template.element_size();
    for (auto dimension : shape) {
        TORCH_CHECK(dimension > 0 && bytes <= INT64_MAX / dimension, "MemFabric shape overflow");
        bytes *= dimension;
    }
    TORCH_CHECK(static_cast<uint64_t>(address) <= UINT64_MAX - (bytes - 1), "MemFabric address overflow");
    const auto device = QueryDeviceAddress(address, shared_gva);
    TORCH_CHECK(device <= INT64_MAX - (bytes - 1) &&
                QueryDeviceAddress(address + bytes - 1, shared_gva) == device + bytes - 1,
                "MemFabric allocation is not a contiguous device mapping");
    auto metadata = at::from_blob(reinterpret_cast<void *>(address), shape, dtype_template.options());
    std::lock_guard<std::mutex> guard(mapping_mutex);
    TORCH_CHECK(mappings.emplace(std::make_pair(address, device_id), Mapping{device, bytes, {}}).second,
                "MemFabric allocation already registered");
    return {metadata, static_cast<int64_t>(device)};
}

void RegisterMemFabricRegion(const at::Tensor &source, int64_t device_address, int64_t device_id)
{
    TORCH_CHECK(source.device().is_cpu() && source.is_contiguous() && source.storage_offset() == 0,
                "MemFabric registration requires base-aligned contiguous CPU metadata");
    TORCH_CHECK(device_address > 0 && device_id >= 0 && source.nbytes() > 0,
                "MemFabric registration requires a nonempty allocation and valid device");
    TORCH_CHECK(c10_npu::getCurrentNPUStream().device_index() == device_id,
                "MemFabric registration must match the current NPU device");
    using QueryMapping = int32_t (*)(uint64_t, uint64_t *);
    auto query = reinterpret_cast<QueryMapping>(dlsym(RTLD_DEFAULT, "hybm_hva_to_dva"));
    TORCH_CHECK(query != nullptr, "MemFabric mapping query is not loaded");
    const auto host = reinterpret_cast<uint64_t>(source.data_ptr());
    const auto bytes = static_cast<uint64_t>(source.nbytes());
    const auto device = static_cast<uint64_t>(device_address);
    TORCH_CHECK(host <= std::numeric_limits<uint64_t>::max() - (bytes - 1) &&
                device <= std::numeric_limits<uint64_t>::max() - (bytes - 1), "MemFabric range overflow");
    uint64_t first = 0;
    uint64_t last = 0;
    TORCH_CHECK(query(host, &first) == 0 && query(host + bytes - 1, &last) == 0 &&
                first == device && last == device + bytes - 1, "MemFabric range is not mapped to the supplied DVA");
    std::lock_guard<std::mutex> guard(mapping_mutex);
    TORCH_CHECK(mappings.emplace(std::make_pair(host, device_id), Mapping{device, bytes, {}}).second,
                "MemFabric allocation already registered");
}

void UnregisterMemFabricRegion(const at::Tensor &source, int64_t device_id)
{
    std::lock_guard<std::mutex> guard(mapping_mutex);
    const auto key = std::make_pair(reinterpret_cast<uint64_t>(source.data_ptr()), device_id);
    auto entry = mappings.find(key);
    TORCH_CHECK(entry != mappings.end() && entry->second.bytes == source.nbytes(),
                "MemFabric unregister does not match a live allocation");
    mappings.erase(entry);
}

bool ValidateMemFabricRegion(const void *host, uint64_t device_address, uint64_t bytes, int64_t device_id)
{
    std::lock_guard<std::mutex> guard(mapping_mutex);
    auto entry = mappings.find(std::make_pair(reinterpret_cast<uint64_t>(host), device_id));
    return entry != mappings.end() && bytes > 0 && bytes <= entry->second.bytes &&
        device_address == entry->second.device_address;
}

std::vector<uint64_t> MemFabricDmaSegmentEnds(const void *host, uint64_t device_address,
                                           uint64_t bytes, int64_t device_id)
{
    std::lock_guard<std::mutex> guard(mapping_mutex);
    auto entry = mappings.find(std::make_pair(reinterpret_cast<uint64_t>(host), device_id));
    TORCH_CHECK(entry != mappings.end() && bytes > 0 && bytes <= entry->second.bytes &&
                device_address == entry->second.device_address,
                "DMA requires a live MemFabric allocation");
    // Lazy, per-registration cache: no runtime queries on later chunks or graph
    // replay, and unregister removes boundaries along with the mapping.
    auto &mapping = entry->second;
    if (mapping.dma_segment_ends.empty()) {
        mapping.dma_segment_ends = QueryDmaSegmentEnds(mapping.device_address, mapping.bytes);
    }
    return mapping.dma_segment_ends;
}
}
