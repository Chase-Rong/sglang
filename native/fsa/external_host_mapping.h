#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

namespace sglang::npu_kernel {
void RegisterMemFabricRegion(const at::Tensor &source, int64_t device_address, int64_t device_id);
std::tuple<at::Tensor, int64_t> CreateMemFabricRegion(int64_t address, at::IntArrayRef shape,
    const at::Tensor &dtype_template, int64_t device_id, bool shared_gva);
void UnregisterMemFabricRegion(const at::Tensor &source, int64_t device_id);
bool ValidateMemFabricRegion(const void *host, uint64_t device_address, uint64_t bytes, int64_t device_id);
std::vector<uint64_t> MemFabricDmaSegmentEnds(const void *host, uint64_t device_address,
                                           uint64_t bytes, int64_t device_id);
}
