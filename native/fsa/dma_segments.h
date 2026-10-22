#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sglang::npu_kernel {
// Ends are increasing exclusive offsets, queried once from the registered
// allocation's actual ACL mappings. Virtual contiguity alone is insufficient.
inline std::vector<std::pair<uint64_t, uint64_t>> SplitDmaRun(
    uint64_t offset, uint64_t bytes, const std::vector<uint64_t> &ends)
{
    if (ends.empty() || bytes == 0 || offset >= ends.back() || bytes > ends.back() - offset) {
        throw std::out_of_range("DMA run is outside registered mapping segments");
    }
    std::vector<std::pair<uint64_t, uint64_t>> result;
    auto end = std::upper_bound(ends.begin(), ends.end(), offset);
    while (bytes != 0) {
        const auto count = std::min(bytes, *end - offset);
        result.emplace_back(offset, count);
        offset += count;
        bytes -= count;
        ++end;
    }
    return result;
}
}
