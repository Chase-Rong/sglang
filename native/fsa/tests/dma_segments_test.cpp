#include "dma_segments.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>

using sglang::npu_kernel::SplitDmaRun;
int main()
{
    using Runs = std::vector<std::pair<uint64_t, uint64_t>>;
    constexpr uint64_t MiB = 1ULL << 20;
    // Component starts 4 MiB before a mapping boundary, not at a slice base.
    const std::vector<uint64_t> ends{4 * MiB, 36 * MiB, 40 * MiB};
    assert((SplitDmaRun(128, 128, ends) == Runs{{128, 128}}));
    assert((SplitDmaRun(128, 8 * MiB, ends) ==
            Runs{{128, 4 * MiB - 128}, {4 * MiB, 4 * MiB + 128}}));
    assert((SplitDmaRun(1, 4 * MiB - 1, ends) == Runs{{1, 4 * MiB - 1}}));
    assert((SplitDmaRun(4 * MiB, 32 * MiB, ends) == Runs{{4 * MiB, 32 * MiB}}));
    assert((SplitDmaRun(1, 40 * MiB - 1, ends) ==
            Runs{{1, 4 * MiB - 1}, {4 * MiB, 32 * MiB}, {36 * MiB, 4 * MiB}}));
    const uint64_t big = 1ULL << 40;
    assert((SplitDmaRun(big - 1, 2, {big, big + 1}) == Runs{{big - 1, 1}, {big, 1}}));
    for (const auto &[offset, size] : Runs{{40 * MiB, 1}, {1, UINT64_MAX}, {0, 0}}) {
        bool rejected = false;
        try { SplitDmaRun(offset, size, ends); }
        catch (const std::out_of_range &) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { SplitDmaRun(0, 1, {}); }
    catch (const std::out_of_range &) { rejected = true; }
    assert(rejected);
    std::mt19937_64 random(34);
    for (int trial = 0; trial < 10000; ++trial) {
        std::vector<uint64_t> boundaries;
        uint64_t total = 0;
        for (int i = 0; i < 8; ++i) {
            total += 1 + random() % (1ULL << 34);
            boundaries.push_back(total);
        }
        const auto start = random() % total;
        const auto count = 1 + random() % (total - start);
        auto cursor = start;
        for (const auto &[offset, size] : SplitDmaRun(start, count, boundaries)) {
            assert(offset == cursor && size > 0);
            const auto end = std::upper_bound(boundaries.begin(), boundaries.end(), offset);
            assert(end != boundaries.end() && size <= *end - offset);
            cursor += size;
        }
        assert(cursor == start + count);
    }
    std::cout << "PASS DMA segment boundary and overflow checks\n";
}
