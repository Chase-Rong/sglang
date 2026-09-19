#include "host_plan.h"
#include "host_callback.h"

#include <algorithm>

namespace sglang::npu_kernel {
namespace {
struct HostPlan {
    at::Tensor keys, slots, owners, lengths, cacheable, applied, state, membership, misses;
    int64_t rows;
    bool commit;
    bool captured = false;

    void Run() noexcept
    {
        const auto width = keys.size(1);
        const auto capacity = applied.size(1);
        const auto *key_data = keys.const_data_ptr<int64_t>();
        const auto *slot_data = slots.const_data_ptr<int64_t>();
        const auto *owner_data = owners.const_data_ptr<int64_t>();
        const auto *length_data = lengths.const_data_ptr<int64_t>();
        const auto *stable = cacheable.const_data_ptr<bool>();
        auto *applied_data = applied.data_ptr<int64_t>();
        auto *state_data = state.data_ptr<int64_t>();
        auto *encoded = membership.data_ptr<int16_t>();
        auto *miss_data = misses.data_ptr<bool>();
        #pragma omp parallel for if(rows > 16) num_threads(4) schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            auto *resident = applied_data + row * capacity;
            auto *row_state = state_data + row * 2;
            if (!commit && (row_state[0] != owner_data[row] || length_data[row] < row_state[1])) {
                std::fill(resident, resident + capacity, -1);
            }
            auto *plan = encoded + row * membership.size(1) + 16384 - width;
            for (int64_t position = 0; position < width; ++position) {
                const auto offset = row * width + position;
                const auto slot = slot_data[offset];
                const auto key = key_data[offset];
                const bool valid = owner_data[row] >= 0 && length_data[row] > 0 &&
                                   key >= 0 && slot >= 0 && slot < capacity;
                if (commit) {
                    if (valid) {
                        resident[slot] = stable[offset] ? key : -1;
                    }
                } else {
                    const bool hit = valid && stable[offset] && resident[slot] == key;
                    plan[position] = valid ? (hit ? slot + 1 : -(slot + 1)) : 0;
                    miss_data[offset] = valid && !hit;
                }
            }
            if (commit) {
                row_state[0] = owner_data[row];
                row_state[1] = length_data[row];
            } else {
                auto *control = encoded + row * membership.size(1) + 16384;
                const int16_t values[] = {0x5A4D, 0x5A45, static_cast<int16_t>(width),
                    static_cast<int16_t>(16384 - width), static_cast<int16_t>(row),
                    0x5A44, static_cast<int16_t>(capacity), 0};
                std::copy(std::begin(values), std::end(values), control);
            }
        }
    }
};
}

void MaterializeHostPlan(const at::Tensor &trigger, const at::Tensor &keys,
                         const at::Tensor &slots, const at::Tensor &owners,
                         const at::Tensor &lengths, const at::Tensor &cacheable,
                         at::Tensor applied, at::Tensor state, at::Tensor membership,
                         at::Tensor misses, int64_t rows, bool commit)
{
    for (const auto &tensor : {keys, slots, owners, lengths, applied, state}) {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() && tensor.scalar_type() == at::kLong,
                    "Host plan state must be contiguous CPU int64");
    }
    TORCH_CHECK(keys.dim() == 2 && rows > 0 && rows <= keys.size(0) && rows <= 32767 &&
                keys.size(1) > 0 && keys.size(1) <= 2051 && slots.sizes() == keys.sizes(),
                "Host plan key/slot geometry mismatch");
    TORCH_CHECK(applied.dim() == 2 && applied.size(0) == keys.size(0) &&
                applied.size(1) >= keys.size(1) && applied.size(1) <= 32767 &&
                state.dim() == 2 && state.size(0) == keys.size(0) && state.size(1) == 2 &&
                owners.numel() == keys.size(0) && lengths.numel() == keys.size(0),
                "Host plan layer state geometry mismatch");
    for (const auto &tensor : {cacheable, misses}) {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() && tensor.scalar_type() == at::kBool &&
                    tensor.sizes() == keys.sizes(), "Host plan masks must match keys");
    }
    TORCH_CHECK(membership.device().is_cpu() && membership.is_contiguous() &&
                membership.scalar_type() == at::kShort && membership.dim() == 2 &&
                membership.size(0) == keys.size(0) && membership.size(1) >= 16392,
                "Host plan membership ABI mismatch");
    HostCallback<HostPlan>::Enqueue(trigger, std::make_unique<HostPlan>(HostPlan{
        keys, slots, owners, lengths, cacheable, applied, state, membership, misses, rows, commit}));
}
}
