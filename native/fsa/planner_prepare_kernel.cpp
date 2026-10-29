// Independent external_plan physical-key adapter. PrefixCounts, empty-first
// protected victim compaction, and hash scan seed are derived from the local
// SGLang LIM implementation (lim_postprocess.h / lim_schedule.h). No LIM state,
// includes, planner entry, or binary is shared. Physical alias lookup replaces
// LIM's logical-source map and float sort, which cannot represent int64 keys.
#include "kernel_operator.h"
#if __CCE_AICORE__ == 310
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
using namespace AscendC;
namespace ExternalPlan {
__simt_callee__ inline uint32_t Prefix(__ubuf__ uint32_t *scratch, uint32_t count) {
    uint32_t lane = threadIdx.x;
    scratch[lane] = count;
    asc_syncthreads();
    for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        uint32_t previous = lane >= offset ? scratch[lane - offset] : 0;
        asc_syncthreads();
        scratch[lane] += previous;
        asc_syncthreads();
    }
    return lane == 0 ? 0 : scratch[lane - 1];
}
__simt_callee__ inline uint32_t Hash(uint32_t key) {
    key ^= key >> 16U; key *= 0x7feb352dU;
    key ^= key >> 15U; key *= 0x846ca68bU;
    return key ^ (key >> 16U);
}
__simt_callee__ inline uint32_t KeyHash(int64_t key) {
    return Hash(static_cast<uint32_t>(key) ^ Hash(static_cast<uint64_t>(key) >> 32U));
}
__simt_callee__ inline int32_t Decode(int32_t entry) { return entry < 0 ? -entry - 1 : entry; }

// Exactly one VF block owns an entire request. All dependency boundaries are
// block barriers; there is no cross-core wait or publication-spin protocol.
__simt_vf__ __launch_bounds__(256) void Prepare(
    __gm__ const int64_t *keys, __gm__ const int32_t *ends,
    __gm__ const int64_t *owners, __gm__ const bool *stable,
    __gm__ const int32_t *logical, __gm__ const int32_t *lengths,
    __gm__ const int64_t *identity, __gm__ const bool *allocated,
    __gm__ const bool *payload, __gm__ bool *reusable,
    __gm__ const int32_t *inflight, __gm__ const bool *blocked,
    __gm__ const int8_t *release, __gm__ int32_t *error,
    __gm__ int64_t *rowSlots, __gm__ bool *rowMiss, __gm__ bool *rowWriter,
    __gm__ bool *rowCache, __gm__ int64_t *rowOwners, __gm__ int32_t *rowRequests,
    __gm__ int32_t *rowUnion, __gm__ int64_t *unionKeys, __gm__ int32_t *unionSlots,
    __gm__ int32_t *unionOwners, __gm__ bool *unionMiss, __gm__ bool *unionCache,
    __gm__ int32_t *counts, __gm__ bool *overflow, __gm__ int64_t *victims,
    __gm__ int32_t *ready, __gm__ int32_t *status, __gm__ int64_t *completion,
    __gm__ int64_t *counters, __gm__ int32_t *safe, __gm__ int16_t *membership,
    __gm__ int32_t *selectionTable, __gm__ int32_t *selectionStatus,
    __ubuf__ int32_t *hash, __ubuf__ uint32_t *protect,
    __ubuf__ uint32_t *scratch,
    uint32_t b, uint32_t B, uint32_t T, uint32_t K, uint32_t W,
    uint32_t C, uint32_t Q, uint32_t R, uint32_t S, int64_t hostTokens) {
    uint32_t lane = threadIdx.x, H = 2 * W;
    bool partition = true;
    int32_t previous = 0;
    for (uint32_t j = 0; j < B; ++j) {
        int32_t next = ends[j];
        if (next < previous || next < 0 || next > T) partition = false;
        previous = next;
    }
    // The Python validation permits only explicitly invalid orphan padding.
    uint32_t orphanBad = 0;
    if (partition) for (uint32_t pos = previous * K + lane; pos < T * K; pos += blockDim.x)
        orphanBad |= keys[pos] != -1;
    Prefix(scratch, orphanBad);
    partition &= scratch[255] == 0;
    int32_t begin = partition ? (b ? ends[b - 1] : 0) : 0;
    int32_t end = partition ? ends[b] : 0;
    int64_t owner = owners[b];
    bool invalid = !partition || end - begin > R || owner < 0 || owner >= Q || lengths[b] < 0;
    if (owner > 0 && end > begin) for (uint32_t j = 0; j < B; ++j) {
        int32_t otherStart = j ? ends[j - 1] : 0;
        if (j != b && owners[j] == owner && ends[j] > otherStart) invalid = true;
    }
    bool active = !invalid && owner > 0 && end > begin;
    bool busy = active && inflight[owner] > 0;
    active &= !busy && (!active || (!blocked[owner] && release[owner] == 0));
    uint32_t total = partition ? (end - begin) * K : 0;
    uint32_t badKey = 0;
    for (uint32_t i = lane; i < total; i += blockDim.x) {
        int64_t key = keys[begin * K + i];
        badKey |= key < -1 || key == INT64_MAX || key >= hostTokens;
        int32_t token = logical[begin * K + i];
        if (key >= 0) badKey |= token < 0 || token >= lengths[b] ||
            token > lengths[b] - (end - begin) + static_cast<int32_t>(i / K);
    }
    Prefix(scratch, badKey);
    invalid |= scratch[255] != 0;
    active &= !invalid;
    if (lane == 0) {
        counts[b] = 0; overflow[b] = invalid;
        status[b] = invalid || busy ? 1 : 0;
        if (invalid || busy) asc_atomic_max(error, invalid ? 2 : 1);
    }
    for (uint32_t i = lane; i < 8; i += blockDim.x) counters[b * 8 + i] = 0;
    for (uint32_t i = lane; i < 9; i += blockDim.x) completion[b * 9 + i] = -1;
    for (uint32_t i = lane; i < H; i += blockDim.x) hash[i] = INT32_MAX;
    for (uint32_t i = lane; i < C; i += blockDim.x) protect[i] = 1;
    for (uint32_t i = lane; i < W; i += blockDim.x) {
        uint32_t u = b * W + i;
        unionKeys[u] = -1; unionSlots[u] = C; unionOwners[u] = -1;
        unionMiss[u] = false; unionCache[u] = false; ready[u] = 0;
    }
    // Only the live plan/header regions are written. In particular no request
    // clears the qseq/kvseq prefix: b=0 owns that prefix even when its range is
    // empty and another request owns row0. No cross-block barrier is required.
    uint32_t clearBegin = partition ? begin : (b == 0 ? 0 : T);
    uint32_t clearEnd = partition ? (b + 1 == B ? T : end) : T;
    for (uint32_t p = lane; p < (clearEnd - clearBegin) * K; p += blockDim.x) {
        uint32_t pos = clearBegin * K + p;
        rowSlots[pos] = -1; rowMiss[pos] = false; rowWriter[pos] = false;
        rowCache[pos] = false; rowUnion[pos] = -1; safe[pos] = -1;
        membership[(pos / K) * 16400 + 16384 - K + pos % K] = 0;
    }
    for (uint32_t p = lane; p < (clearEnd - clearBegin) * (C / 128); p += blockDim.x)
        selectionTable[clearBegin * (C / 128) + p] = 0;
    for (uint32_t p = lane; p < (clearEnd - clearBegin) * S; p += blockDim.x)
        selectionStatus[clearBegin * S + p] = 0;
    asc_threadfence(); asc_syncthreads();
    // CAS publishes an immutable occurrence index, whose key was present
    // before launch. A later atomicMin only changes it to the same key's first
    // occurrence, so comparison never waits for a separate key publication.
    for (uint32_t i = lane; active && i < total; i += blockDim.x) {
        uint32_t pos = begin * K + i;
        int64_t key = keys[pos];
        if (key < 0) continue;
        uint32_t bucket = KeyHash(key) & (H - 1);
        for (uint32_t probe = 0; probe < H; ++probe) {
            int32_t found = asc_atomic_cas(hash + bucket, INT32_MAX, static_cast<int32_t>(i));
            if (found == INT32_MAX || keys[begin * K + found] == key) {
                asc_atomic_min(hash + bucket, static_cast<int32_t>(i));
                rowUnion[pos] = bucket;
                break;
            }
            bucket = (bucket + 1) & (H - 1);
        }
    }
    asc_threadfence(); asc_syncthreads();
    uint32_t span = (total + 255) / 256, lo = lane * span;
    uint32_t hi = lo + span < total ? lo + span : total;
    uint32_t unique = 0;
    for (uint32_t i = lo; active && i < hi; ++i) {
        int32_t bucket = rowUnion[begin * K + i];
        unique += bucket >= 0 && hash[bucket] == i;
    }
    uint32_t prefix = Prefix(scratch, unique), count = scratch[255];
    // Use unionOwners to retain first occurrence before hash entries change.
    for (uint32_t i = lo; active && i < hi; ++i) {
        int32_t bucket = rowUnion[begin * K + i];
        if (bucket >= 0 && hash[bucket] == i) {
            uint32_t u = b * W + prefix++;
            unionKeys[u] = keys[begin * K + i]; unionOwners[u] = begin * K + i;
        }
    }
    asc_threadfence(); asc_syncthreads();
    for (uint32_t u = lane; u < count; u += blockDim.x)
        hash[rowUnion[unionOwners[b * W + u]]] = u;
    asc_syncthreads();
    for (uint32_t i = lane; active && i < total; i += blockDim.x) {
        uint32_t pos = begin * K + i;
        int32_t bucket = rowUnion[pos];
        if (bucket >= 0 && !stable[pos]) {
            int32_t u = Decode(hash[bucket]);
            asc_atomic_min(hash + bucket, -u - 1);
        }
    }
    asc_syncthreads();
    for (uint32_t i = lane; active && i < total; i += blockDim.x) {
        uint32_t pos = begin * K + i;
        int32_t bucket = rowUnion[pos];
        if (bucket >= 0) {
            int32_t entry = hash[bucket], u = Decode(entry);
            rowUnion[pos] = u;
            if (unionOwners[b * W + u] == pos) unionCache[b * W + u] = entry >= 0;
        }
    }
    asc_threadfence(); asc_syncthreads();
    // Scan the resident once. All physical aliases of a mutable selected key
    // lose reusable eligibility before hit classification. Payload stays valid.
    for (uint32_t s = lane; active && s < C; s += blockDim.x) {
        uint32_t ix = owner * C + s;
        int64_t key = identity[ix];
        if (key < 0) continue;
        uint32_t bucket = KeyHash(key) & (H - 1);
        for (uint32_t probe = 0; probe < H && hash[bucket] != INT32_MAX; ++probe) {
            int32_t entry = hash[bucket], u = Decode(entry);
            if (u < count && unionKeys[b * W + u] == key) {
                if (!unionCache[b * W + u]) reusable[ix] = false;
                if (allocated[ix] && payload[ix] && reusable[ix] && unionCache[b * W + u])
                    asc_atomic_min(unionSlots + b * W + u, static_cast<int32_t>(s));
                break;
            }
            bucket = (bucket + 1) & (H - 1);
        }
    }
    asc_threadfence(); asc_syncthreads();
    uint32_t unionSpan = (count + 255) / 256;
    lo = lane * unionSpan; hi = lo + unionSpan < count ? lo + unionSpan : count;
    uint32_t missing = 0;
    for (uint32_t u = lo; u < hi; ++u) {
        uint32_t ix = b * W + u;
        bool miss = unionSlots[ix] == C;
        unionMiss[ix] = miss;
        missing += miss;
        if (!miss) asc_atomic_min(protect + unionSlots[ix], 0U);
    }
    prefix = Prefix(scratch, missing);
    uint32_t missCount = scratch[255];
    for (uint32_t u = lo; u < hi; ++u)
        if (unionMiss[b * W + u]) ready[b * W + u] = ++prefix;
    asc_threadfence(); asc_syncthreads();
    // Derived from LIM SelectVictims: compact unallocated slots in ascending
    // order, then unprotected allocated slots from a hashed 2048-slot chunk.
    uint32_t slotSpan = (C + 255) / 256;
    lo = lane * slotSpan; hi = lo + slotSpan < C ? lo + slotSpan : C;
    uint32_t empty = 0;
    for (uint32_t s = lo; active && s < hi; ++s) empty += protect[s] && !allocated[owner * C + s];
    prefix = Prefix(scratch, empty);
    uint32_t totalEmpty = scratch[255];
    for (uint32_t s = lo; active && s < hi; ++s)
        if (protect[s] && !allocated[owner * C + s]) {
            if (prefix < missCount) victims[b * C + prefix] = s;
            ++prefix;
        }
    asc_syncthreads();
    uint32_t first = Hash(static_cast<uint32_t>(lengths[b])) % ((C + 2047) / 2048) * 2048;
    uint32_t eligible = 0;
    for (uint32_t i = lo; active && i < hi; ++i) {
        uint32_t s = (first + i) % C;
        eligible += protect[s] && allocated[owner * C + s];
    }
    prefix = Prefix(scratch, eligible) + totalEmpty;
    for (uint32_t i = lo; active && i < hi; ++i) {
        uint32_t s = (first + i) % C;
        if (protect[s] && allocated[owner * C + s]) {
            if (prefix < missCount) victims[b * C + prefix] = s;
            ++prefix;
        }
    }
    asc_threadfence(); asc_syncthreads();
    for (uint32_t u = lane; u < W; u += blockDim.x) {
        uint32_t ix = b * W + u;
        if (u < count) {
            if (unionMiss[ix]) unionSlots[ix] = victims[b * C + ready[ix] - 1];
            ready[ix] = unionMiss[ix] ? 1 : 2;
        } else unionSlots[ix] = -1;
    }
    asc_threadfence(); asc_syncthreads();
    // Reuse dead protected scratch for row-nonempty flags; no int64 atomic
    // writes to row owners or cross-request scalar cacheline sharing.
    for (uint32_t row = lane; active && row < end - begin; row += blockDim.x)
        protect[row] = 0;
    asc_syncthreads();
    for (uint32_t i = lane; active && i < total; i += blockDim.x) {
        uint32_t pos = begin * K + i;
        int32_t u = rowUnion[pos];
        if (u < 0) continue;
        uint32_t ix = b * W + u;
        int32_t slot = unionSlots[ix];
        rowSlots[pos] = slot; rowMiss[pos] = unionMiss[ix]; rowCache[pos] = unionCache[ix];
        rowWriter[pos] = unionMiss[ix] && unionOwners[ix] == pos;
        safe[pos] = logical[pos];
        asc_atomic_max(protect + pos / K - clearBegin, 1U);
        membership[(pos / K) * 16400 + 16384 - K + pos % K] =
            unionMiss[ix] ? -slot - 1 : slot + 1;
    }
    asc_syncthreads();
    for (uint32_t row = clearBegin + lane; row < clearEnd; row += blockDim.x) {
        bool rowActive = active && row < end;
        rowOwners[row] = rowActive && protect[row - clearBegin] ? owner : 0;
        rowRequests[row] = rowActive ? b : -1;
        uint32_t h = row * 16400 + 16384;
        membership[h] = 0x5A4D; membership[h + 1] = 0x5A45;
        membership[h + 2] = K; membership[h + 3] = 16384 - K;
        membership[h + 4] = rowActive ? owner : 0; membership[h + 5] = 0x5A53;
        membership[h + 6] = C; membership[h + 7] = 3; membership[h + 8] = W;
        for (uint32_t j = 9; j < 16; ++j) membership[h + j] = 0;
    }
    asc_threadfence(); asc_syncthreads();
    if (b == 0) for (uint32_t j = lane; j < B; j += blockDim.x) {
        auto seq = reinterpret_cast<__gm__ int32_t *>(membership);
        seq[j] = partition ? ends[j] : T;
        seq[B + j] = partition ? lengths[j] : 0;
    }
    if (lane == 0) counts[b] = count;
    asc_threadfence();
}
}

#endif

extern "C" __global__ __aicore__ void external_plan_prepare(
    GM_ADDR keys, GM_ADDR ends, GM_ADDR owners, GM_ADDR stable, GM_ADDR logical, GM_ADDR lengths,
    GM_ADDR identity, GM_ADDR allocated, GM_ADDR payload, GM_ADDR reusable,
    GM_ADDR inflight, GM_ADDR blocked, GM_ADDR release, GM_ADDR error,
    GM_ADDR rowSlots, GM_ADDR rowMiss, GM_ADDR rowWriter, GM_ADDR rowCache,
    GM_ADDR rowOwners, GM_ADDR rowRequests, GM_ADDR rowUnion,
    GM_ADDR unionKeys, GM_ADDR unionSlots, GM_ADDR unionOwners, GM_ADDR unionMiss, GM_ADDR unionCache,
    GM_ADDR counts, GM_ADDR overflow, GM_ADDR victims, GM_ADDR ready, GM_ADDR status,
    GM_ADDR completion, GM_ADDR counters, GM_ADDR safe, GM_ADDR membership,
    GM_ADDR selectionTable, GM_ADDR selectionStatus,
    uint32_t B, uint32_t T, uint32_t K, uint32_t W, uint32_t C, uint32_t Q,
    uint32_t R, uint32_t S, int64_t hostTokens) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if __CCE_AICORE__ == 310
    if ASCEND_IS_AIV {
    TPipe pipe;
    TBuf<TPosition::VECCALC> hashBuffer, protectedBuffer, scratchBuffer;
    pipe.InitBuffer(hashBuffer, 2 * W * 4);
    pipe.InitBuffer(protectedBuffer, C * 4);
    pipe.InitBuffer(scratchBuffer, 256 * 4);
    auto hash = hashBuffer.Get<int32_t>();
    auto protect = protectedBuffer.Get<uint32_t>();
    auto scratch = scratchBuffer.Get<uint32_t>();
    for (uint32_t b = GetBlockIdx(); b < B; b += GetBlockNum() * 2) {
        SetFlag<HardEvent::S_V>(EVENT_ID7); WaitFlag<HardEvent::S_V>(EVENT_ID7);
        asc_vf_call<ExternalPlan::Prepare>(dim3(256),
            (__gm__ const int64_t *)keys, (__gm__ const int32_t *)ends,
            (__gm__ const int64_t *)owners, (__gm__ const bool *)stable,
            (__gm__ const int32_t *)logical, (__gm__ const int32_t *)lengths,
            (__gm__ const int64_t *)identity, (__gm__ const bool *)allocated,
            (__gm__ const bool *)payload, (__gm__ bool *)reusable,
            (__gm__ const int32_t *)inflight, (__gm__ const bool *)blocked,
            (__gm__ const int8_t *)release, (__gm__ int32_t *)error,
            (__gm__ int64_t *)rowSlots, (__gm__ bool *)rowMiss, (__gm__ bool *)rowWriter,
            (__gm__ bool *)rowCache, (__gm__ int64_t *)rowOwners, (__gm__ int32_t *)rowRequests,
            (__gm__ int32_t *)rowUnion, (__gm__ int64_t *)unionKeys, (__gm__ int32_t *)unionSlots,
            (__gm__ int32_t *)unionOwners, (__gm__ bool *)unionMiss, (__gm__ bool *)unionCache,
            (__gm__ int32_t *)counts, (__gm__ bool *)overflow, (__gm__ int64_t *)victims,
            (__gm__ int32_t *)ready, (__gm__ int32_t *)status, (__gm__ int64_t *)completion,
            (__gm__ int64_t *)counters, (__gm__ int32_t *)safe, (__gm__ int16_t *)membership,
            (__gm__ int32_t *)selectionTable, (__gm__ int32_t *)selectionStatus,
            (__ubuf__ int32_t *)hash.GetPhyAddr(), (__ubuf__ uint32_t *)protect.GetPhyAddr(),
            (__ubuf__ uint32_t *)scratch.GetPhyAddr(), b, B, T, K, W, C, Q, R, S, hostTokens);
        SetFlag<HardEvent::V_S>(EVENT_ID7); WaitFlag<HardEvent::V_S>(EVENT_ID7);
    }
    }
#endif
}
