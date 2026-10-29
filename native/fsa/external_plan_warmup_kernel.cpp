// Standalone external-plan prefix warmup, with one unique payload writer.
#include "kernel_operator.h"
using namespace AscendC;
namespace {
constexpr int64_t MAGIC = 0x485350335631LL;
template <class T> __aicore__ inline GlobalTensor<T> G(GM_ADDR p) {
    GlobalTensor<T> g;
    g.SetGlobalBuffer((__gm__ T *)p);
    return g;
}
__aicore__ inline void Flush(GM_ADDR p, int64_t bytes) {
    auto g = G<int8_t>(p);
    for (int64_t i = 0; i < bytes; i += 64)
        DataCacheCleanAndInvalid<int8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            g[i]);
}
} // namespace
namespace {
// The sole payload writer for FILL_ONLY. One AIV has bounded
// local delay only; it never waits for a consumer or another unscheduled core.
__aicore__ inline void FillUnique(GM_ADDR source, GM_ADDR dest, GM_ADDR requestp, GM_ADDR keyp,
                                  GM_ADDR slotp, GM_ADDR countp, GM_ADDR missp, GM_ADDR preparep,
                                  GM_ADDR readyp, GM_ADDR statusp, GM_ADDR counterp, int64_t B,
                                  int64_t W, int64_t C, int64_t delay, int64_t fail) {
    auto request = G<int64_t>(requestp), keys = G<int64_t>(keyp);
    auto slots = G<int32_t>(slotp), count = G<int32_t>(countp), ready = G<int32_t>(readyp);
    auto status = G<int32_t>(statusp), prepared = G<int32_t>(preparep);
    auto miss = G<bool>(missp);
    auto counter = G<int64_t>(counterp);
    TPipe pipe;
    TBuf<TPosition::VECCALC> copyBuf;
    pipe.InitBuffer(copyBuf, 1024);
    auto token = copyBuf.Get<bfloat16_t>();
    auto src = G<bfloat16_t>(source), dst = G<bfloat16_t>(dest);
    for (volatile int64_t d = 0; d < delay; ++d) {
    } // bounded injection, default zero
    for (int64_t b = 0; b < B; ++b) {
        auto cnt = count.GetValue(b);
        auto req = request.GetValue(b);
        if (cnt < 0 || cnt > W || cnt > C || req <= 0)
            continue;
        if (prepared.GetValue(b) != 0) {
            status.SetValue(b, prepared.GetValue(b));
            for (int64_t u = 0; u < cnt; ++u)
                ready.SetValue(b * W + u, 3);
            continue;
        }
        int64_t loads = 0, writes = 0;
        for (int64_t u = 0; u < cnt; ++u) {
            auto ix = b * W + u;
            if (miss.GetValue(ix)) {
                DataCopy(token, src[keys.GetValue(ix) * 512], 512);
                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);
                ++loads;
                DataCopy(dst[(req * C + slots.GetValue(ix)) * 512], token, 512);
                SetFlag<HardEvent::MTE3_S>(0);
                WaitFlag<HardEvent::MTE3_S>(0);
                ++writes;
                if (b == fail) {
                    // Fault injection is deliberately AFTER one completed real copy.
                    // Keep exact copy counters; no dependent may publish this partial plan.
                    status.SetValue(b, 3);
                    for (int64_t remaining = u; remaining < cnt; ++remaining)
                        ready.SetValue(b * W + remaining, 3);
                    break;
                }
            }
            ready.SetValue(ix, 2);
        }
        if (b == fail && writes == 0) {
            // With no misses there is no copy to fault after: reject with zero writes.
            status.SetValue(b, 3);
            for (int64_t u = 0; u < cnt; ++u)
                ready.SetValue(b * W + u, 3);
        }
        counter.SetValue(b * 2, loads);
        counter.SetValue(b * 2 + 1, writes);
    }
    Flush(readyp, B * W * 4);
    Flush(statusp, B * 4);
    Flush(counterp, B * 2 * 8);
}
} // namespace
extern "C" __global__ __aicore__ void
external_plan_warmup(GM_ADDR source, GM_ADDR dest, GM_ADDR tablep, GM_ADDR lengthp, GM_ADDR requestp,
                 GM_ADDR keyp, GM_ADDR slotp, GM_ADDR countp, GM_ADDR missp, GM_ADDR ownerp,
                 GM_ADDR cachep, GM_ADDR idp, GM_ADDR readyp, GM_ADDR statusp, GM_ADDR completionp,
                 GM_ADDR counterp, int64_t B, int64_t W, int64_t Q, int64_t C, int64_t N,
                 int64_t hostTokens, int64_t delay, int64_t fail) {
    if (GetBlockIdx() != 0)
        return;
    auto table = G<int32_t>(tablep), length = G<int32_t>(lengthp), slots = G<int32_t>(slotp);
    auto count = G<int32_t>(countp), owner = G<int32_t>(ownerp), ready = G<int32_t>(readyp),
         status = G<int32_t>(statusp);
    auto reqs = G<int64_t>(requestp), keys = G<int64_t>(keyp), id = G<int64_t>(idp),
         done = G<int64_t>(completionp), counter = G<int64_t>(counterp);
    auto miss = G<bool>(missp), cache = G<bool>(cachep);
    // End this validation pipe before entering the shared unique-copy pipe.
    {
        TPipe pipe;
        TBuf<TPosition::VECCALC> bitmapBuf;
        pipe.InitBuffer(bitmapBuf, 32768 * 4);
        auto bitmap = bitmapBuf.Get<int32_t>();
        for (int64_t b = 0; b < B; ++b) {
            status.SetValue(b, 0);
            counter.SetValue(b * 2, 0);
            counter.SetValue(b * 2 + 1, 0);
            for (int64_t j = 0; j < 9; ++j)
                done.SetValue(b * 9 + j, -1);
            for (int64_t u = 0; u < W; ++u)
                ready.SetValue(b * W + u, 0);
            auto req = reqs.GetValue(b);
            auto cnt = count.GetValue(b), len = length.GetValue(b);
            if (req == 0) {
                if (cnt != 0 || len != 0)
                    status.SetValue(b, 1);
                continue;
            }
            bool valid = req > 0 && req < Q && cnt >= 0 && cnt <= C && cnt == len && len <= N * 128;
            for (int64_t o = 0; o < B; ++o)
                if (o != b && reqs.GetValue(o) == req)
                    valid = false;
            bool identityOk = id.GetValue(b * 9) == MAGIC && id.GetValue(b * 9 + 1) > 0 &&
                              id.GetValue(b * 9 + 2) >= 0 && id.GetValue(b * 9 + 3) == req &&
                              id.GetValue(b * 9 + 4) >= 0 && id.GetValue(b * 9 + 5) >= 0 &&
                              id.GetValue(b * 9 + 6) > 0 &&
                              id.GetValue(b * 9 + 7) == id.GetValue(b * 9 + 6) &&
                              id.GetValue(b * 9 + 8) >= 0;
            for (int64_t o = 0; o < B; ++o)
                if (reqs.GetValue(o) > 0 && (id.GetValue(o * 9 + 1) != id.GetValue(b * 9 + 1) ||
                                             id.GetValue(o * 9 + 2) != id.GetValue(b * 9 + 2)))
                    identityOk = false;
            if (!valid || !identityOk) {
                status.SetValue(b, identityOk ? 1 : 2);
                continue;
            }
            for (int64_t s = 0; s < C; ++s)
                bitmap.SetValue(s, -1);
            for (int64_t u = 0; u < cnt; ++u) {
                auto ix = b * W + u, key = keys.GetValue(ix);
                auto slot = slots.GetValue(ix);
                if (slot < 0 || slot >= C)
                    valid = false;
                else {
                    if (bitmap.GetValue(slot) != -1)
                        valid = false;
                    bitmap.SetValue(slot, u);
                }
                if (key < 0 || key >= hostTokens || (u && keys.GetValue(ix - 1) >= key) ||
                    !cache.GetValue(ix))
                    valid = false;
            }
            for (int64_t u = 0; u < C; ++u)
                bitmap.SetValue(u, -1);
            for (int64_t u = 0; u < cnt; ++u) {
                auto ix = b * W + u;
                auto logical = owner.GetValue(ix);
                if (logical < 0 || logical >= len) {
                    valid = false;
                    continue;
                }
                if (bitmap.GetValue(logical) != -1)
                    valid = false;
                bitmap.SetValue(logical, u);
                int64_t phys = table.GetValue(b * N + logical / 128);
                if (phys < 0 || phys >= hostTokens / 128 ||
                    phys * 128 + logical % 128 != keys.GetValue(ix))
                    valid = false;
            }
            for (int64_t u = cnt; u < W; ++u)
                if (keys.GetValue(b * W + u) != -1 || slots.GetValue(b * W + u) != -1 ||
                    owner.GetValue(b * W + u) != -1 || miss.GetValue(b * W + u) ||
                    cache.GetValue(b * W + u))
                    valid = false;
            if (!valid)
                status.SetValue(b, 1);
        }
    }
    FillUnique(source, dest, requestp, keyp, slotp, countp, missp, statusp, readyp, statusp,
               counterp, B, W, C, delay, fail);
    for (int64_t b = 0; b < B; ++b)
        if (status.GetValue(b) == 0 && reqs.GetValue(b) > 0) {
            auto cnt = count.GetValue(b);
            bool ok = cnt >= 0 && cnt <= C;
            if (ok)
                for (int64_t u = 0; u < cnt; ++u)
                    if (ready.GetValue(b * W + u) != 2)
                        ok = false;
            if (ok)
                for (int64_t j = 0; j < 9; ++j)
                    done.SetValue(b * 9 + j, id.GetValue(b * 9 + j));
            else
                status.SetValue(b, 3);
        }
    Flush(completionp, B * 9 * 8);
    Flush(statusp, B * 4);
}
