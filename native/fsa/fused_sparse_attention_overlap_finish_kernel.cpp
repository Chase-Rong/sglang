// Phase 3 direct kernels. Single AIV, stream ordering; no inter-core polling.
#include "kernel_operator.h"
using namespace AscendC;
namespace {
constexpr int64_t MAGIC=0x485350335631LL;
template<class T> __aicore__ inline GlobalTensor<T> G(GM_ADDR p) {
    GlobalTensor<T> g; g.SetGlobalBuffer((__gm__ T*)p); return g;
}
__aicore__ inline void Flush(GM_ADDR p,int64_t bytes) {
    auto g=G<int8_t>(p);
    for(int64_t i=0;i<bytes;i+=64) DataCacheCleanAndInvalid<int8_t,CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(g[i]);
}
}
extern "C" __global__ __aicore__ void fused_sparse_attention_overlap_finish(
 GM_ADDR outputp,GM_ADDR requestp,GM_ADDR rowp,GM_ADDR rowunionp,GM_ADDR slotp,GM_ADDR countp,
 GM_ADDR idp,GM_ADDR readyp,GM_ADDR statusp,GM_ADDR completionp,GM_ADDR safep,GM_ADDR membershipp,GM_ADDR missp,GM_ADDR counterp,
 int64_t B,int64_t T,int64_t K,int64_t W,int64_t C) {
 if(GetBlockIdx()!=0)return;
 auto req=G<int64_t>(requestp),id=G<int64_t>(idp),done=G<int64_t>(completionp);
 auto row=G<int32_t>(rowp),ru=G<int32_t>(rowunionp),slots=G<int32_t>(slotp),count=G<int32_t>(countp);
 auto ready=G<int32_t>(readyp),status=G<int32_t>(statusp),safe=G<int32_t>(safep);
 auto miss=G<bool>(missp);auto counters=G<int64_t>(counterp);
 auto member=G<int16_t>(membershipp);auto out=G<uint16_t>(outputp);
 for(int64_t b=0;b<B;++b) {
    if(counters.GetValue(b*8+3)!=0)status.SetValue(b,4);
    if(status.GetValue(b)!=0||req.GetValue(b)<=0)continue;
    auto n=count.GetValue(b);bool ok=n>=0&&n<=W&&n<=C;
    if(ok)for(int64_t u=0;u<n;++u)if(ready.GetValue(b*W+u)!=2)ok=false;
    int64_t expected=0;if(ok)for(int64_t u=0;u<n;++u)if(miss.GetValue(b*W+u))++expected;
    if(counters.GetValue(b*8)!=expected||counters.GetValue(b*8+1)!=expected||counters.GetValue(b*8+2)!=0)ok=false;
    for(int64_t t=0;t<T;++t)if(row.GetValue(t)==b) {
      auto h=t*16400+16384;
      if(member.GetValue(h)!=0x5A4D||member.GetValue(h+1)!=0x5A45||member.GetValue(h+2)!=K||
         member.GetValue(h+3)!=16384-K||member.GetValue(h+4)!=req.GetValue(b)||
         member.GetValue(h+5)!=0x5A53||member.GetValue(h+6)!=C||member.GetValue(h+7)!=3||member.GetValue(h+8)!=W)ok=false;
      for(int64_t k=0;k<K;++k){auto u=ru.GetValue(t*K+k);auto v=member.GetValue(t*16400+16384-K+k);
         if(u>=0&&(u>=n||(v<0?-v:v)!=slots.GetValue(b*W+u)+1||safe.GetValue(t*K+k)<0))ok=false;
         if(u<0&&(v!=0||safe.GetValue(t*K+k)!=-1))ok=false;
      }
    }
    if(ok)for(int64_t j=0;j<9;++j)done.SetValue(b*9+j,id.GetValue(b*9+j));
    else status.SetValue(b,4);
 }
 for(int64_t t=0;t<T;++t) {
    auto b=row.GetValue(t);
    if(b<0||b>=B||req.GetValue(b)<=0) {
       bool failed=b>=B;
       for(int64_t q=0;q<B;++q)if(status.GetValue(q)!=0)failed=true;
       for(int64_t x=0;x<32*512;++x)out.SetValue(t*32*512+x,failed?0x7fc0:0);
    }else if(status.GetValue(b)!=0)for(int64_t x=0;x<32*512;++x)out.SetValue(t*32*512+x,0x7fc0);
 }
 Flush(outputp,T*32*512*2);Flush(statusp,B*4);Flush(completionp,B*9*8);
}
