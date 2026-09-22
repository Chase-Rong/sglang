// Native contract test: wrong hit protection, hash deletion, reset or slot reuse
// must fail independently of the chosen replacement order.
#include "clock_plan.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <vector>
using sglang::npu_kernel::ClockWorkspaceSize;
using sglang::npu_kernel::PlanClockRow;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::abort(); } } while(0)
template<typename Index>
struct Row {
    int64_t capacity;
    std::vector<int64_t> resident, ages;
    std::vector<Index> workspace;
    std::vector<uint8_t> protected_slots;
    int64_t state[3]={-1,-1,-1};
    explicit Row(int64_t c):capacity(c),resident(c,-1),ages(c,0),workspace(ClockWorkspaceSize(c),-1),protected_slots(c){}
    std::vector<int64_t> run(const std::vector<int64_t>& keys,int64_t owner=1,int64_t length=100) {
        auto before=resident; auto prior_owner=state[0];auto prior_len=state[1];auto prior_epoch=state[2];
        std::vector<int64_t> out(keys.size(),-999);
        PlanClockRow(keys.data(),keys.size(),owner,length,resident.data(),ages.data(),state,out.data(),capacity,workspace.data(),protected_slots.data());
        std::set<int64_t> used;
        for(size_t i=0;i<out.size();++i) {
            CHECK(out[i]>=0 && out[i]<capacity);CHECK(used.insert(out[i]).second);
            if(owner>=0 && length>0 && keys[i]>=0)CHECK(resident[out[i]]==keys[i]);
        }
        if(owner>=0 && length>0 && owner==prior_owner && length>=prior_len && prior_epoch<std::numeric_limits<int64_t>::max()/2) {
            // Every previously resident requested key must retain at least one
            // original slot. A miss appearing before a hit must not evict it.
            for(auto key:keys) if(key>=0) {
                auto it=std::find(before.begin(),before.end(),key);
                if(it!=before.end()) {
                    bool retained=false;
                    for(size_t j=0;j<before.size();++j)if(before[j]==key && resident[j]==key)retained=true;
                    CHECK(retained);
                }
            }
        }
        return out;
    }
};
template<typename Index>
void CheckContract() {
    Row<Index> r(8);auto first=r.run({11,22,33,44});auto warm=r.run({44,11,33,22});
    CHECK(warm[0]==first[3] && warm[1]==first[0] && warm[2]==first[2] && warm[3]==first[1]);
    r.run({100,200,300,400,44,11,33,22});
    auto dup=r.run({77,77,77,-1,-1,11,22,33});
    CHECK(dup[0]!=dup[1] && dup[1]!=dup[2]);
    r.run({-1,-1,-1,-1,-1,-1,-1,-1});
    const auto unchanged=r.resident;
    r.run({-1,-1,-1,-1,-1,-1,-1,-1});CHECK(r.resident==unchanged);
    r.run({555},2,100);CHECK(std::count(r.resident.begin(),r.resident.end(),555)==1);
    CHECK(std::count(r.resident.begin(),r.resident.end(),-1)==7);
    r.run({std::numeric_limits<int64_t>::max(), 0, 1});
    r.run({1, std::numeric_limits<int64_t>::max(), 0});
    // Exercise deletion from the head, middle and tail of one real hash chain.
    std::vector<int64_t> collisions;
    for(int64_t key=0;collisions.size()<32;++key)
        if(sglang::npu_kernel::ClockBucket(key,15)==0)collisions.push_back(key);
    Row<Index> collided(8);
    for(int step=0;step<100;++step) {
        std::vector<int64_t> keys;
        for(int i=0;i<7;++i)keys.push_back(collisions[(step*3+i)%32]);
        collided.run(keys);
    }
    r.run({666},2,50);CHECK(std::count(r.resident.begin(),r.resident.end(),-1)==7);
    r.state[2]=std::numeric_limits<int64_t>::max()/2;r.run({777},2,50);
    CHECK(std::count(r.resident.begin(),r.resident.end(),-1)==7);
    r.run({-1,-1},-1,0);r.run({123},2,51);
    CHECK(std::count(r.resident.begin(),r.resident.end(),-1)==7);
    std::mt19937_64 rng(917);
    for(int capacity:{1,2,3,8,31,64,4096}) {
        Row<Index> a(capacity),b(capacity);
        for(int step=0;step<1500;++step) {
            int width=1+rng()%std::min(capacity,2051);
            std::vector<int64_t> keys(width);
            for(auto& key:keys)key=rng()%7==0?-1:static_cast<int64_t>((rng()%(capacity*3))*8192);
            auto &row=step%2?a:b;
            row.run(keys,step/137+1,100+step%89);
        }
    }
}
void CheckMaximumSlotAndHandWrap() {
    constexpr int64_t capacity=32767, width=2051;
    Row<int64_t> legacy(capacity);
    Row<int16_t> compact(capacity);
    std::vector<int64_t> keys(width), legacy_slots(width), compact_slots(width);
    bool maximum_slot_seen=false, hand_wrapped=false;
    int64_t previous_hand=0;
    for(int64_t step=0;step<20;++step) {
        // Keep one owner and increasing lengths so the table fills past the
        // largest legal int16 slot without an intervening reset.
        for(int64_t i=0;i<width;++i)keys[i]=(step*width+i)*128;
        PlanClockRow(keys.data(),width,1,100+step,legacy.resident.data(),
                     legacy.ages.data(),legacy.state,legacy_slots.data(),capacity,
                     legacy.workspace.data(),legacy.protected_slots.data());
        PlanClockRow(keys.data(),width,1,100+step,compact.resident.data(),
                     compact.ages.data(),compact.state,compact_slots.data(),capacity,
                     compact.workspace.data(),compact.protected_slots.data());
        CHECK(legacy_slots==compact_slots);
        CHECK(legacy.resident==compact.resident && legacy.ages==compact.ages);
        CHECK(std::equal(legacy.state,legacy.state+3,compact.state));
        CHECK(std::equal(legacy.workspace.begin(),legacy.workspace.end(),compact.workspace.begin()));
        CHECK(legacy.protected_slots==compact.protected_slots);
        for(auto slot:compact_slots) {
            CHECK(slot>=0 && slot<capacity);
            maximum_slot_seen|=slot==capacity-1;
        }
        hand_wrapped|=compact.workspace.back()<previous_hand;
        previous_hand=compact.workspace.back();
    }
    CHECK(maximum_slot_seen && hand_wrapped);
}
int main() {
    CheckContract<int64_t>();
    CheckContract<int16_t>();
    CheckMaximumSlotAndHandWrap();
    std::puts("CLOCK_CONTRACT_OK");
}
