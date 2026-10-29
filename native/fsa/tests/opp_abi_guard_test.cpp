#include "opp_abi_guard.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace sglang::npu_kernel;

void *Load(const char *path) {
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) std::cerr << dlerror() << '\n';
    assert(handle);
    return handle;
}
OverlapOppApi Api(void *handle) {
    return {dlsym(handle, "aclnnFusedSparseAttentionOverlapGetWorkspaceSize"),
            dlsym(handle, "aclnnFusedSparseAttentionOverlap")};
}
int Calls(void *handle) {
    auto count = reinterpret_cast<int (*)()>(dlsym(handle, "fixtureCallCount"));
    assert(count);
    return count();
}
int main(int argc, char **argv) {
    assert(argc == 4);
    void *legacy = Load(argv[1]), *wrong = Load(argv[2]), *current = Load(argv[3]);
    auto old_api = Api(legacy), wrong_api = Api(wrong), new_api = Api(current);
    void *wrong_marker = dlsym(wrong, kFusedSparseAttentionOverlapAbiSymbol);
    void *new_marker = dlsym(current, kFusedSparseAttentionOverlapAbiSymbol);
    assert(!dlsym(legacy, kFusedSparseAttentionOverlapAbiSymbol));
    assert(ValidateOverlapOppAbi(old_api, nullptr));
    assert(ValidateOverlapOppAbi(wrong_api, wrong_marker));
    assert(ValidateOverlapOppAbi(old_api, new_marker));
    assert(ValidateOverlapOppAbi({old_api.get_workspace_size, new_api.launch}, new_marker));
    assert(ValidateOverlapOppAbi({new_api.get_workspace_size, old_api.launch}, new_marker));
    assert(ValidateOverlapOppAbi({nullptr, new_api.launch}, new_marker));
    assert(ValidateOverlapOppAbi({new_api.get_workspace_size, nullptr}, new_marker));
    assert(Calls(legacy) == 0 && Calls(wrong) == 0 && Calls(current) == 0);
    assert(ValidateOverlapOppAbi(new_api, new_marker) == nullptr);
    // The caller uses the same validated pair, without re-resolving names.
    reinterpret_cast<void (*)()>(new_api.get_workspace_size)();
    reinterpret_cast<void (*)()>(new_api.launch)();
    assert(Calls(current) == 2 && Calls(legacy) == 0 && Calls(wrong) == 0);
    dlclose(current); dlclose(wrong); dlclose(legacy);
    std::cout << "OPP_ABI_GUARD_PASS rejected7_before_call accepted_same_dso_abi3\n";
}
