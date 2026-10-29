#pragma once
#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <utility>
namespace sglang::npu_kernel {
class SharedScopeExit {
 public:
  explicit SharedScopeExit(std::function<void()> fn):fn_(std::move(fn)) {}
  SharedScopeExit(const SharedScopeExit&)=delete;
  ~SharedScopeExit() noexcept { try { fn_(); } catch (...) {} }
 private:
  std::function<void()> fn_;
};
// ACL HugeMem is thread-local, not a resource attached to a captured arena token.
// Cleanup belongs to the caller before queueing, or the executing handler scope.
// A queued-but-never-executed cancellation must NOT release another op's TLS arena.
class SharedAclLifetime {
 public:
  enum Phase { Prepared, Queued, Executing, Cleaned };
  explicit SharedAclLifetime(std::function<void()> fn):cleanup_(std::move(fn)) {}
  SharedAclLifetime(const SharedAclLifetime&)=delete;
  void SetCleanup(std::function<void()> fn) { cleanup_=std::move(fn); }
  void MarkQueued() { phase_.store(Queued); }
  void EnterHandler() { phase_.store(Executing); }
  void Cleanup() noexcept {
    if(phase_.exchange(Cleaned)==Cleaned)return;
    try { cleanup_(); } catch (...) {}
    cleanup_=nullptr;
  }
  ~SharedAclLifetime() noexcept {
    auto phase=phase_.load();
    if(phase==Prepared)Cleanup();
    else if(phase!=Cleaned) {
      // Raw ACL descriptor allocations are intentionally left quarantined. No
      // Tensor owner is retained, and no TLS API runs in a task deleter.
      std::fputs("external_plan: queued ACL call never completed; descriptor/HugeMem resources quarantined without TLS cleanup\n",stderr);
    }
  }
 private:
  std::atomic<Phase> phase_{Prepared};
  std::function<void()> cleanup_;
};
}
