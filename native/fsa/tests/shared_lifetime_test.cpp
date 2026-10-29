#include "shared_lifetime.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

using sglang::npu_kernel::SharedAclLifetime;
using sglang::npu_kernel::SharedScopeExit;

void CheckCleanup(int failure)
{
    int cleaned = 0;
    int uninitialized = 0;
    {
        SharedScopeExit thread([&] { ++uninitialized; });
        auto lifetime = std::make_shared<SharedAclLifetime>([&] { ++cleaned; });
        if (failure == 1) {
            // Workspace-query/submission-preparation failure: caller owns cleanup.
            lifetime.reset();
            assert(cleaned == 1);
        } else {
            auto callback = [lifetime, &cleaned, failure] {
                lifetime->EnterHandler();
                SharedScopeExit handler([lifetime] { lifetime->Cleanup(); });
                assert(cleaned == 0);
                if (failure == 2) throw std::runtime_error("API failure");
            };
            lifetime->MarkQueued();
            if (failure != 3) {
                try { callback(); }
                catch (const std::runtime_error &) {}
                // The queue and caller still hold the callback/owner, but the
                // executing operation's TLS arena must already be released.
                assert(cleaned == 1);
                lifetime->Cleanup();
                assert(cleaned == 1);
            } else {
                // Unknown queue/TLS on cancellation: no deferred TLS cleanup.
                assert(cleaned == 0);
            }
        }
    }
    assert(uninitialized == 1);
    assert(cleaned == (failure == 3 ? 0 : 1));
    std::cout << "case=" << failure << " cleanup=" << cleaned
              << " uninit=" << uninitialized << "\n";
}

int main()
{
    CheckCleanup(0);
    CheckCleanup(1);
    CheckCleanup(2);
    CheckCleanup(3);
}
