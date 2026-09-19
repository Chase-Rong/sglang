#pragma once

#include <acl/acl.h>
#include <ATen/ATen.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace sglang::npu_kernel {
template<class Payload>
class HostCallback {
    struct Retained {
        aclmdlRI graph;
        std::vector<std::unique_ptr<Payload>> payloads;
    };

    static auto &Registry()
    {
        static auto *registry = new std::unordered_map<aclmdlRI, std::unique_ptr<Retained>>();
        return *registry;
    }

    static auto &Mutex()
    {
        static auto *mutex = new std::mutex();
        return *mutex;
    }

    static void Destroy(void *data) noexcept
    {
        auto *retained = static_cast<Retained *>(data);
        std::lock_guard<std::mutex> lock(Mutex());
        Registry().erase(retained->graph);
    }

    static void Run(void *data) noexcept
    {
        auto *payload = static_cast<Payload *>(data);
        payload->Run();
        if (!payload->captured) {
            delete payload;
        }
    }

public:
    static void Enqueue(const at::Tensor &trigger, std::unique_ptr<Payload> payload)
    {
        const auto stream = c10_npu::getCurrentNPUStream();
        TORCH_CHECK(trigger.device().type() == at::DeviceType::PrivateUse1 &&
                    trigger.device().index() == stream.device_index(), "Host callback device mismatch");
        aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        aclmdlRI graph = nullptr;
        TORCH_CHECK(aclmdlRICaptureGetInfo(stream.stream(), &status, &graph) == ACL_SUCCESS &&
                    status != ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED, "Host callback capture query failed");
        payload->captured = status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
        const bool captured = payload->captured;
        auto *raw = payload.get();
        if (captured) {
            TORCH_CHECK(graph != nullptr, "Host callback capture has no graph");
            std::lock_guard<std::mutex> lock(Mutex());
            auto &retained = Registry()[graph];
            if (!retained) {
                retained = std::make_unique<Retained>();
                retained->graph = graph;
                if (aclmdlRIDestroyRegisterCallback(graph, Destroy, retained.get()) != ACL_SUCCESS) {
                    Registry().erase(graph);
                    TORCH_CHECK(false, "Host callback graph retention failed");
                }
            }
            retained->payloads.push_back(std::move(payload));
        }
        const auto result = aclrtLaunchHostFunc(stream.stream(), Run, raw);
        if (result == ACL_SUCCESS && !captured) {
            payload.release();
        }
        TORCH_CHECK(result == ACL_SUCCESS, "Host callback launch failed: ", result);
    }
};
}
