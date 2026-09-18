#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(ACL_MEMCPY_HOST_TO_DEVICE == 1);
static_assert(ACL_MEMCPY_DEVICE_TO_HOST == 2);

extern "C" int fsa_sdma_pin_host(void* pointer, uint64_t size)
{
    return aclrtHostRegisterV2(pointer, size, ACL_HOST_REG_PINNED);
}

extern "C" int fsa_sdma_unpin_host(void* pointer)
{
    return aclrtHostUnregister(pointer);
}

extern "C" int fsa_sdma_copy(const uint64_t* sources, const uint64_t* destinations,
                             const uint64_t* sizes, size_t count, int direction,
                             void* stream, size_t* failed_index)
{
    *failed_index = std::numeric_limits<size_t>::max();
    if (direction != ACL_MEMCPY_HOST_TO_DEVICE && direction != ACL_MEMCPY_DEVICE_TO_HOST &&
        direction != ACL_MEMCPY_DEVICE_TO_DEVICE) {
        return ACL_ERROR_INVALID_PARAM;
    }
    for (size_t index = 0; index < count; ++index) {
        const auto result = aclrtMemcpyAsync(
            reinterpret_cast<void*>(destinations[index]), sizes[index],
            reinterpret_cast<const void*>(sources[index]), sizes[index],
            static_cast<aclrtMemcpyKind>(direction), stream);
        if (result != ACL_SUCCESS) {
            *failed_index = index;
            return result;
        }
    }
    return ACL_SUCCESS;
}
