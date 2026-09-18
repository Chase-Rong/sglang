#include <acl/acl.h>
#include <cstdint>
#include <sys/shm.h>

extern "C" int unified_host_device_pointer(void* host, void** device)
{
    return aclrtHostGetDevicePointer(host, device, 0);
}

extern "C" int unified_host_create(uint64_t bytes, uint32_t flags, int* segment,
                                   void** host, void** device)
{
    *host = nullptr;
    *device = nullptr;
    *segment = shmget(IPC_PRIVATE, bytes, IPC_CREAT | 0600);
    if (*segment < 0) return -1;
    void* address = shmat(*segment, nullptr, 0);
    if (address == reinterpret_cast<void*>(-1)) {
        shmctl(*segment, IPC_RMID, nullptr);
        return -2;
    }
    if (shmctl(*segment, IPC_RMID, nullptr) != 0) {
        shmdt(address);
        return -3;
    }
    auto result = aclrtHostRegisterV2(address, bytes, flags);
    if (result != ACL_SUCCESS) {
        shmdt(address);
        return result;
    }
    result = aclrtHostGetDevicePointer(address, device, 0);
    if (result != ACL_SUCCESS) {
        aclrtHostUnregister(address);
        shmdt(address);
        return result;
    }
    *host = address;
    return 0;
}

extern "C" int unified_host_attach(int segment, uint64_t bytes, uint32_t flags,
                                   void** host, void** device)
{
    *host = nullptr;
    *device = nullptr;
    struct shmid_ds information {};
    if (shmctl(segment, IPC_STAT, &information) != 0 || information.shm_segsz != bytes) return -4;
    void* address = shmat(segment, nullptr, 0);
    if (address == reinterpret_cast<void*>(-1)) return -2;
    auto result = aclrtHostRegisterV2(address, bytes, flags);
    if (result != ACL_SUCCESS) {
        shmdt(address);
        return result;
    }
    result = aclrtHostGetDevicePointer(address, device, 0);
    if (result != ACL_SUCCESS) {
        aclrtHostUnregister(address);
        shmdt(address);
        return result;
    }
    *host = address;
    return 0;
}

extern "C" int unified_host_release(void* host)
{
    const auto result = aclrtHostUnregister(host);
    if (result != ACL_SUCCESS) return result;
    return shmdt(host);
}
