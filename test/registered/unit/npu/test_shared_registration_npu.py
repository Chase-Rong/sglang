"""Isolate each access capability of a freshly registered SysV allocation."""

import ctypes
import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_UNIFIED_HOST_TEST") != "1", reason="requires idle NPU"
)


def test_shared_registration_capability():
    import torch_npu
    from sdma_copy_lab import AclCopyBridge
    from sglang.srt.hardware_backend.npu.fsa_native import (
        RegisteredHostRegion,
        copy_rows_from_registered_host,
    )

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch.npu.set_device(device_id)
    flags = int(os.environ["SGLANG_UNIFIED_HOST_FLAGS"])
    access = os.environ.get("UNIFIED_ACCESS", "dma_host")
    if (
        flags == 268435456
        and access != "aiv"
        and os.environ.get("UNIFIED_REPRO_PIN_CRASH") != "1"
    ):
        pytest.skip("fresh SysV PINNED DMA crashed; explicit reproduction only")
    size = 8 * 1024 * 1024
    libc = ctypes.CDLL(None)
    libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
    libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    libc.shmat.restype = ctypes.c_void_p
    libc.shmdt.argtypes = [ctypes.c_void_p]
    libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    segment = libc.shmget(0, size, 0o1000 | 0o600)
    assert segment >= 0
    host = libc.shmat(segment, None, 0)
    removed = libc.shmctl(segment, 0, None)
    assert host != ctypes.c_void_p(-1).value
    if removed:
        libc.shmdt(host)
        raise RuntimeError("failed to mark shared segment for removal")
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtHostRegisterV2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
    ]
    acl.aclrtHostGetDevicePointer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
    ]
    acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    registered = False
    bridge = AclCopyBridge(os.environ["SGLANG_SDMA_TEST_LIBRARY"])
    stream = torch.npu.current_stream()
    try:
        code = acl.aclrtHostRegisterV2(host, size, flags)
        print("REGISTER", flags, code, flush=True)
        assert code == 0
        registered = True
        mapped = ctypes.c_void_p()
        code = acl.aclrtHostGetDevicePointer(host, ctypes.byref(mapped), 0)
        print("MAPPING", code, mapped.value, flush=True)
        tensor = torch.frombuffer(
            (ctypes.c_ubyte * size).from_address(host), dtype=torch.bfloat16
        ).reshape(-1, 512)
        device = torch.empty_like(tensor, device=f"npu:{device_id}")
        tensor.fill_(0.75)
        if access == "aiv":
            assert code == 0 and mapped.value
            region = RegisteredHostRegion(
                tensor, host, mapped.value, device_id, "probe", False
            )
            ids = torch.arange(tensor.shape[0], dtype=torch.int64, device=device.device)
            valid = torch.ones_like(ids, dtype=torch.bool)
            copy_rows_from_registered_host(region, device, ids, ids, valid)
            stream.synchronize()
            assert torch.equal(device.cpu(), tensor)
            print("AIV_PASS", flush=True)
            return
        direction = 3 if access == "dma_alias" else 1
        source = mapped.value if direction == 3 else host
        assert source
        destination = str(
            Path(os.environ["SGLANG_FSA_WORK_DIR"])
            / f"registration_trace_{flags}_{access}"
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(destination),
        ):
            for generation in range(3):
                tensor.fill_(generation + 0.5)
                started = time.perf_counter()
                bridge.submit(
                    source,
                    device.data_ptr(),
                    ((0, 0, size),),
                    direction,
                    stream,
                    (tensor, device),
                )
                submitted_ms = (time.perf_counter() - started) * 1000
                stream.synchronize()
                assert torch.equal(device.cpu(), tensor)
                print("DMA_H2D_PASS", generation, submitted_ms, flush=True)
                device.add_(0.25)
                bridge.submit(
                    device.data_ptr(),
                    source,
                    ((0, 0, size),),
                    3 if direction == 3 else 2,
                    stream,
                    (tensor, device),
                )
                stream.synchronize()
                assert torch.equal(device.cpu(), tensor)
                print("DMA_D2H_PASS", generation, flush=True)
                bridge.release_after_drain()
    finally:
        stream.synchronize()
        bridge.release_after_drain()
        if registered:
            code = acl.aclrtHostUnregister(host)
            print("UNREGISTER", code, flush=True)
            if code:
                raise RuntimeError("retain Host mapping after failed unregister")
        print("DETACH", libc.shmdt(host), flush=True)
