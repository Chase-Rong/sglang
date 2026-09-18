"""Validation adapter for MemFabric offload allocations without an extra mirror."""

import ctypes
import math
import os

import torch

from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion


class MemFabricHostRegion(RegisteredHostRegion):
    pool_initialized = False

    def export_handle(self):
        raise NotImplementedError(
            "MemFabric TP sharing is not qualified by this adapter"
        )

    @classmethod
    def attach(cls, handle, *, device_id, name=""):
        raise NotImplementedError("MemFabric handles cannot use SysV attachment")

    @classmethod
    def initialize_pool(cls, device_id, rank=0, world_size=1):
        import memfabric_hybrid as fabric

        if cls.pool_initialized:
            return
        config = fabric.offload.OffloadConfig()
        config.device_id = device_id
        config.rank_id = rank
        config.world_size = world_size
        config.scene = (
            fabric.offload.Scene.LOCAL
            if world_size == 1
            else fabric.offload.Scene.SHARED
        )
        config.reserve_size = 1 << 30
        config.alloc_size = (1 << 30) if rank == 0 else 0
        print(
            "MEMFABRIC_INIT",
            {"device": device_id, "rank": rank, "world": world_size},
            flush=True,
        )
        result = fabric.offload.initialize(config)
        print("MEMFABRIC_INIT_RESULT", result, flush=True)
        if result:
            raise RuntimeError(f"MemFabric Host initialization failed: {result}")
        cls.pool_initialized = True

    @classmethod
    def allocate(cls, shape, dtype, *, device_id=0, name=""):
        import memfabric_hybrid as fabric

        cls.initialize_pool(device_id)
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        pointer = fabric.offload.malloc(size)
        if not pointer:
            raise RuntimeError("MemFabric Host malloc failed")
        raw = (ctypes.c_ubyte * size).from_address(pointer)
        tensor = torch.frombuffer(raw, dtype=dtype).reshape(shape)
        device = ctypes.c_void_p()
        if hasattr(fabric.offload, "get_dva"):
            device.value = fabric.offload.get_dva(pointer)
            code = 0 if device.value else -1
        else:
            library = ctypes.CDLL(os.environ["SGLANG_UNIFIED_HOST_LIBRARY"])
            library.unified_host_device_pointer.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            library.unified_host_device_pointer.restype = ctypes.c_int
            code = library.unified_host_device_pointer(pointer, ctypes.byref(device))
        print(
            "MEMFABRIC_ALLOCATION",
            {
                "bytes": size,
                "host": pointer,
                "mapping_result": code,
                "device": device.value,
            },
            flush=True,
        )
        if code:
            fabric.offload.free(pointer)
            raise RuntimeError(f"MemFabric device mapping query failed: {code}")
        region = cls(tensor, pointer, device.value, device_id, name, False)
        region.binding_registered = False
        if os.environ.get("SGLANG_MEMFABRIC_REGISTER_BINDING") == "1":
            from sglang.srt.hardware_backend.npu.fsa_native import native_ops

            try:
                native_ops().register_memfabric_region(tensor, device.value, device_id)
            except BaseException:
                fabric.offload.free(pointer)
                raise
            region.binding_registered = True
        return region

    def close(self, *, drained=False):
        import memfabric_hybrid as fabric

        if self.closed:
            return
        if not drained:
            raise RuntimeError("drain before closing MemFabric Host")
        if getattr(self, "binding_registered", False):
            from sglang.srt.hardware_backend.npu.fsa_native import native_ops

            native_ops().unregister_memfabric_region(self.tensor, self.device_id)
            self.binding_registered = False
        fabric.offload.free(self.host_ptr)
        self.closed = True
        self.host_ptr = self.device_ptr = 0
        self.tensor = torch.empty(0, dtype=self.tensor.dtype)

    @classmethod
    def close_pool(cls):
        if cls.pool_initialized:
            import memfabric_hybrid as fabric

            fabric.offload.uninitialize()
            cls.pool_initialized = False
