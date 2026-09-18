"""Fresh single-registration shared Host candidate; validation scope only."""

import ctypes
import math
import os

import torch

from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion


class UnifiedHostRegion(RegisteredHostRegion):
    @staticmethod
    def _library():
        library = ctypes.CDLL(os.environ["SGLANG_UNIFIED_HOST_LIBRARY"])
        pointer = ctypes.POINTER(ctypes.c_void_p)
        library.unified_host_create.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int),
            pointer,
            pointer,
        ]
        library.unified_host_attach.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint32,
            pointer,
            pointer,
        ]
        library.unified_host_release.argtypes = [ctypes.c_void_p]
        for name in (
            "unified_host_create",
            "unified_host_attach",
            "unified_host_release",
        ):
            getattr(library, name).restype = ctypes.c_int
        return library

    @classmethod
    def _construct(cls, shape, dtype, device_id, name, segment=None):
        if torch.npu.current_device() != device_id:
            raise ValueError("set the target device before shared allocation")
        if not shape or any(
            type(dimension) is not int or dimension <= 0 for dimension in shape
        ):
            raise ValueError("invalid Host shape")
        count = math.prod(shape)
        size = count * torch.empty((), dtype=dtype).element_size()
        flags = int(os.environ.get("SGLANG_UNIFIED_HOST_FLAGS", "268435458"))
        library = cls._library()
        host, device = ctypes.c_void_p(), ctypes.c_void_p()
        identity = ctypes.c_int(-1 if segment is None else segment)
        if segment is None:
            code = library.unified_host_create(
                size,
                flags,
                ctypes.byref(identity),
                ctypes.byref(host),
                ctypes.byref(device),
            )
        else:
            code = library.unified_host_attach(
                segment, size, flags, ctypes.byref(host), ctypes.byref(device)
            )
        print(
            "UNIFIED_REGISTER",
            {
                "code": code,
                "flags": flags,
                "segment": identity.value,
                "host": host.value,
                "device": device.value,
            },
            flush=True,
        )
        if code:
            raise RuntimeError(
                f"Unified Host registration failed: code={code}, flags={flags}"
            )
        try:
            buffer = (ctypes.c_ubyte * size).from_address(host.value)
            tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
            region = cls(tensor, host.value, device.value, device_id, name, False)
            region.segment = identity.value
            region.flags = flags
            region.library = library
            return region
        except BaseException:
            library.unified_host_release(host)
            raise

    @classmethod
    def allocate(cls, shape, dtype, *, device_id=0, name=""):
        return cls._construct(tuple(shape), dtype, device_id, name)

    def export_handle(self):
        self._require_live()
        namespace = os.stat("/proc/self/ns/ipc")
        return dict(
            segment=self.segment,
            shape=tuple(self.tensor.shape),
            dtype=str(self.tensor.dtype),
            flags=self.flags,
            namespace=(namespace.st_dev, namespace.st_ino),
        )

    @classmethod
    def attach(cls, handle, *, device_id, name=""):
        namespace = os.stat("/proc/self/ns/ipc")
        if (namespace.st_dev, namespace.st_ino) != handle["namespace"]:
            raise ValueError("shared Host IPC namespace differs")
        if (
            int(os.environ.get("SGLANG_UNIFIED_HOST_FLAGS", "268435458"))
            != handle["flags"]
        ):
            raise ValueError("shared Host registration flags differ")
        return cls._construct(
            handle["shape"],
            getattr(torch, handle["dtype"].removeprefix("torch.")),
            device_id,
            name,
            segment=handle["segment"],
        )

    def close(self, *, drained=False):
        if self.closed:
            return
        if not drained:
            raise RuntimeError("shared Host users must drain before close")
        code = self.library.unified_host_release(self.host_ptr)
        if code:
            raise RuntimeError(f"Unified Host release failed: {code}")
        self.closed = True
        self.host_ptr = self.device_ptr = 0
        self.tensor = torch.empty(0, dtype=self.tensor.dtype)
