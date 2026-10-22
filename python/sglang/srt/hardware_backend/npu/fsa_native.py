import ctypes
import os
from dataclasses import dataclass
from math import prod
from pathlib import Path
from threading import RLock
from typing import Optional, Sequence, Tuple
from uuid import uuid4

import torch


def _ctype_for_dtype(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return ctypes.c_uint16
    if dtype == torch.float32:
        return ctypes.c_float
    if dtype == torch.float64:
        return ctypes.c_double
    if dtype == torch.int8:
        return ctypes.c_int8
    if dtype == torch.uint8:
        return ctypes.c_uint8
    if dtype == torch.int16:
        return ctypes.c_int16
    if dtype == torch.int32:
        return ctypes.c_int32
    if dtype == torch.int64:
        return ctypes.c_int64
    if dtype == torch.bool:
        return ctypes.c_bool
    raise TypeError(f"unsupported shm tensor dtype: {dtype}")


def create_shm_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int = 0,
    name: str = "",
) -> Tuple[torch.Tensor, int, int]:
    """Create host shared memory and register it to an NPU device.

    Returns ``(host_tensor, host_ptr, dev_ptr)``. ``host_tensor`` is a CPU
    tensor backed by the registered shared memory. ``dev_ptr`` is the
    device-visible address and can be passed to sparse KV kernels through
    ``src_ptr``/``dst_ptr``.
    """
    shape_tuple = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape_tuple):
        raise ValueError(f"shape dimensions must be non-negative, got {shape_tuple}")

    numel = int(prod(shape_tuple))
    elem_size = torch.empty((), dtype=dtype).element_size()
    size = numel * elem_size
    if size <= 0:
        raise ValueError(f"shm tensor size must be positive, got shape={shape_tuple}")
    buffer_type = _ctype_for_dtype(dtype) * numel
    name = name or f"sgl-tensor-{uuid4().hex}"

    host_ptr, dev_ptr = native_ops().shm_allocator_create_and_register(
        size, device_id, name
    )
    try:
        buffer = buffer_type.from_address(host_ptr)
        tensor = torch.frombuffer(buffer, dtype=dtype).view(shape_tuple)
        if tensor.element_size() != elem_size:
            raise RuntimeError("shm tensor element size mismatch")
    except Exception:
        native_ops().shm_allocator_free_by_name(name, device_id)
        raise
    return tensor, int(host_ptr), int(dev_ptr)


def free_shm(device_id: int = 0) -> None:
    """Free this process's registrations for one device after all users drain."""
    native_ops().shm_allocator_free_all(device_id)


@dataclass
class SharedHostHandle:
    """Linux IPC-namespace-local segment identity; never a process pointer.

    An exporter must retain its mapping until every consumer has attached.
    Consumers register their own device address and detach independently.
    This handle belongs on a trusted node-local bootstrap channel, not PD wire.
    """

    segment: int
    nbytes: int
    creator: int
    created: int
    ipc_device: int
    ipc_inode: int
    shape: Tuple[int, ...]
    dtype: str
    version: int = 1


@dataclass
class RegisteredHostRegion:
    """Host tensor plus its NPU-visible registered address.

    Explicit close requires drained device, transport and captured-graph users.
    Tensor aliases must not outlive close. There is deliberately no destructor
    that could unregister memory while asynchronous users still reference it.
    """

    tensor: torch.Tensor
    host_ptr: int
    device_ptr: int
    device_id: int
    name: str
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        device_id: int = 0,
        name: str = "",
    ) -> "RegisteredHostRegion":
        name = name or f"sgl-region-{uuid4().hex}"
        tensor, host_ptr, device_ptr = create_shm_tensor(
            shape, dtype, device_id=device_id, name=name
        )
        return cls(tensor, host_ptr, device_ptr, device_id, name)

    def export_handle(self) -> SharedHostHandle:
        self._require_live()
        namespace = os.stat("/proc/self/ns/ipc")
        segment, size, creator, created = native_ops().shm_allocator_export(
            self.name, self.device_id
        )
        return SharedHostHandle(
            segment,
            size,
            creator,
            created,
            namespace.st_dev,
            namespace.st_ino,
            tuple(self.tensor.shape),
            str(self.tensor.dtype),
        )

    @classmethod
    def attach(
        cls, handle: SharedHostHandle, *, device_id: int, name: str = ""
    ) -> "RegisteredHostRegion":
        namespace = os.stat("/proc/self/ns/ipc")
        if handle.version != 1 or (handle.ipc_device, handle.ipc_inode) != (
            namespace.st_dev,
            namespace.st_ino,
        ):
            raise ValueError("shared Host handle version or IPC namespace mismatch")
        dtype_name = handle.dtype.removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("shared Host handle has unsupported dtype")
        ctype = _ctype_for_dtype(dtype)
        if any(type(dim) is not int or dim <= 0 for dim in handle.shape):
            raise ValueError("shared Host handle has invalid shape")
        numel = prod(handle.shape)
        if numel * torch.empty((), dtype=dtype).element_size() != handle.nbytes:
            raise ValueError("shared Host handle shape and byte size mismatch")
        name = name or f"sgl-attach-{uuid4().hex}"
        host_ptr, device_ptr = native_ops().shm_allocator_attach(
            handle.segment,
            handle.nbytes,
            handle.creator,
            handle.created,
            device_id,
            name,
        )
        try:
            buffer = (ctype * numel).from_address(host_ptr)
            tensor = torch.frombuffer(buffer, dtype=dtype).view(handle.shape)
            return cls(tensor, host_ptr, device_ptr, device_id, name)
        except Exception:
            native_ops().shm_allocator_free_by_name(name, device_id)
            raise

    def close(self, *, drained: bool = False) -> None:
        if self.closed:
            return
        if not drained:
            raise RuntimeError("registered Host users must be drained before close")
        native_ops().shm_allocator_free_by_name(self.name, self.device_id)
        self.closed = True
        self.host_ptr = 0
        self.device_ptr = 0
        self.tensor = torch.empty(0, dtype=self.tensor.dtype)

    def _require_live(self) -> None:
        if self.closed:
            raise RuntimeError("registered Host region is closed")

    @property
    def nbytes(self) -> int:
        self._require_live()
        return self.tensor.numel() * self.tensor.element_size()

    def view(self, shape: Sequence[int]) -> torch.Tensor:
        self._require_live()
        view = self.tensor.view(tuple(shape))
        if view.data_ptr() != self.host_ptr:
            raise RuntimeError("registered region view is not base-aligned")
        return view


def copy_rows_to_registered_host(
    src: torch.Tensor,
    dst_region: RegisteredHostRegion,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    block_dim: int = 8,
) -> torch.Tensor:
    """Copy device rows into a registered Host region.

    ``src_index``, ``dst_index`` and ``valid_mask`` must use the device
    expected by the compiled kernel.  The destination pointer is explicitly
    passed as the NPU-visible address; the CPU process pointer never crosses
    the PD control protocol.
    """

    dst_region._require_live()
    return unidex_copy_inplace(
        src=src,
        dst=dst_region.tensor,
        src_index=src_index,
        dst_index=dst_index,
        valid_mask=valid_mask,
        src_address_ndims=1,
        dst_address_ndims=1,
        block_dim=block_dim,
        dst_ptr=dst_region.device_ptr,
    )


def copy_rows_from_registered_host(
    src_region: RegisteredHostRegion,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    block_dim: int = 8,
) -> torch.Tensor:
    """Copy rows from a registered Host region into an NPU tensor."""

    src_region._require_live()
    return unidex_copy_inplace(
        src=src_region.tensor,
        dst=dst,
        src_index=src_index,
        dst_index=dst_index,
        valid_mask=valid_mask,
        src_address_ndims=1,
        dst_address_ndims=1,
        block_dim=block_dim,
        src_ptr=src_region.device_ptr,
    )


def _infer_rows_and_block_bytes(
    tensor: torch.Tensor, address_ndims: int, name: str
) -> Tuple[int, int]:
    if address_ndims <= 0 or address_ndims >= tensor.dim():
        raise ValueError(
            f"{name}_address_ndims must be in [1, {tensor.dim() - 1}], "
            f"got {address_ndims}"
        )

    rows = prod(tensor.shape[:address_ndims])
    block_elements = prod(tensor.shape[address_ndims:])
    return int(rows), int(block_elements * tensor.element_size())


def unidex_copy_inplace(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
    dst_ptr: Optional[int] = None,
) -> torch.Tensor:
    """Copy selected logical rows from ``src`` into ``dst`` in place.

    ``src_ptr`` and ``dst_ptr`` may override the Tensor addresses with
    device-visible shared-memory addresses. The caller owns those allocations
    and must keep them alive until work on the current NPU stream completes.
    """
    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, dst_block_bytes = _infer_rows_and_block_bytes(
        dst, dst_address_ndims, "dst"
    )
    if src_block_bytes != dst_block_bytes:
        raise ValueError(
            "src and dst logical rows must have the same byte size, got "
            f"{src_block_bytes} and {dst_block_bytes}"
        )
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            f"src and dst must have the same dtype, got {src.dtype} and {dst.dtype}"
        )

    native_ops().unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        src_block_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
        dst_ptr,
    )
    return dst


_native_library = None
_native_lock = RLock()


def load_native_extension() -> bool:
    global _native_library
    configured = os.environ.get("SGLANG_NPU_FSA_EXTENSION")
    if not configured:
        return False
    path = str(Path(configured).expanduser().resolve(strict=True))
    with _native_lock:
        if _native_library is not None:
            if path != _native_library:
                raise RuntimeError("FSA extension cannot change after loading")
            return True
        torch.ops.load_library(path)
        version = torch.ops.sgl_fsa.abi_version()
        if version != 1:
            raise RuntimeError(f"unsupported standalone FSA ABI version {version}")
        _native_library = path
    return True


def native_ops():
    if not load_native_extension():
        raise RuntimeError(
            "Set SGLANG_NPU_FSA_EXTENSION to the standalone FSA binding library; "
            "the installed sgl-kernel-npu wheel is left unchanged"
        )
    return torch.ops.sgl_fsa
