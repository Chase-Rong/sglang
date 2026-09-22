"""Host copies of donated NPU Mamba/KDA checkpoints, using active-pool layout.

KDA conv is [layers, size + 1, window, channels] and temporal state is
[layers, size + 1, heads, value_dim, key_dim]. Host checkpoint slots are
[host_size, layers, 1, ...]. Copy both without transposing or converting dtype.
Device slot zero is padding; Host slot zero is an ordinary allocated slot.

The caller supplies a completed checkpoint at the tree's donated depth, holds
its slot until the transfer event completes, and waits before using a restored
slot. This adapter cannot infer checkpoint depth from slot IDs. ReplaySSM rings
and speculative rollback state are outside this checkpoint format.
"""

import torch

from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost


class NpuMambaPoolHost(MambaPoolHost):
    def __init__(
        self,
        device_pool,
        host_to_device_ratio,
        host_size,
        pin_memory=True,
        device="cpu",
        allocator_type="default",
        layout="page_first",
    ):
        if layout not in ("page_first", "page_first_direct"):
            raise ValueError("NPU Mamba Host checkpoints require page_first layout")
        if torch.device(device).type != "cpu":
            raise ValueError("NPU Mamba Host checkpoints must reside on CPU")
        if getattr(device_pool, "enable_linear_replayssm", False) or getattr(
            device_pool, "enable_linear_replayssm_spec", False
        ):
            raise NotImplementedError(
                "NPU HiCache does not restore ReplaySSM ring state"
            )
        self._validate_device_buffers(device_pool)
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            layout=layout,
        )
        self.layer_num = self.num_mamba_layers

    @staticmethod
    def _validate_device_buffers(device_pool):
        cache = device_pool.mamba_cache
        if not cache.conv or device_pool.num_mamba_layers <= 0 or device_pool.size <= 0:
            raise ValueError("NPU Mamba checkpoint pool needs state buffers and layers")
        expected_prefix = (device_pool.num_mamba_layers, device_pool.size + 1)
        for buffer in (cache.temporal, *cache.conv):
            if buffer.ndim < 3 or tuple(buffer.shape[:2]) != expected_prefix:
                raise ValueError(
                    "NPU Mamba state must use [layers, size + 1, ...] layout"
                )
        if any(buffer.dtype != cache.conv[0].dtype for buffer in cache.conv):
            raise ValueError("Mamba Host pool requires a common conv-state dtype")

    def _init_write_back_staging_buffers(self):
        super()._init_write_back_staging_buffers()
        self.can_use_write_back_jit = False

    def _checkpoint_pairs(self, host_indices, device_indices, *, to_host):
        values = []
        for indices in (host_indices, device_indices):
            if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("Mamba checkpoint indices must be integer vectors")
            if indices.device.type != "cpu":
                raise ValueError(
                    "Stage Mamba indices on CPU once before layer transfers"
                )
            values.append(indices.detach().to(device="cpu", dtype=torch.int64).tolist())
        host_slots, device_slots = values
        if len(host_slots) != len(device_slots):
            raise ValueError("Mamba checkpoint copy requires equal slot counts")
        destinations = host_slots if to_host else device_slots
        if len(set(destinations)) != len(destinations):
            raise ValueError(
                "Mamba checkpoint copy requires distinct destination slots"
            )
        if any(index < 0 or index >= self.size for index in host_slots):
            raise ValueError("Mamba Host checkpoint slot is out of range")
        if any(index <= 0 or index > self.device_pool.size for index in device_slots):
            raise ValueError("Mamba device checkpoint slot is reserved or out of range")
        return tuple(zip(host_slots, device_slots, strict=True))

    def _buffer_pairs(self, device_pool):
        if getattr(self, "_destroyed", False):
            raise RuntimeError("NPU Mamba Host checkpoint pool is destroyed")
        if device_pool is not self.device_pool:
            raise ValueError("Mamba checkpoint copy uses the wrong device pool")
        host_buffers = (self.temporal_buffer, *self.conv_buffer)
        device_buffers = (
            device_pool.mamba_cache.temporal,
            *device_pool.mamba_cache.conv,
        )
        if len(host_buffers) != len(device_buffers):
            raise ValueError("Mamba Host/device component counts differ")
        pairs = tuple(zip(host_buffers, device_buffers, strict=True))
        for host_buffer, device_buffer in pairs:
            expected = (self.size, self.num_mamba_layers, 1, *device_buffer.shape[2:])
            if (
                host_buffer.shape != expected
                or host_buffer.dtype != device_buffer.dtype
            ):
                raise ValueError("Mamba Host/device checkpoint shapes or dtypes differ")
        return pairs

    @staticmethod
    def _copy_layer(buffers, pairs, layer_id, *, to_host):
        for host_buffer, device_buffer in buffers:
            for host_slot, device_slot in pairs:
                host_value = host_buffer[host_slot, layer_id, 0]
                device_value = device_buffer[layer_id, device_slot]
                if not host_value.numel():
                    continue
                destination, source = (
                    (host_value, device_value)
                    if to_host
                    else (device_value, host_value)
                )
                destination.copy_(source, non_blocking=True)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel_ascend",
        *,
        is_draft=False,
    ):
        self.prepare_load_to_device(
            device_pool, host_indices, device_indices, io_backend, is_draft=is_draft
        )(layer_id)

    def prepare_load_to_device(
        self, device_pool, host_indices, device_indices, io_backend, *, is_draft=False
    ):
        """Capture immutable CPU slot pairs once per L2 transfer, never by identity."""
        if io_backend != "kernel_ascend" or is_draft:
            raise ValueError(
                "NPU Mamba Host checkpoints require the Ascend main-model path"
            )
        buffers = self._buffer_pairs(device_pool)
        pairs = self._checkpoint_pairs(host_indices, device_indices, to_host=False)

        def load_layer(layer_id):
            if getattr(self, "_destroyed", False):
                raise RuntimeError("NPU Mamba Host checkpoint pool is destroyed")
            if not 0 <= layer_id < self.num_mamba_layers:
                raise ValueError("Mamba checkpoint layer index is out of range")
            self._copy_layer(buffers, pairs, layer_id, to_host=False)

        return load_layer

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel_ascend"
    ):
        if io_backend != "kernel_ascend":
            raise ValueError("NPU Mamba Host checkpoints require Ascend copies")
        buffers = self._buffer_pairs(device_pool)
        pairs = self._checkpoint_pairs(host_indices, device_indices, to_host=True)
        for layer_id in range(self.num_mamba_layers):
            self._copy_layer(buffers, pairs, layer_id, to_host=True)
