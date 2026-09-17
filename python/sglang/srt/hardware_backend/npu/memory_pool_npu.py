import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu


def _init_npu_conv_state(
    conv_state_in,
    conv_state_shape,
    speculative_num_draft_tokens: Optional[int] = None,
    is_kda: bool = False,
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # Both KDA and Mamba/GDN NPU conv states use the unified
    # [layers, pool, window, channels] layout. KDA shapes arrive as
    # (window, channels) while Mamba/GDN shapes arrive as (channels, window);
    # resolve the correct axis ordering and extend the window by
    # speculative_num_draft_tokens - 1 so that verify can write all draft
    # token conv states directly into conv_states (GDN rollback scheme).
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                (conv_shape[0] if is_kda else conv_shape[1]) + extra_conv_len,
                conv_shape[1] if is_kda else conv_shape[0],
            ),
            dtype=conv_state_in.dtype,
            device=conv_state_in.device,
        )
        for conv_shape in conv_state_shape
    ]
    return conv_state


class NPUMHATokenToKVPool(MHATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        **kwargs,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.use_triton_prefix_kv_cache_store = (
            envs.SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE.get()
        )
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
            **kwargs,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.v_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )

            if self.use_fia:
                # Use per-layer Python lists to avoid torch.compile capturing
                # the entire multi-layer tensor (OOM during graph capture).
                # Each layer view: [P*ps, 1, H, D], sharing the contiguous
                # storage allocated above.
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]
                self.v_buffer = [
                    self.v_buffer[i].view(-1, 1, self.head_num, self.v_head_dim)
                    for i in range(self.layer_num)
                ]

    def _init_kv_copy_and_warmup(self):
        # implementation relies on self.data_strides / self.data_ptrs, which the
        # NPU paged buffer layout never builds.
        self._kv_copy_config = None

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
            num_rows = loc.numel()
            expected_k_numel = num_rows * self.head_num * self.head_dim
            expected_v_numel = num_rows * self.head_num * self.v_head_dim
            if (
                cache_k.numel() != expected_k_numel
                or cache_v.numel() != expected_v_numel
            ):
                raise ValueError(
                    "NPU FIA KV scatter row mismatch: "
                    f"loc_rows={num_rows}, cache_k_shape={tuple(cache_k.shape)}, "
                    f"cache_v_shape={tuple(cache_v.shape)}, "
                    f"head_num={self.head_num}, head_dim={self.head_dim}, "
                    f"v_head_dim={self.v_head_dim}."
                )

            # aclnnScatterNdUpdate on the deployed CANN rejects the otherwise
            # valid 4-D [slot, 1, head, dim] update during tiling. Flatten only
            # the singleton FIA layout axis and scatter through an equivalent
            # 3-D view; the underlying KV storage and attention layout stay
            # unchanged.
            loc_indices = loc.contiguous().view(-1, 1)
            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer.view(-1, self.head_num, self.head_dim),
                loc_indices,
                cache_k.contiguous().view(num_rows, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
                loc_indices,
                cache_v.contiguous().view(num_rows, self.head_num, self.v_head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.v_head_dim
                ),
                slot_indices=loc,
            )

    def set_kv_buffer_prefix_valid(
        self,
        layer: "RadixAttention",
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if not self.use_triton_prefix_kv_cache_store:
            return super().set_kv_buffer_prefix_valid(
                layer,
                loc_2d,
                commit_lens,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override,
            )

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got {tuple(loc_2d.shape)}")

        num_rows = loc_2d.numel()
        if (
            cache_k.numel() != num_rows * self.head_num * self.head_dim
            or cache_v.numel() != num_rows * self.head_num * self.v_head_dim
        ):
            raise ValueError(
                "dense NPU KV rows must match loc_2d size: "
                f"cache_k={tuple(cache_k.shape)}, cache_v={tuple(cache_v.shape)}, "
                f"loc_2d={tuple(loc_2d.shape)}"
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
        v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
        if loc_2d.device != k_buffer_layer.device:
            loc_2d = loc_2d.to(device=k_buffer_layer.device, non_blocking=True)
        if commit_lens.device != k_buffer_layer.device:
            commit_lens = commit_lens.to(
                device=k_buffer_layer.device, non_blocking=True
            )
        self._debug_prefix_valid_backend = "npu_triton"
        from sgl_kernel_npu.mem_cache.kv_cache_store import (
            store_kv_cache_prefix_valid_npu_triton,
        )

        store_kv_cache_prefix_valid_npu_triton(
            k_buffer_layer.view(-1, self.head_num, self.head_dim),
            v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
            cache_k.reshape(num_rows, self.head_num, self.head_dim),
            cache_v.reshape(num_rows, self.head_num, self.v_head_dim),
            loc_2d,
            commit_lens,
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.npu.synchronize()
        buf_of_layers = []
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            buf_of_layers.append([k_layer, v_layer])
        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[local_layer_id][i // chunk_size][0],
                    kv_cache_cpu[local_layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()


class NPUMHATokenToKOnlyPool(MHATokenToKOnlyPool):
    """NPU paged K-only cache used by MiniMax sparse index-only layers."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super(MHATokenToKOnlyPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            if self.use_fia:
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]

        self._finalize_allocation_log(size)

    def _get_key_buffer(self, layer_id: int):
        k_buffer = self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return k_buffer.view(self.dtype)
        return k_buffer

    def set_k_buffer(
        self,
        layer_id: int,
        loc_info,
        cache_k: torch.Tensor,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc_info)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer].view(
            -1, self.head_num, self.head_dim
        )
        loc = loc.to(device=cache_k.device, dtype=torch.int32).contiguous()
        torch_npu.npu_scatter_nd_update_(
            k_buffer_layer,
            loc.view(-1, 1),
            cache_k.contiguous().view(-1, self.head_num, self.head_dim),
        )

    def get_contiguous_buf_infos(self):
        data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        return get_tensor_size_bytes(self.k_buffer), 0


class NPUMiniMaxSparseKVPool(MiniMaxSparseKVPool):
    """MiniMax sparse wrapper backed by NPU paged MHA/index pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            main_pool_cls=NPUMHATokenToKVPool,
            index_kv_pool_cls=NPUMHATokenToKVPool,
            index_k_pool_cls=NPUMHATokenToKOnlyPool,
            **kwargs,
        )

    def get_index_k_state_buf_infos(self):
        pool = self.index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.get_key_buffer(i).data_ptr() for i in range(n)]
        data_lens = [pool.get_key_buffer(i).nbytes for i in range(n)]
        if pool.use_fia:
            item_lens = [
                pool.get_key_buffer(i)[0].nbytes * pool.page_size for i in range(n)
            ]
        else:
            item_lens = [pool.get_key_buffer(i)[0].nbytes for i in range(n)]
        return data_ptrs, data_lens, item_lens


class NPUMLATokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        index_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        kv_cache_dim: Optional[int] = None,
        index_kpool: int = 1,
        index_kpool_compress: bool = False,
        tail_extra_slots: int = 0,
        max_running_requests: Optional[int] = None,
        skip_topk_layers: Optional[List[bool]] = None,
        prefill_layer_slots: Optional[int] = None,
        decode_host_budget: Optional[int] = None,
        shared_host_group=None,
        integrated_offload: bool = False,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.tail_extra_slots = tail_extra_slots
        self._kpool_use_compress = index_kpool > 1 and index_kpool_compress
        self.skip_topk_layers = (
            list(skip_topk_layers)
            if skip_topk_layers is not None
            else [False] * layer_num
        )
        assert len(self.skip_topk_layers) == layer_num
        if prefill_layer_slots is not None and not 0 < prefill_layer_slots <= layer_num:
            raise ValueError("Prefill layer slots must fit the logical layer count")
        self.prefill_layer_slots = prefill_layer_slots
        self.integrated_offload = integrated_offload
        if integrated_offload and (prefill_layer_slots is None or decode_host_budget is None
                                   or not self._kpool_use_compress):
            raise ValueError("integrated offload requires Prefill slots, Host KV and resident KPool")
        self.prefill_offload_runtime = None
        self.decode_host_backing = None
        if decode_host_budget is not None:
            if (prefill_layer_slots is not None and not integrated_offload) or index_head_dim is None:
                raise ValueError("Decode Host KV requires a dedicated DSA Decode pool")
            from sglang.srt.hardware_backend.npu.decode_host_kv import DecodeHostKvBacking

            self.decode_host_backing = DecodeHostKvBacking(
                layers=layer_num, pages=size // page_size + 1, page_size=page_size,
                latent_dim=kv_lora_rank, rope_dim=qk_rope_head_dim,
                dtype=self.store_dtype, device_id=torch.npu.current_device(),
                host_budget=decode_host_budget, shared_group=shared_host_group,
                integrated=integrated_offload,
            )
        physical_layers = prefill_layer_slots or layer_num
        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = self._allocate_full_component(0,
                (
                    physical_layers,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_lora_rank,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = self._allocate_full_component(1,
                (
                    physical_layers,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.index_k_buffer = None
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        layer_num if self._kpool_use_compress else physical_layers,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )

        self._init_kpool_compress_tail_buffers(
            index_kpool=index_kpool,
            index_kpool_compress=index_kpool_compress,
            tail_extra_slots=tail_extra_slots,
            index_head_dim=index_head_dim,
            layer_num=layer_num,
            device=device,
            max_running_requests=max_running_requests,
        )
        self._finalize_allocation_log(size)

    def _allocate_full_component(self, component, shape, *, dtype, device):
        if self.decode_host_backing is None or self.integrated_offload:
            return torch.zeros(shape, dtype=dtype, device=device)
        return [self.decode_host_backing.layer_views(layer)[component]
                for layer in range(self.layer_num)]

    def layer_receive_lease(self, room, session_id, host_pages, device_pages, prefix=0):
        from uuid import uuid4

        from sglang.srt.hardware_backend.npu.layer_protocol import LayerReceiveLease
        from sglang.srt.hardware_backend.npu.layer_transfer import RegisteredLayerRegion
        from sglang.srt.hardware_backend.npu.prefill_offload import KvComponent, PdSourceKind

        if self.decode_host_backing is None:
            raise RuntimeError("layer receive requires registered Decode Host backing")
        if not hasattr(self, "_layer_receive_registration_id"):
            self._layer_receive_registration_id = uuid4().hex
        layers = {}
        for local_layer in range(self.layer_num):
            latent, rope = self.decode_host_backing.regions[local_layer]
            indexer = self.index_k_buffer[local_layer]
            regions = {}
            for component, tensor, owner, kind in (
                (KvComponent.LATENT_K, latent.tensor, latent, PdSourceKind.HOST_REGISTERED),
                (KvComponent.ROPE_K, None if rope is None else rope.tensor, rope, PdSourceKind.HOST_REGISTERED),
                (KvComponent.INDEXER_K, indexer, indexer, PdSourceKind.NPU_REGISTERED),
            ):
                if self._kpool_use_compress and component == KvComponent.INDEXER_K:
                    continue
                if tensor is None or tensor.numel() == 0:
                    continue
                regions[component] = RegisteredLayerRegion(
                    tensor.data_ptr(), tensor.nbytes,
                    tensor.shape[-1] * tensor.element_size() * self.page_size,
                    f"{self._layer_receive_registration_id}:{local_layer}:{component.value}",
                    kind, str(tensor.dtype), owner,
                )
            layers[self.start_layer + local_layer] = regions
        device_id = self.decode_host_backing.pool.device_id
        shared_group = getattr(self.decode_host_backing.pool, "shared_group", None)
        if shared_group is not None:
            shared_group.validate_identity((
                room, tuple(map(int, host_pages)), tuple(map(int, device_pages)), prefix,
            ))

        def complete_visibility(layer, start, end):
            torch.npu.set_device(device_id)
            torch.npu.synchronize(device_id)

        page_ids = {KvComponent.LATENT_K: tuple(map(int, host_pages)),
                    KvComponent.INDEXER_K: tuple(map(int, device_pages))}
        if self._kpool_use_compress:
            del page_ids[KvComponent.INDEXER_K]
        if self.qk_rope_head_dim:
            page_ids[KvComponent.ROPE_K] = tuple(map(int, host_pages))
        return LayerReceiveLease(
            room, session_id, self.page_size, layers,
            page_ids,
            complete_visibility, prefix=prefix,
            shared_group_id=shared_group.group_id if shared_group is not None else "",
            shared_host_writer=shared_group is None or shared_group.is_writer,
        )

    def physical_layer_index(self, layer_id: int) -> int:
        local_layer = layer_id - self.start_layer
        if not 0 <= local_layer < self.layer_num:
            raise IndexError(f"logical KV layer {layer_id} is outside this pool")
        return local_layer % (self.prefill_layer_slots or self.layer_num)

    def prepare_layer_access(self, layer_id: int) -> int:
        physical_layer = self.physical_layer_index(layer_id)
        if self.prefill_layer_slots is not None:
            if self.prefill_offload_runtime is None:
                raise RuntimeError("bounded Prefill pool has no layer runtime")
            self.prefill_offload_runtime.prepare_layer(layer_id)
        return physical_layer

    def raw_layer_components(self, layer_id: int):
        physical_layer = self.physical_layer_index(layer_id)
        return (
            self.k_buffer[physical_layer],
            self.v_buffer[physical_layer],
            None
            if self.index_k_buffer is None
            else self.index_k_buffer[layer_id - self.start_layer if self._kpool_use_compress else physical_layer],
        )

    def _should_allocate_index_layer(self, local_layer_idx: int) -> bool:
        return not self.skip_topk_layers[local_layer_idx]

    def _init_kpool_compress_tail_buffers(
        self,
        index_kpool: int,
        index_kpool_compress: bool,
        tail_extra_slots: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        max_running_requests: Optional[int],
    ) -> None:
        if not self._kpool_use_compress or index_head_dim is None:
            self._compress_tail_k = None
            self._compress_tail_score = None
            return

        assert (
            max_running_requests is not None
        ), "NPUMLATokenToKVPool with kpool compress requires max_running_requests"
        req_pool_size = max_running_requests + 1
        tail_dtype = torch.bfloat16
        tail_width = index_kpool + tail_extra_slots
        self._compress_tail_k: Optional[List[torch.Tensor]] = [
            torch.zeros(
                req_pool_size if self._should_allocate_index_layer(i) else 0,
                tail_width,
                index_head_dim,
                dtype=tail_dtype,
                device=device,
            )
            for i in range(layer_num)
        ]
        self._compress_tail_score: Optional[List[torch.Tensor]] = [
            torch.zeros(
                req_pool_size if self._should_allocate_index_layer(i) else 0,
                tail_width,
                index_head_dim,
                dtype=tail_dtype,
                device=device,
            )
            for i in range(layer_num)
        ]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_buffer[layer_id - self.start_layer]

    def get_compress_tail_buffers(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            self._kpool_use_compress
        ), "get_compress_tail_buffers called when kpool compress is disabled"
        idx = layer_id - self.start_layer
        return (
            self._compress_tail_k[idx],
            self._compress_tail_score[idx],
        )

    def kpool_decode_update_index_cache(
        self,
        layer_id: int,
        key: torch.Tensor,
        slot_score: torch.Tensor,
        ape: torch.Tensor,
        block_tables: torch.Tensor,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        round_scale: bool = False,
    ) -> None:
        # from sglang.srt.layers.attention.dsa.kpool_bf16_index import (
        #     kpool_decode_update_and_maybe_write_cache_bf16,
        # )
        from sglang.srt.layers.attention.dsa.kpool_index_npu import (
            kpool_decode_update_and_maybe_write_cache_bf16,
        )

        assert (
            self._kpool_use_compress
        ), "kpool_decode_update_index_cache called when kpool compress is disabled"
        idx = layer_id - self.start_layer
        buf = self.get_index_k_with_scale_buffer(layer_id)
        kpool_decode_update_and_maybe_write_cache_bf16(
            # pool=self,
            buf=buf,
            tail_k=self._compress_tail_k[idx],
            tail_score=self._compress_tail_score[idx],
            key=key,
            slot_score=slot_score,
            ape=ape,
            block_tables=block_tables,
            req_pool_indices=req_pool_indices,
            positions=positions,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            pool_size=self.index_kpool,
            slots_per_page=self.page_size,
        )

    def set_compress_tail_for_request(
        self,
        layer_id: int,
        req_pool_idx: torch.Tensor,
        key_tail: torch.Tensor,
        score_tail: torch.Tensor,
        n_remain: int,
        dst_logical_start: int,
    ) -> None:
        assert (
            self._kpool_use_compress
        ), "set_compress_tail_for_request called when kpool compress is disabled"
        idx = layer_id - self.start_layer
        if n_remain > 0:
            slots = (
                torch.arange(n_remain, device=key_tail.device, dtype=torch.long)
                + int(dst_logical_start)
            ) % self._compress_tail_k[idx].shape[1]
            self._compress_tail_k[idx][req_pool_idx, slots] = key_tail
            self._compress_tail_score[idx][req_pool_idx, slots] = score_tail

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = 0
        for k_cache in self.k_buffer:
            if self.decode_host_backing is None or self.integrated_offload:
                kv_size_bytes += get_tensor_size_bytes(k_cache)
        for v_cache in self.v_buffer:
            if self.decode_host_backing is None or self.integrated_offload:
                kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.integrated_offload and not getattr(self.prefill_offload_runtime, "active", False):
            return self.decode_host_backing.layer_views(layer_id - self.start_layer)
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[self.prepare_layer_access(layer_id)],
            self.v_buffer[self.prepare_layer_access(layer_id)],
        )

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        layers = [self.raw_layer_components(self.start_layer + i)[2] for i in range(self.layer_num)]
        data_ptrs = [layers[i].data_ptr() for i in range(self.layer_num)]
        data_lens = [layers[i].nbytes for i in range(self.layer_num)]
        item_lens = [layers[i][0].nbytes for i in range(self.layer_num)]
        return data_ptrs, data_lens, item_lens

    @property
    def kpool_use_compress(self) -> bool:
        # PD disaggregation's shared DSA-tail helpers (append_dsa_tail,
        # get_dsa_tail_state_indices) read this flag off either
        # DSATokenToKVPool or this pool; the internal name stays private.
        return self._kpool_use_compress

    def get_compress_tail_buf_infos(self):
        """Buffer infos for per-request DSA kpool compress-tail rows."""
        if not self._kpool_use_compress:
            return [], [], []
        # Keys first, then scores — keep zero-row (skip-topk) entries so layer
        # offsets stay aligned across PD peers; item_len=0 makes transfer
        # backends skip them.
        tail_buffers = list(self._compress_tail_k) + list(self._compress_tail_score)
        data_ptrs = [buf.data_ptr() for buf in tail_buffers]
        data_lens = [buf.nbytes for buf in tail_buffers]
        item_lens = [buf[0].nbytes if buf.shape[0] > 0 else 0 for buf in tail_buffers]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.integrated_offload:
            return self.get_kv_buffer(layer_id)[0]
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.k_buffer[self.prepare_layer_access(layer_id)].view(self.dtype)
        return self.k_buffer[self.prepare_layer_access(layer_id)]

    def get_value_buffer(self, layer_id: int):
        if self.integrated_offload:
            return self.get_kv_buffer(layer_id)[1]
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.v_buffer[self.prepare_layer_access(layer_id)].view(self.dtype)
        return self.v_buffer[self.prepare_layer_access(layer_id)]

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.index_k_buffer[layer_id - self.start_layer if self._kpool_use_compress else self.prepare_layer_access(layer_id)].view(self.dtype)
        return self.index_k_buffer[layer_id - self.start_layer if self._kpool_use_compress else self.prepare_layer_access(layer_id)]

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        sections = [("k", self.k_buffer), ("v", self.v_buffer)]
        if self.index_head_dim is not None:
            sections.append(("index_k", self.index_k_buffer))
        kv_data_ptrs, kv_data_lens, kv_item_lens = [], [], []
        skipped_empty = []
        for name, bufs in sections:
            for i in range(self.layer_num):
                physical = i % (self.prefill_layer_slots or self.layer_num)
                if name == "index_k" and self._kpool_use_compress:
                    physical = i
                buf = bufs[physical]
                if buf.numel() == 0:
                    # Zero-size placeholder (e.g. v/rope cache when
                    # qk_rope_head_dim == 0); skipping keeps the registration
                    # free of null ptrs and PD-symmetric.
                    skipped_empty.append((name, i))
                    continue
                kv_data_ptrs.append(buf.data_ptr())
                kv_data_lens.append(buf.nbytes)
                kv_item_lens.append(buf[0].nbytes)
        if skipped_empty:
            logger.info(
                f"[PD] {type(self).__name__} skips zero-size KV buffers: {skipped_empty}"
            )
        zero_bufs = [
            (name, i)
            for name, bufs in sections
            for i in range(self.layer_num)
            if bufs[i % len(bufs)].numel() != 0 and bufs[i % len(bufs)].data_ptr() == 0
        ]
        if zero_bufs:
            logger.error(
                f"[PD] {type(self).__name__} has non-empty null-ptr buffers: {zero_bufs}"
            )
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        runtime = self.prefill_offload_runtime
        async_save = self.integrated_offload and getattr(runtime, "active", False) and getattr(runtime, "d2h_async", False)
        if async_save:
            runtime.validate_write(layer_id)
        elif self.decode_host_backing is not None:
            self.decode_host_backing.write_current(layer_id - self.start_layer, loc, cache_k, cache_v)
            if not self.integrated_offload or not getattr(self.prefill_offload_runtime, "active", False):
                return

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[self.prepare_layer_access(layer_id)].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        if self.qk_rope_head_dim > 0:
            torch_npu.npu_scatter_nd_update_(
                self.v_buffer[self.prepare_layer_access(layer_id)].view(
                    -1, 1, self.qk_rope_head_dim
                ),
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.qk_rope_head_dim),
            )

        if async_save:
            runtime.on_kv_written(layer_id, loc)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[layer_id - self.start_layer if self._kpool_use_compress else self.prepare_layer_access(layer_id)].view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf, ik_buf/None]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.npu.synchronize()
        buf_of_layers = []
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            ik_layer = (
                self.index_k_buffer[local_layer_id].view(-1, 1, self.index_head_dim)
                if has_ik
                else None
            )
            buf_of_layers.append([k_layer, v_layer, ik_layer])

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            ik_layer = (
                self.index_k_buffer[local_layer_id].view(-1, 1, self.index_head_dim)
                if has_ik
                else None
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                k_cpu, v_cpu = chunk[0], chunk[1]
                assert k_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
                if has_ik:
                    ik_cpu = chunk[2]
                    ik_layer[chunk_indices] = ik_cpu.to(
                        ik_layer.device, non_blocking=True
                    )
        torch.npu.synchronize()
