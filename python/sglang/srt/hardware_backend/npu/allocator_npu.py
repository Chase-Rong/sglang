from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class NPUPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.roundup = page_size - 1
        from sglang.srt.hardware_backend.npu.hicache_offload import get_offload_backing

        self.offload_backing = get_offload_backing(kvcache)
        if self.offload_backing is not None:
            self.offload_backing.bind_hicache(getattr(kvcache, "full_kv_pool", kvcache))

    def _reserve_host_pages(self, pages):
        backing = getattr(self, "offload_backing", None)
        return backing is None or backing.page_binding.ensure_pages(pages)

    def alloc(self, need_size):
        count = need_size // self.page_size
        if count > len(self.free_pages):
            self.merge_and_sort_free()
        if count > len(self.free_pages) or not self._reserve_host_pages(
            self.free_pages[:count]
        ):
            return None
        return super().alloc(need_size)

    def _release_page_ids(self, *page_ids):
        backing = getattr(self, "offload_backing", None)
        if backing is not None and page_ids:
            backing.page_binding.release_device_pages(torch.cat(page_ids))
        return super()._release_page_ids(*page_ids)

    def alloc_for_host_restore(self, host_indices):
        if self.offload_backing is None:
            raise RuntimeError("Host restore allocation requires integrated offload")
        device_indices = super().alloc(host_indices.numel())
        if device_indices is None:
            return None
        try:
            self.offload_backing.page_binding.restore_mapping(
                host_indices, device_indices
            )
        except Exception:
            super()._release_page_ids(
                device_indices[:: self.page_size] // self.page_size
            )
            raise
        return device_indices

    def clear(self):
        backing = getattr(self, "offload_backing", None)
        if backing is not None:
            backing.hicache_pool.clear()
        super().clear()

    def evict_to_free_tokens(self, tree_cache, num_tokens):
        super().evict_to_free_tokens(tree_cache, num_tokens)
        backing = getattr(self, "offload_backing", None)
        if backing is not None:
            binding = backing.page_binding
            binding.reclaim_for_capacity(num_tokens)
            shortfall = num_tokens - backing.hicache_pool.available_size()
            if shortfall > 0:
                tree_cache.evict_host(shortfall)
                binding.reclaim_for_capacity(num_tokens)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if num_new_pages is None:
            num_new_pages_tensor = (
                (seq_lens + self.roundup) // self.page_size
                - (prefix_lens + self.roundup) // self.page_size
            ).sum()
            num_new_pages_item = num_new_pages_tensor.item()
        else:
            num_new_pages_item = num_new_pages
        if num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return None

        if not self._reserve_host_pages(self.free_pages[:num_new_pages_item]):
            return None

        if num_new_pages_item < 200:
            from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel

            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            max_num_extend_tokens = next_power_of_2(extend_num_tokens)
            bs = prefix_lens.shape[0]
            alloc_extend_kernel[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                next_power_of_2(bs),
                self.page_size,
                max_num_extend_tokens,
            )

        else:
            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
            alloc_extend_naive(
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.page_size,
                self.device,
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )

        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        if not self._reserve_host_pages(self.free_pages[:num_new_pages]):
            return None

        need_new_pages = (seq_lens % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = end_new_pages - need_new_pages
        if num_new_pages == 0:
            out_indices = last_loc + 1
        else:
            start_new_pages = start_new_pages.clamp(max=num_new_pages - 1)
            out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[
                start_new_pages
            ] * self.page_size * need_new_pages

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices.int()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.free_group is None:
            device = free_index.device
            free_page_indices = torch.unique(free_index.cpu() // self.page_size)
            self._release_page_ids(free_page_indices.to(device))
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)
