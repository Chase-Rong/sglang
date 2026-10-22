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
        from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import (
            get_offload_backing,
        )

        self.offload_backing = get_offload_backing(kvcache)
        if self.offload_backing is not None:
            pool = getattr(kvcache, "full_kv_pool", kvcache)
            if page_size != pool.page_size or size != pool.size:
                raise ValueError(
                    "Offload allocator must use the pool's physical token/page capacity"
                )
            self.offload_backing.bind_hicache(pool)

    def _reserve_host_pages(self, pages):
        if pages.numel() == 0:
            return True
        backing = getattr(self, "offload_backing", None)
        return backing is None or backing.page_binding.ensure_pages(pages)

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        if self.offload_backing is None:
            return super().create_prefill_budget(
                tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
            )
        from sglang.srt.mem_cache.prefill_budget import PrefillBudget

        class OffloadPrefillBudget(PrefillBudget):
            def _available_and_evictable(self):
                device_tokens = super()._available_and_evictable()
                return min(
                    device_tokens,
                    self.allocator._schedulable_host_tokens(
                        self.tree_cache, device_tokens
                    ),
                )

        return OffloadPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def _schedulable_host_tokens(self, tree_cache, device_tokens):
        """Count free, retired, and fully evictable cache-only Host pages."""
        backing = self.offload_backing
        binding = backing.page_binding
        with binding.lock:
            capacity = backing.hicache_pool.available_size() + sum(
                pages.numel() * self.page_size for _, pages in binding.retired
            )
            if capacity >= device_tokens:
                return capacity
            core = getattr(tree_cache, "tree_core", None)
            root = getattr(core, "root_node", None)
            if root is None:
                return capacity
            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

            stack = [(root, False)]
            removable = {}
            candidate_pages = []
            while stack:
                node, visited = stack.pop()
                if not visited:
                    stack.append((node, True))
                    stack.extend((child, False) for child in node.children.values())
                    continue
                full = node.component_data[ComponentType.FULL]
                can_remove = (
                    node is not root
                    and full.value is None
                    and full.host_value is not None
                    and not any(
                        data.lock_ref or data.host_lock_ref
                        for data in node.component_data
                    )
                    and node.write_through_pending_id is None
                    and node.load_back_pending_id is None
                    and all(removable[id(child)] for child in node.children.values())
                )
                removable[id(node)] = can_remove
                if can_remove:
                    candidate_pages.append(
                        torch.unique(
                            full.host_value.to(device="cpu", dtype=torch.int64)
                            // self.page_size
                        )
                    )
            if candidate_pages:
                counts = torch.bincount(
                    torch.cat(candidate_pages), minlength=binding.references.numel()
                )
                reclaimable = (
                    (counts > 0)
                    & (binding.references == counts)
                    & (binding.cache_references == counts)
                )
                reclaimable[0] = False
                capacity += int(reclaimable.sum()) * self.page_size
            return capacity

    def alloc(self, need_size):
        if self.offload_backing is not None and (
            need_size < 0 or need_size % self.page_size
        ):
            raise ValueError(
                "Offload allocation size must be nonnegative and page aligned"
            )
        count = need_size // self.page_size
        if count > len(self.free_pages):
            self.merge_and_sort_free()
        if count > len(self.free_pages) or not self._reserve_host_pages(
            self.free_pages[:count]
        ):
            return None
        reserved_pages = (
            self.free_pages[:count].clone()
            if self.offload_backing is not None
            else None
        )
        free_pages_before = self.free_pages
        try:
            return super().alloc(need_size)
        except BaseException:
            if reserved_pages is not None:
                self.free_pages = free_pages_before
                self.offload_backing.page_binding.release_device_pages(reserved_pages)
            raise

    def get_extend_allocation_demand(
        self, prefix_lens_cpu, seq_lens_cpu, *, conservative_num_tokens, shard_size
    ):
        if self.offload_backing is None:
            return super().get_extend_allocation_demand(
                prefix_lens_cpu,
                seq_lens_cpu,
                conservative_num_tokens=conservative_num_tokens,
                shard_size=shard_size,
            )
        if shard_size != 1:
            raise ValueError("NPU offload requires unsharded physical KV pages")
        count = self._physical_extend_page_count(prefix_lens_cpu, seq_lens_cpu)
        return count * self.page_size

    def _physical_extend_page_count(self, prefix_lens_cpu, seq_lens_cpu):
        if (
            prefix_lens_cpu.shape != seq_lens_cpu.shape
            or prefix_lens_cpu.ndim != 1
            or prefix_lens_cpu.device.type != "cpu"
            or seq_lens_cpu.device.type != "cpu"
            or (prefix_lens_cpu < 0).any()
            or (seq_lens_cpu < prefix_lens_cpu).any()
        ):
            raise ValueError("Offload extend requires ordered CPU request lengths")
        return int(
            (
                (seq_lens_cpu + self.roundup) // self.page_size
                - (prefix_lens_cpu + self.roundup) // self.page_size
            ).sum()
        )

    def _release_page_ids(self, *page_ids):
        backing = getattr(self, "offload_backing", None)
        if backing is not None and page_ids:
            backing.page_binding.release_device_pages(torch.cat(page_ids))
        elif page_ids:
            from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import (
                get_indexer_mapping,
            )

            mapping = get_indexer_mapping(self)
            if mapping is not None:
                mapping.release_device_pages(torch.cat(page_ids))
        return super()._release_page_ids(*page_ids)

    def free_group_end(self):
        if self.offload_backing is None:
            return super().free_group_end()
        token_indices = self.free_group
        page_ids = list(self.free_page_ids_group)
        if token_indices:
            page_ids.append(torch.cat(token_indices) // self.page_size)
        if page_ids:
            pages = torch.unique(
                torch.cat(page_ids).to(device="cpu", dtype=torch.int64)
            )
            self._release_page_ids(pages.to(self.device))
        self.free_group = None
        self.free_page_ids_group = []
        if self.debug_mode:
            self._debug_check_no_duplicate_pages()

    def alloc_for_host_restore(self, host_indices):
        if self.offload_backing is None:
            raise RuntimeError("Host restore allocation requires integrated offload")
        if host_indices.numel() % self.page_size:
            raise ValueError("Host restore requires whole pages")
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
        else:
            from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import (
                get_indexer_mapping,
            )

            mapping = get_indexer_mapping(self)
            if mapping is not None:
                mapping.clear()
        super().clear()

    def evict_to_free_tokens(self, tree_cache, num_tokens):
        super().evict_to_free_tokens(tree_cache, num_tokens)
        backing = getattr(self, "offload_backing", None)
        if backing is not None:
            binding = backing.page_binding
            binding.reclaim_for_capacity(num_tokens)
            shortfall = num_tokens - backing.hicache_pool.available_size()
            if shortfall > 0 and tree_cache is not None:
                tree_cache.evict_host(shortfall)
                binding.reclaim_for_capacity(num_tokens)

    def check_decode_capacity(
        self, *, num_tokens, tree_cache, requests=None, spec_algorithm=None
    ):
        device_fits = super().check_decode_capacity(
            num_tokens=num_tokens,
            tree_cache=tree_cache,
            requests=requests,
            spec_algorithm=spec_algorithm,
        )
        if self.offload_backing is None:
            return device_fits
        required = (num_tokens + self.roundup) // self.page_size * self.page_size
        return (
            device_fits
            and self.offload_backing.hicache_pool.available_size() >= required
        )

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

        if self.offload_backing is not None:
            num_new_pages_item = self._physical_extend_page_count(
                prefix_lens_cpu, seq_lens_cpu
            )
            if int((seq_lens_cpu - prefix_lens_cpu).sum()) != extend_num_tokens:
                raise ValueError(
                    "Offload extend token count differs from request lengths"
                )
            if num_new_pages is not None and num_new_pages != num_new_pages_item:
                raise ValueError(
                    "Offload extend page count must count physical KV pages"
                )
        elif num_new_pages is None:
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

        reserved_pages = self.free_pages[:num_new_pages_item]
        try:
            if num_new_pages_item < 200 and not (
                self.offload_backing is not None and num_new_pages_item == 0
            ):
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

        except BaseException:
            if self.offload_backing is not None:
                self.offload_backing.page_binding.release_device_pages(reserved_pages)
            raise

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

        if num_new_pages == 0:
            out_indices = last_loc + 1
            if self.debug_mode:
                assert len(torch.unique(out_indices)) == len(out_indices)
            return out_indices.int()

        need_new_pages = (seq_lens % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = (end_new_pages - need_new_pages).clamp(max=num_new_pages - 1)
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
