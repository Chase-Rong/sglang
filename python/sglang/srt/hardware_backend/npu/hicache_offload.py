"""HiCache ownership and device-to-Host page views for sparse offload."""

from collections import deque
from threading import RLock

import torch


class OffloadPageBinding:
    def __init__(self, host_pool, device, device_pages):
        self.host_pool = host_pool
        self.page_size = host_pool.page_size
        self.device = torch.device(device)
        self.device_to_host_cpu = torch.zeros(device_pages, dtype=torch.int64)
        self.device_to_host = self.device_to_host_cpu.to(self.device)
        self.references = torch.zeros(host_pool.page_num, dtype=torch.int64)
        self.cache_references = torch.zeros_like(self.references)
        self.retired = deque()
        self.ready_event = None
        self.reader_events = deque()
        self.evict_host = None
        shared = getattr(host_pool.backing.pool, "shared_group", None)
        self.shared_group = (
            shared if shared is not None and shared.group.world_size > 1 else None
        )
        self.lock = RLock()
        self.indexer_mapping = None
        if getattr(host_pool.device_pool, "_kpool_use_compress", False):
            from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import (
                IndexerPageMapping,
            )

            self.indexer_mapping = IndexerPageMapping(
                self.page_size, host_pool.device_pool.index_kpool, device_pages
            )
        host_pool.binding = self

    def indexer_page_locations(self, device_pages):
        if self.indexer_mapping is None:
            raise RuntimeError("This HiCache pool has no compressed Indexer mapping")
        return self.indexer_mapping.indexer_page_locations(device_pages)

    def _rows(self, pages):
        return (pages[:, None] * self.page_size + torch.arange(self.page_size)).reshape(
            -1
        )

    def _complete_pages(self, rows):
        if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "HiCache ownership requires a one-dimensional integer row tensor"
            )
        values = rows.detach().to(device="cpu", dtype=torch.int64)
        if values.numel() % self.page_size:
            raise ValueError("HiCache ownership requires complete pages")
        grouped = values.view(-1, self.page_size)
        starts = grouped[:, 0]
        if (starts % self.page_size).any() or not torch.equal(
            grouped, starts[:, None] + torch.arange(self.page_size)
        ):
            raise ValueError("HiCache ownership requires aligned contiguous pages")
        return starts // self.page_size

    def _validate_device_pages(self, pages):
        if pages.numel() and (
            pages.min() <= 0 or pages.max() >= self.device_to_host_cpu.numel()
        ):
            raise ValueError("Offload device page is outside the allocated pool")

    def _update(self, device_pages, host_pages):
        self.device_to_host_cpu[device_pages] = host_pages
        self.device_to_host.index_copy_(
            0, device_pages.to(self.device), host_pages.to(self.device)
        )

    def reap(self):
        with self.lock:
            while self.reader_events and self.reader_events[0].query():
                self.reader_events.popleft()
            while (
                self.shared_group is None
                and self.retired
                and self.retired[0][0].query()
            ):
                _, pages = self.retired.popleft()
                self.host_pool.free_unreferenced(self._rows(pages))

    def record_reader(self, event):
        with self.lock:
            self.reader_events.append(event)

    def reclaim_for_capacity(self, count):
        with self.lock:
            self.reap()
            if self.host_pool.available_size() >= count or not self.retired:
                return
            pending = list(self.retired)
            for event, _ in pending:
                event.synchronize()
            if self.shared_group is not None:
                self.shared_group.group.barrier()
            for _, pages in pending:
                self.host_pool.free_unreferenced(self._rows(pages))
            self.retired.clear()

    def ensure_pages(self, device_pages):
        with self.lock:
            self.reap()
            pages = torch.unique(
                device_pages.detach().to(device="cpu", dtype=torch.int64)
            )
            self._validate_device_pages(pages)
            missing = pages[self.device_to_host_cpu[pages] == 0]
            if not missing.numel():
                return True
            count = missing.numel() * self.page_size
            self.reclaim_for_capacity(count)
            if self.host_pool.available_size() < count and self.evict_host is not None:
                self.evict_host(count - self.host_pool.available_size())
                self.reclaim_for_capacity(count)
            rows = self.host_pool.alloc(count)
            if rows is None:
                return False
            host_pages = rows[:: self.page_size] // self.page_size
            self.references.index_add_(0, host_pages, torch.ones_like(host_pages))
            self._update(missing, host_pages)
            return True

    def host_rows(self, device_rows):
        rows = device_rows.to(dtype=torch.int64)
        pages = torch.div(rows, self.page_size, rounding_mode="floor")
        valid = (rows > 0) & (pages < self.device_to_host.numel())
        mapped = self.device_to_host[pages.clamp(0, self.device_to_host.numel() - 1)]
        return torch.where(
            valid & (mapped > 0),
            mapped * self.page_size + rows.remainder(self.page_size),
            0,
        )

    def host_table(self, device_table):
        pages = device_table.to(dtype=torch.int64)
        valid = (pages > 0) & (pages < self.device_to_host.numel())
        mapped = self.device_to_host[pages.clamp(0, self.device_to_host.numel() - 1)]
        return torch.where(valid, mapped, 0).to(device_table.dtype)

    def retain_device_rows(self, device_rows):
        with self.lock:
            pages = self._complete_pages(device_rows)
            rows = device_rows.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            self._validate_device_pages(pages)
            host_pages = self.device_to_host_cpu[pages]
            if (host_pages == 0).any():
                raise RuntimeError("Cannot cache offload pages without Host ownership")
            self.references.index_add_(0, host_pages, torch.ones_like(host_pages))
            self.cache_references.index_add_(0, host_pages, torch.ones_like(host_pages))
            return self.device_to_host_cpu[
                rows // self.page_size
            ] * self.page_size + rows.remainder(self.page_size)

    def _release(self, pages):
        if not pages.numel():
            return
        unique_pages, counts = torch.unique(pages, return_counts=True)
        if (self.references[unique_pages] < counts).any():
            raise RuntimeError("Offload Host page reference underflow")
        self.references[unique_pages] -= counts
        unused = unique_pages[self.references[unique_pages] == 0]
        if unused.numel():
            if self.ready_event is not None:
                torch.npu.current_stream().wait_event(self.ready_event)
            for reader in self.reader_events:
                torch.npu.current_stream().wait_event(reader)
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            self.retired.append((event, unused))

    def release_cached_rows(self, rows):
        with self.lock:
            pages = self._complete_pages(rows)
            if pages.numel() and (
                pages.min() <= 0 or pages.max() >= self.references.numel()
            ):
                raise ValueError("Cannot release reserved or invalid Host pages")
            unique_pages, counts = torch.unique(pages, return_counts=True)
            if (self.cache_references[unique_pages] < counts).any():
                raise RuntimeError("Offload Host cache reference underflow")
            self._release(pages)
            self.cache_references[unique_pages] -= counts
            return rows.numel()

    def release_device_pages(self, pages):
        with self.lock:
            pages = torch.unique(pages.detach().to(device="cpu", dtype=torch.int64))
            self._validate_device_pages(pages)
            for reader in self.reader_events:
                torch.npu.current_stream().wait_event(reader)
            if self.indexer_mapping is not None:
                self.indexer_mapping.release_device_pages(pages)
            host_pages = self.device_to_host_cpu[pages]
            owned = host_pages > 0
            self._update(pages, torch.zeros_like(pages))
            self._release(host_pages[owned])

    def restore_mapping(self, host_rows, device_rows):
        with self.lock:
            host_pages = self._complete_pages(host_rows)
            device_pages = self._complete_pages(device_rows)
            host = host_rows.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            if host_pages.numel() != device_pages.numel():
                raise ValueError("HiCache restore requires equally sized whole pages")
            if device_pages.unique().numel() != device_pages.numel():
                raise ValueError("HiCache restore has duplicate device pages")
            self._validate_device_pages(device_pages)
            if (host_pages <= 0).any() or (host_pages >= self.references.numel()).any():
                raise ValueError("HiCache restore names invalid Host pages")
            if (
                not self.host_pool.slot_used[host].all()
                or (self.references[host_pages] == 0).any()
            ):
                raise RuntimeError("HiCache restore names unowned Host memory")
            old_pages = self.device_to_host_cpu[device_pages]
            changed = old_pages != host_pages
            self.references.index_add_(
                0, host_pages[changed], torch.ones_like(host_pages[changed])
            )
            self._release(old_pages[changed & (old_pages > 0)])
            self._update(device_pages, host_pages)

    def clear(self):
        with self.lock:
            if self.references.any() or self.retired:
                torch.npu.synchronize()
                if self.shared_group is not None:
                    self.shared_group.group.barrier()
            self.retired.clear()
            self.reader_events.clear()
            self.references.zero_()
            self.cache_references.zero_()
            self.device_to_host_cpu.zero_()
            self.device_to_host.zero_()
            self.ready_event = None
            if self.indexer_mapping is not None:
                self.indexer_mapping.clear()


def get_offload_backing(kv_pool):
    pool = getattr(kv_pool, "full_kv_pool", kv_pool)
    return (
        getattr(pool, "decode_host_backing", None)
        if getattr(pool, "integrated_offload", False)
        else None
    )
