"""Logical-request-derived compressed KPool locations for HiCache.

Attach an IndexerPageMapping as OffloadPageBinding.indexer_mapping and delegate
indexer_page_locations to it. The binding must release these entries before
reusing device pages, after transfer/read completion, and clear them on reset.
Restore must register the resident prefix plus allocated rows before enqueueing
the sidecar H2D transfer; physical KV/Host indices alone are insufficient.
"""

from math import lcm
from threading import RLock

import torch


def indexer_tree_page_size(allocator, page_size):
    """Keep radix sharing on entire compressed groups without resizing KV pages."""
    allocator = getattr(allocator, "full_attn_allocator", allocator)
    backing = getattr(allocator, "offload_backing", None)
    if backing is None:
        return page_size
    pool = backing.hicache_pool.device_pool
    if not getattr(pool, "_kpool_use_compress", False):
        return page_size
    physical_page_size = pool.page_size
    if (
        physical_page_size <= 0
        or pool.index_kpool <= 1
        or physical_page_size % pool.index_kpool
        or page_size <= 0
        or page_size % physical_page_size
        or allocator.page_size != physical_page_size
    ):
        raise ValueError("Incompatible compressed indexer tree/physical page geometry")
    return lcm(page_size, physical_page_size * pool.index_kpool)


class IndexerPageMapping:
    def __init__(self, page_size: int, index_kpool: int, device_pages: int):
        if page_size <= 0 or index_kpool <= 1 or page_size % index_kpool:
            raise ValueError("Compressed KPool must divide the KV page size")
        if device_pages <= 1:
            raise ValueError("Device capacity must include reserved page zero")
        self.page_size = page_size
        self.index_kpool = index_kpool
        self.device_pages = device_pages
        self.slots_per_page = page_size // index_kpool
        self._locations = {}
        self._owners = {}
        self._lock = RLock()

    def _integers(self, values, name):
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 1
            or values.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        return values.detach().to(device="cpu", dtype=torch.int64)

    def _pages(self, values):
        pages = self._integers(values, "Device pages")
        if ((pages <= 0) | (pages >= self.device_pages)).any():
            raise ValueError("Device page is reserved or outside the indexer pool")
        return pages

    def _install(self, pages, anchors, logical_pages):
        pages = self._pages(pages)
        anchors = self._pages(anchors)
        logical_pages = self._integers(logical_pages, "Logical pages")
        if pages.shape != anchors.shape or pages.shape != logical_pages.shape:
            raise ValueError("Indexer mapping input lengths differ")
        if (logical_pages < 0).any():
            raise ValueError("Logical pages must be nonnegative")
        locations = (
            anchors * self.page_size
            + logical_pages.remainder(self.index_kpool) * self.slots_per_page
        )
        with self._lock:
            pending = {}
            owners = {}
            for page, location in zip(pages.tolist(), locations.tolist()):
                previous = pending.get(page, self._locations.get(page))
                if previous is not None and previous != location:
                    raise RuntimeError(
                        "Cannot relocate a live compressed indexer slice"
                    )
                owner = owners.get(location, self._owners.get(location))
                if owner is not None and owner != page:
                    raise RuntimeError(
                        "Shared-prefix KPool branches alias a compressed slice; "
                        "group-aligned sharing or copy-on-write is required"
                    )
                pending[page] = location
                owners[location] = page
            for anchor in anchors.tolist():
                if pending.get(anchor, self._locations.get(anchor)) != (
                    anchor * self.page_size
                ):
                    raise RuntimeError("Missing live logical KPool group anchor")
            self._locations.update(pending)
            self._owners.update(owners)
        return locations

    def update_request_pages(self, request_pages):
        """Register one complete logical page-table prefix, including a partial tail."""
        pages = self._pages(request_pages)
        if pages.unique().numel() != pages.numel():
            raise ValueError("A logical request page table must not repeat KV pages")
        logical_pages = torch.arange(pages.numel(), dtype=torch.int64)
        anchors = pages[logical_pages // self.index_kpool * self.index_kpool]
        return self._install(pages, anchors, logical_pages)

    def _whole_page_ids(self, rows):
        rows = self._integers(rows, "Token rows")
        if rows.numel() % self.page_size:
            raise ValueError("Restore requires a whole-page logical prefix and suffix")
        blocks = rows.reshape(-1, self.page_size)
        starts = blocks[:, 0]
        if (starts % self.page_size).any() or not torch.equal(
            blocks, starts[:, None] + torch.arange(self.page_size)
        ):
            raise ValueError("Restore rows must be ordered within aligned KV pages")
        return self._pages(starts // self.page_size)

    def prepare_restore(self, resident_prefix_rows, restored_rows):
        """Register before H2D enqueue, using the actual root-to-node logical prefix.

        The caller obtains resident_prefix_rows from the FULL radix ancestors,
        not from a possibly stale Req.prefix_indices. restored_rows must follow
        that prefix in the same logical order as the FULL load-back transfer.
        On enqueue failure release only newly allocated pages via the binding.
        """
        prefix_pages = self._whole_page_ids(resident_prefix_rows)
        restored_pages = self._whole_page_ids(restored_rows)
        locations = self.update_request_pages(torch.cat((prefix_pages, restored_pages)))
        return locations[prefix_pages.numel() :]

    def update_requests(
        self, req_to_token, req_pool_indices, seq_lens_cpu, *, start_lens_cpu=None
    ):
        """Read real scheduler page starts and anchors after its table write.

        Admission registers the full prefix. Decode passes previous lengths so
        only newly allocated pages are read, with no D2H on steps inside a page.
        This is a scheduler operation, never an attention/Graph-capture hook.
        """
        lengths = self._integers(seq_lens_cpu, "Sequence lengths")
        starts = (
            torch.zeros_like(lengths)
            if start_lens_cpu is None
            else self._integers(start_lens_cpu, "Previous sequence lengths")
        )
        indices = self._integers(req_pool_indices, "Request indices")
        if lengths.shape != starts.shape or lengths.shape != indices.shape:
            raise ValueError("Request metadata lengths differ")
        if (
            req_to_token.ndim != 2
            or req_to_token.dtype not in (torch.int32, torch.int64)
            or ((starts < 0) | (lengths < starts)).any()
            or (lengths > req_to_token.shape[1]).any()
            or ((indices < 0) | (indices >= req_to_token.shape[0])).any()
        ):
            raise ValueError("Invalid request table or sequence bounds")
        requests = []
        logical = []
        for request, start, length in zip(
            indices.tolist(), starts.tolist(), lengths.tolist()
        ):
            first = (start + self.page_size - 1) // self.page_size
            end = (length + self.page_size - 1) // self.page_size
            requests.extend([request] * (end - first))
            logical.extend(range(first, end))
        if not logical:
            return
        logical_pages = torch.tensor(logical, dtype=torch.int64)
        columns = logical_pages * self.page_size
        anchor_columns = (
            logical_pages // self.index_kpool * self.index_kpool * self.page_size
        )
        row_indices = torch.tensor(requests, device=req_to_token.device)
        rows = torch.stack(
            (
                req_to_token[row_indices, columns.to(req_to_token.device)],
                req_to_token[row_indices, anchor_columns.to(req_to_token.device)],
            )
        ).to(device="cpu", dtype=torch.int64)
        if (rows % self.page_size).any():
            raise ValueError("Logical page starts must name aligned physical KV rows")
        self._install(
            rows[0] // self.page_size, rows[1] // self.page_size, logical_pages
        )

    def indexer_page_locations(self, device_pages):
        pages = self._pages(device_pages)
        with self._lock:
            missing = [page for page in pages.tolist() if page not in self._locations]
            if missing:
                raise RuntimeError(f"No logical indexer mapping for KV pages {missing}")
            return torch.tensor(
                [self._locations[page] for page in pages.tolist()], dtype=torch.int64
            )

    def release_device_pages(self, device_pages):
        """Forget freed pages after transfers finish; keep every live group anchor."""
        pages = set(self._pages(device_pages).tolist())
        with self._lock:
            for page, location in self._locations.items():
                if page not in pages and location // self.page_size in pages:
                    raise RuntimeError(
                        "Cannot free a KPool anchor with live dependent pages"
                    )
            for page in pages:
                location = self._locations.pop(page, None)
                if location is not None:
                    del self._owners[location]

    def clear(self):
        """Reset only after the binding has drained its readers and transfers."""
        with self._lock:
            self._locations.clear()
            self._owners.clear()


def maybe_update_indexer_mapping(
    batch, req_pool_indices_cpu, seq_lens_cpu, *, start_lens_cpu=None
):
    """Scheduler hook for the compressed integrated-offload pool only."""
    allocator = batch.tree_cache.token_to_kv_pool_allocator
    allocator = getattr(allocator, "full_attn_allocator", allocator)
    backing = getattr(allocator, "offload_backing", None)
    if backing is None:
        return
    pool = backing.hicache_pool.device_pool
    if not getattr(pool, "_kpool_use_compress", False):
        return
    mapping = getattr(backing.page_binding, "indexer_mapping", None)
    if mapping is None:
        raise RuntimeError("Compressed HiCache requires binding.indexer_mapping")
    mapping.update_requests(
        batch.req_to_token_pool.req_to_token,
        req_pool_indices_cpu,
        seq_lens_cpu,
        start_lens_cpu=start_lens_cpu,
    )
