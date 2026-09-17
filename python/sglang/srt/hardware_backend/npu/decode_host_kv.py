"""Registered Decode backing with resident Indexer and stream-ordered writes."""

import torch

from sglang.srt.hardware_backend.npu.registered_host_kv import NpuRegisteredHostKvPool


class DecodeHostKvBacking:
    def __init__(
        self,
        *,
        layers,
        pages,
        page_size,
        latent_dim,
        rope_dim,
        dtype,
        device_id,
        host_budget,
        shared_group=None,
        integrated=False,
    ):
        if rope_dim < 0 or min(layers, pages, page_size, latent_dim) <= 0:
            raise ValueError(
                "Host MLA backing dimensions must be positive, with optional RoPE"
            )
        required = layers * pages * page_size * (latent_dim + rope_dim)
        required *= torch.empty((), dtype=dtype).element_size()
        if host_budget <= 0 or required > host_budget:
            raise ValueError(
                f"Decode Host KV requires {required} bytes, budget is {host_budget}"
            )
        if integrated:
            from sglang.srt.hardware_backend.npu.memfabric_host_kv import (
                MemFabricHostKvPool,
            )

            self.pool = MemFabricHostKvPool(
                device_id, required_bytes=required, shared_group=shared_group
            )
        elif shared_group is None:
            self.pool = NpuRegisteredHostKvPool(device_id)
        else:
            self.pool = NpuRegisteredHostKvPool(device_id, shared_group=shared_group)
        self.page_size = page_size
        self.pages = pages
        self.regions = []
        self.rope_dim = rope_dim
        self.dtype = dtype
        self._component_names = []
        try:
            self._allocate_layers(layers, pages, page_size, latent_dim, rope_dim, dtype)
        except BaseException:
            self.pool.close(drained=True)
            raise

    def _allocate_layers(self, layers, pages, page_size, latent_dim, rope_dim, dtype):
        for layer_id in range(layers):
            components = []
            names = []
            for component, width in (("latent", latent_dim), ("rope", rope_dim)):
                if width == 0:
                    components.append(None)
                    names.append(None)
                    continue
                name = f"decode-{layer_id}-{component}"
                entry = self.pool.allocate_component(
                    name, (pages * page_size, width), dtype, page_size=page_size
                )
                components.append(entry.region)
                names.append(name)
            self.regions.append(tuple(components))
            self._component_names.append(tuple(names))

    def layer_views(self, layer_id):
        return tuple(
            (
                region.view((self.pages, self.page_size, 1, region.tensor.shape[-1]))
                if region is not None
                else torch.empty((self.pages, self.page_size, 1, 0), dtype=self.dtype)
            )
            for region in self.regions[layer_id]
        )

    def write_current(self, layer_id, locations, latent, rope):
        destinations = locations.reshape(-1).to(dtype=torch.int64).contiguous()
        source_rows = torch.arange(
            destinations.numel(), device=locations.device, dtype=torch.int64
        )
        valid = (destinations > 0) & (destinations < self.pages * self.page_size)
        for name, region, values in zip(
            self._component_names[layer_id], self.regions[layer_id], (latent, rope)
        ):
            if region is None:
                if values.numel() != 0:
                    raise ValueError("NoPE Host backing cannot accept RoPE values")
                continue
            rows = values.reshape(-1, region.tensor.shape[-1]).contiguous()
            if (
                rows.shape[0] != destinations.numel()
                or rows.dtype != region.tensor.dtype
            ):
                raise ValueError("current-token rows do not match Decode Host geometry")
            if rows.device != locations.device:
                raise ValueError("current-token data and locations must share an NPU")
            self.pool.copy_to_host(name, rows, source_rows, destinations, valid)

    @property
    def nbytes(self):
        return sum(
            region.nbytes
            for layer in self.regions
            for region in layer
            if region is not None
        )

    def enqueue_prefill_rows_from_slot(
        self, layer_id, components, rows, valid, *, plan=None
    ):
        for name, region, source in zip(
            self._component_names[layer_id], self.regions[layer_id], components
        ):
            if region is not None:
                width = region.tensor.shape[-1]
                if source.dtype != region.tensor.dtype:
                    raise ValueError("Prefill slot and Host KV dtypes differ")
                if plan is not None:
                    self.pool.copy_prefill_to_host(name, source.view(-1, width), plan)
                else:
                    self.pool.copy_to_host_unpublished(
                        name, source.view(-1, width), rows, rows, valid
                    )

    def publish_prefill_writes(self):
        if self.pool.shared_group is not None:
            for components in self._component_names:
                for name in components:
                    if name is not None:
                        self.pool.shared_group.publish_write()
