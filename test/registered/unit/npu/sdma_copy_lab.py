"""Validation-only ACL copy backend; requires TASK_QUEUE_ENABLE=0 in its process."""

import ctypes
import os


def row_copy_runs(
    source_rows, destination_rows, valid, source_limit, destination_limit, row_bytes
):
    if len(source_rows) != len(destination_rows) or len(source_rows) != len(valid):
        raise ValueError("copy metadata lengths differ")
    if min(source_limit, destination_limit, row_bytes) <= 0:
        raise ValueError("invalid copy geometry")
    runs = []
    seen = set()
    for source, destination, enabled in zip(source_rows, destination_rows, valid):
        if not enabled:
            continue
        if not 0 <= source < source_limit or not 0 <= destination < destination_limit:
            raise ValueError("valid copy row is outside its allocation")
        if destination in seen:
            raise ValueError("duplicate destination row")
        seen.add(destination)
        source_offset, destination_offset = source * row_bytes, destination * row_bytes
        if (
            runs
            and source_offset == runs[-1][0] + runs[-1][2]
            and destination_offset == runs[-1][1] + runs[-1][2]
        ):
            previous_source, previous_destination, previous_size = runs[-1]
            runs[-1] = (
                previous_source,
                previous_destination,
                previous_size + row_bytes,
            )
        else:
            runs.append((source_offset, destination_offset, row_bytes))
    return tuple(runs)


class AclCopyBridge:
    def __init__(self, library_path):
        if os.environ.get("TASK_QUEUE_ENABLE") != "0":
            raise RuntimeError(
                "The validation bridge requires TASK_QUEUE_ENABLE=0 before importing torch_npu"
            )
        self.library = ctypes.CDLL(str(library_path))
        self.copy = self.library.fsa_sdma_copy
        self.copy.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.copy.restype = ctypes.c_int
        self.library.fsa_sdma_pin_host.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.library.fsa_sdma_pin_host.restype = ctypes.c_int
        self.library.fsa_sdma_unpin_host.argtypes = [ctypes.c_void_p]
        self.library.fsa_sdma_unpin_host.restype = ctypes.c_int
        self.pending = []
        self.streams = {}
        self.segments = {1: 0, 2: 0, 3: 0}

    def submit(self, source, destination, runs, direction, stream, owners):
        if not runs:
            return
        array_type = ctypes.c_uint64 * len(runs)
        sources = array_type(*(source + entry[0] for entry in runs))
        destinations = array_type(*(destination + entry[1] for entry in runs))
        sizes = array_type(*(entry[2] for entry in runs))
        failed_index = ctypes.c_size_t()
        descriptor = (sources, destinations, sizes, owners)
        self.pending.append(descriptor)
        self.streams[stream.npu_stream] = stream
        code = self.copy(
            sources,
            destinations,
            sizes,
            len(runs),
            direction,
            stream.npu_stream,
            ctypes.byref(failed_index),
        )
        if code:
            raise RuntimeError(
                f"aclrtMemcpyAsync failed: code={code}, segment={failed_index.value}, direction={direction}"
            )
        self.segments[direction] += len(runs)

    def release_after_drain(self):
        if any(not stream.query() for stream in self.streams.values()):
            raise RuntimeError(
                "SDMA descriptors cannot be released before stream completion"
            )
        self.pending.clear()
        self.streams.clear()


class BoundRowCopy:
    def __init__(
        self,
        source_rows,
        destination_rows,
        valid,
        source_limit,
        destination_limit,
        row_bytes,
    ):
        self.runs = row_copy_runs(
            source_rows,
            destination_rows,
            valid,
            source_limit,
            destination_limit,
            row_bytes,
        )
        self.row_bytes = row_bytes
        self.source_limit = source_limit
        self.destination_limit = destination_limit

    def submit(
        self, bridge, source_tensor, destination_tensor, direction, stream, region
    ):
        region._require_live()
        if not source_tensor.is_contiguous() or not destination_tensor.is_contiguous():
            raise ValueError("copy allocation must be contiguous")
        if source_tensor.dtype != destination_tensor.dtype:
            raise ValueError("copy dtypes differ")
        if (
            source_tensor.numel() * source_tensor.element_size()
            < self.source_limit * self.row_bytes
        ):
            raise ValueError("source allocation too small")
        if (
            destination_tensor.numel() * destination_tensor.element_size()
            < self.destination_limit * self.row_bytes
        ):
            raise ValueError("destination allocation too small")
        if direction == 1:
            if (
                source_tensor.data_ptr() != region.host_ptr
                or destination_tensor.device.type != "npu"
            ):
                raise ValueError(
                    "H2D requires original Host pointer and an NPU destination"
                )
        elif direction == 2:
            if (
                destination_tensor.data_ptr() != region.host_ptr
                or source_tensor.device.type != "npu"
            ):
                raise ValueError("D2H requires an NPU source and original Host pointer")
        else:
            raise ValueError("unsupported copy direction")
        bridge.submit(
            source_tensor.data_ptr(),
            destination_tensor.data_ptr(),
            self.runs,
            direction,
            stream,
            (source_tensor, destination_tensor, region),
        )


class FixedPrefillSdmaFixture:
    """Fixed-shape SFA fixture only; never converts device indices in timed copies."""

    def __init__(self, pool, prefix, chunk, library_path, *, pinned_staging=False):
        import torch

        self.torch = torch
        self.pool = pool
        self.bridge = AclCopyBridge(library_path)
        self.host_pool = pool.decode_host_backing.pool
        self.original_load = self.host_pool.copy_from_host
        self.original_save = self.host_pool.copy_to_host_unpublished
        self.staging = {}
        if pinned_staging:
            for names, regions in zip(
                pool.decode_host_backing._component_names,
                pool.decode_host_backing.regions,
            ):
                for name, region in zip(names, regions):
                    if region is not None:
                        pinned = torch.empty_like(region.tensor, pin_memory=True)
                        pinned.copy_(region.tensor)
                        self.staging[name] = pinned
        backing = pool.decode_host_backing
        total_rows = backing.pages * backing.page_size
        width = pool.kv_lora_rank
        row_bytes = width * pool.dtype.itemsize
        prefixes = list(range(128, 128 + prefix))
        additions = list(range(128 + prefix, 128 + prefix + chunk))
        self.prefix = prefix
        self.chunk = chunk
        self.verify_metadata = True
        self.prefixes = prefixes
        self.additions = additions
        self.load = BoundRowCopy(
            prefixes, prefixes, [True] * prefix, total_rows, total_rows, row_bytes
        )
        self.slot_save = BoundRowCopy(
            additions, additions, [True] * chunk, total_rows, total_rows, row_bytes
        )
        self.compact_save = BoundRowCopy(
            list(range(chunk)), additions, [True] * chunk, chunk, total_rows, row_bytes
        )
        self.host_pool.copy_from_host = self._load
        self.host_pool.copy_to_host_unpublished = self._save

    def _verify(
        self, source_indices, destination_indices, valid, sources, destinations
    ):
        if self.verify_metadata:
            if source_indices.cpu().reshape(-1).tolist() != sources:
                raise ValueError(
                    "SDMA fixture source rows differ from actual runtime rows"
                )
            if destination_indices.cpu().reshape(-1).tolist() != destinations:
                raise ValueError(
                    "SDMA fixture destination rows differ from actual runtime rows"
                )
            if not valid.cpu().all().item():
                raise ValueError("SDMA fixed fixture requires all rows to be valid")

    def _load(self, name, destination, source_indices, destination_indices, valid):
        if not self.pool.prefill_offload_runtime.active:
            return self.original_load(
                name, destination, source_indices, destination_indices, valid
            )
        if (
            source_indices.numel() != self.prefix
            or destination_indices.numel() != self.prefix
        ):
            raise ValueError("SDMA fixture does not match this prefix")
        region = self.host_pool.component(name).region
        self._verify(
            source_indices, destination_indices, valid, self.prefixes, self.prefixes
        )
        if name in self.staging:
            self.bridge.submit(
                self.staging[name].data_ptr(),
                destination.data_ptr(),
                self.load.runs,
                1,
                self.torch.npu.current_stream(),
                (self.staging[name], destination),
            )
            return destination
        self.load.submit(
            self.bridge,
            region.tensor,
            destination,
            1,
            self.torch.npu.current_stream(),
            region,
        )
        return destination

    def _save(self, name, source, source_indices, destination_indices, valid):
        if not self.pool.prefill_offload_runtime.active:
            return self.original_save(
                name, source, source_indices, destination_indices, valid
            )
        if (
            source_indices.numel() != self.chunk
            or destination_indices.numel() != self.chunk
        ):
            raise ValueError("SDMA fixture does not match this chunk")
        region = self.host_pool.component(name).region
        plan = self.compact_save if source.shape[0] == self.chunk else self.slot_save
        source_rows = (
            list(range(self.chunk)) if plan is self.compact_save else self.additions
        )
        self._verify(
            source_indices, destination_indices, valid, source_rows, self.additions
        )
        if name in self.staging:
            self.bridge.submit(
                source.data_ptr(),
                self.staging[name].data_ptr(),
                plan.runs,
                2,
                self.torch.npu.current_stream(),
                (source, self.staging[name]),
            )
            return region.tensor
        if self.host_pool.shared_group is None or self.host_pool.shared_group.is_writer:
            plan.submit(
                self.bridge,
                source,
                region.tensor,
                2,
                self.torch.npu.current_stream(),
                region,
            )
        return region.tensor

    def close_after_drain(self):
        self.host_pool.copy_from_host = self.original_load
        self.host_pool.copy_to_host_unpublished = self.original_save
        self.bridge.release_after_drain()

    def export_staging_after_drain(self):
        self.bridge.release_after_drain()
        for name, tensor in self.staging.items():
            self.host_pool.component(name).region.tensor.copy_(tensor)
