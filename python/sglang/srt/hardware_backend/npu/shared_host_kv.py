"""Node-local replicated MLA backing with collective attach and one writer."""

import os
from pathlib import Path
from uuid import uuid4

import torch


def local_shared_identity(writer_rank):
    namespace = os.stat("/proc/self/ns/ipc")
    return (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        namespace.st_dev,
        namespace.st_ino,
        writer_rank,
    )


class SharedHostGroup:
    def __init__(self, group, device_id, *, writer_rank=0):
        if not 0 <= writer_rank < group.world_size:
            raise ValueError("shared Host writer rank is outside the TP group")
        self.group = group
        self.device_id = device_id
        self.writer_rank = writer_rank
        self.sequence = 0
        self.failed = False
        identity = local_shared_identity(writer_rank)
        peers = group.all_gather_object(identity)
        if any(peer != identity for peer in peers):
            raise ValueError("shared Host TP ranks must share a host and IPC namespace")
        self.group_id = group.broadcast_object(
            uuid4().hex if self.is_writer else None, src=writer_rank
        )
        self.fence = torch.zeros(1, dtype=torch.int32, device=f"npu:{device_id}")

    @property
    def is_writer(self):
        return self.group.rank_in_group == self.writer_rank

    def validate_identity(self, identity):
        if self.failed:
            raise RuntimeError("shared Host TP group has failed")
        peers = self.group.all_gather_object(identity)
        if any(peer != identity for peer in peers):
            self.failed = True
            raise ValueError(
                "shared Host ranks disagree on KV geometry or page identity"
            )

    def allocate(self, region_type, name, shape, dtype, page_size):
        self.validate_identity(
            (self.sequence, name, tuple(shape), str(dtype), page_size)
        )
        self.sequence += 1
        region = None
        payload = None
        error = None
        if self.is_writer:
            try:
                region = region_type.allocate(shape, dtype, device_id=self.device_id)
                payload = (region.export_handle(), None)
            except Exception as exc:
                payload = (None, str(exc))
        try:
            handle, error = self.group.broadcast_object(payload, src=self.writer_rank)
            if error is None and not self.is_writer:
                try:
                    region = region_type.attach(handle, device_id=self.device_id)
                except Exception as exc:
                    error = str(exc)
            errors = self.group.all_gather_object(error)
            if any(message is not None for message in errors):
                self.failed = True
                raise RuntimeError(f"shared Host allocation/attach failed: {errors}")
        except Exception:
            self.failed = True
            if region is not None:
                region.close(drained=True)
            raise
        return region

    def publish_write(self):
        """Order the owner's preceding copy before every rank's next reader.

        This collective is captured along with current-token writes. It is a
        stream dependency; mapped Host visibility still needs the native gate.
        """
        if self.failed:
            raise RuntimeError("shared Host TP group has failed")
        self.group.broadcast(self.fence, src=self.writer_rank)


def shared_host_group_from_runtime(device_id):
    from sglang.srt.environ import envs

    if not envs.SGLANG_NPU_SHARED_HOST_KV.get():
        return None
    from sglang.srt.distributed.parallel_state import get_attn_tp_group
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.attn_dcp_size != 1:
        raise ValueError("shared MLA Host backing requires replicated KV with DCP=1")
    return SharedHostGroup(get_attn_tp_group(), device_id)


def shared_abort_drain_readiness(group, local_drained):
    reports = group.all_gather_object(local_drained)
    return {
        room: all(report.get(room, False) for report in reports)
        for room in local_drained
    }
