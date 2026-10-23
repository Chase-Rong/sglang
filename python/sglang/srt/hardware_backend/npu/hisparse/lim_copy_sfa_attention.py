"""Single-launch A5 miss-copy, sparse attention and residency publication."""

import torch

from sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_indexer import native_indexer_ops


class NativeCopySparseAttention:
    def __init__(self, device, row_capacity=384):
        self.ops = native_indexer_ops()
        self.core_count = torch.npu.get_device_properties(device).cube_core_num
        self.workspace = torch.empty(
            self.core_count * 3 * 128 * 576 * 2
            + (row_capacity * 4 + 31) // 32 * 32
            + row_capacity * 512 * 512 * 2,
            dtype=torch.uint8,
            device=device,
        )

    def forward(
        self,
        query,
        host,
        region,
        resident,
        planner,
        ends,
        lengths,
        kv_lengths,
        table,
        requests,
        *,
        scale,
        mutable_tail=12,
        reuse_plan=False,
    ):
        if region is None:
            if host.device != query.device:
                raise ValueError("Host CopySFA requires registered MemFabric backing")
            address = host.data_ptr()
        else:
            if (
                host.device.type != "cpu"
                or not host.is_contiguous()
                or host.data_ptr() != region.host_ptr
                or host.nbytes > region.nbytes
                or region.device_id != query.device.index
                or region.device_ptr <= 0
            ):
                raise ValueError("CopySFA Host view does not match its registration")
            address = region.device_ptr
        rows, batch = query.shape[0], ends.numel()
        output = torch.empty_like(query)
        self.ops.copy_sfa_out(
            query,
            host,
            resident,
            planner.topk[:rows],
            ends,
            lengths,
            kv_lengths,
            table,
            requests,
            planner.membership[:rows],
            planner.union_sources[:batch],
            planner.union_slots[:batch],
            planner.union_physical[:batch],
            planner.physical,
            planner.state,
            output,
            self.workspace,
            address,
            mutable_tail,
            scale,
            min(self.core_count, max(8, rows)),
            **({"reuse_plan": True} if reuse_plan else {}),
        )
        return output
