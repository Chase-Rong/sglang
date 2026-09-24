"""CPU integration of packed MTP indexer routing and L2 completion ordering.

Production controller, engine, pool group and indexer classes are imported
directly. Only device streams/events and the KV anchor are CPU substitutes.
"""

from types import SimpleNamespace

import torch
from test_hicache_offload_completion import Device

from sglang.srt.hardware_backend.npu.hicache_indexer_pool import NPUIndexerPoolHost
from sglang.srt.mem_cache import l2_transfer
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry


class CpuAnchor:
    """External KV ownership interface; payload is already resident on Host."""

    page_size = 8
    page_num = 7
    size = logical_size = 56
    layer_num = 3
    device = "cpu"
    layout = "layer_first"
    can_use_write_back_jit = False

    def __init__(self, target, draft):
        self.device_pool = target
        self.mtp_draft_device_pools = (draft,)
        self.binding = self
        self.reader_events = []

    def indexer_page_locations(self, pages):
        locations = {3: 24, 5: 28, 1: 8, 6: 12}
        return torch.tensor([locations[int(page)] for page in pages])

    def record_reader(self, event):
        self.reader_events.append(event)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        assert device_pool is self.device_pool and not is_draft
        assert layer_id in (0, 1)
        # Mapping is CPU metadata and already installed by this test anchor.


def token_rows(pages):
    return (torch.tensor(pages)[:, None] * 8 + torch.arange(8)).flatten()


def indexer_device(layers):
    return SimpleNamespace(
        layer_num=layers,
        start_layer=0,
        end_layer=layers,
        page_size=8,
        index_kpool=2,
        _kpool_use_compress=True,
        tail_extra_slots=3,
        index_k_buffer=torch.zeros((layers, 9, 8, 1, 4), dtype=torch.bfloat16),
    )


def test_packed_draft_indexer_is_restored_before_stage_zero_completion(monkeypatch):
    device = Device()
    monkeypatch.setattr(l2_transfer, "device_module", device)
    l2_transfer._timing_events_supported.cache_clear()
    try:
        target, draft = indexer_device(2), indexer_device(1)
        target.index_k_buffer[0, 3].fill_(11)
        target.index_k_buffer[1, 3].fill_(22)
        draft.index_k_buffer[0, 3].fill_(73)
        anchor = CpuAnchor(target, draft)
        indexer = NPUIndexerPoolHost(target, anchor, pin_memory=False)
        host_rows = token_rows([4, 2])
        indexer.backup_from_device_all_layer(
            target, host_rows, token_rows([5, 3]), "direct"
        )
        target.index_k_buffer.fill_(-1)
        draft.index_k_buffer.fill_(-1)

        # The target has FULL layers at stages 0 and 2. Stage 3 is a packed
        # draft lookup key; the controller must fold it into execution stage 0.
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.layer_num = 3
        controller.mem_pool_host = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=anchor,
                    device_pool=target,
                    layer_mapper={0: 0, 2: 1}.get,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=indexer,
                    device_pool=target,
                    layer_mapper={0: 0, 2: 1, 3: 2}.get,
                    packed_draft_device_pools=(draft,),
                ),
            ]
        )
        transfers = controller._l2_load_transfers(
            host_rows,
            token_rows([6, 1]),
            [
                PoolTransfer(
                    name=PoolName.INDEXER,
                    host_indices=host_rows,
                    device_indices=token_rows([6, 1]),
                    indices_from_pool=PoolName.KV,
                )
            ],
        )
        completed_stages = []

        def on_layer_done(stage):
            assert torch.all(draft.index_k_buffer[0, 1] == 73), (
                "Packed draft INDEXER must be restored before stage 0 completes"
            )
            assert torch.all(target.index_k_buffer[0, 1] == 11)
            assert torch.all(target.index_k_buffer[1, 1] == (22 if stage == 2 else -1))
            completed_stages.append(stage)

        engine = l2_transfer.L2TransferEngine("direct")
        completion = engine.submit_host_to_device(
            transfers, layer_num=3, on_layer_done=on_layer_done
        )
        assert completed_stages == [0, 1, 2]
        assert not completion.finish_event.query()
        device.progress()
        assert completion.finish_event.query()
        assert anchor.reader_events[-1].query()
    finally:
        l2_transfer._timing_events_supported.cache_clear()
