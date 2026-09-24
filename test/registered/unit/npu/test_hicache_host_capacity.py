"""Independent Host capacity with real pools, ownership and CPU payloads.

Only physical MemFabric registration and NPU events are simulated by the
existing fixtures. These checks do not qualify native DMA or graph execution.
"""

import pytest
import torch
from test_hicache_memfabric_pool import runtime as runtime
from test_prefill_offload_runtime import production_pool_type


def test_more_idle_prefixes_than_device_pages_survive_mtp_restore(runtime):
    target, target_device = runtime.make(pages=3, host_tokens=32)
    draft, draft_device = runtime.make(pages=3, host_tokens=32, layers=1)
    host = target.bind_hicache(target_device)
    host.attach_mtp_drafts((draft_device,))
    binding = host.binding
    assert host.usable_size == 32 and target_device.size == 8
    assert binding.device_to_host_cpu.numel() == 3
    assert binding.references.numel() == 9
    prefixes = []
    for value in range(1, 9):
        assert binding.ensure_pages(torch.tensor([1]))
        cached = host.acquire_device_pages(torch.arange(4, 8))
        target.regions[0][0].tensor[cached] = value
        draft.regions[0][0].tensor[cached] = value + 100
        prefixes.append(cached)
        binding.release_device_pages(torch.tensor([1]))
    assert host.available_size() == 0
    assert not binding.device_to_host_cpu.any()
    # All eight prefixes survive despite only two usable device pages.
    for value, cached in enumerate(prefixes, 1):
        assert torch.all(target.regions[0][0].tensor[cached] == value)
        assert torch.all(draft.regions[0][0].tensor[cached] == value + 100)
        assert binding.references[cached[0] // 4] == 1
    assert not binding.ensure_pages(torch.tensor([1]))
    high_host_rows = prefixes[-1]
    assert high_host_rows.tolist() == [32, 33, 34, 35]
    destination = torch.arange(8, 12)
    host.load_to_device_per_layer(
        target_device, high_host_rows, destination, 0, "kernel_ascend"
    )
    host.load_to_device_per_layer(
        draft_device, high_host_rows, destination, 2, "kernel_ascend", is_draft=True
    )
    assert torch.equal(binding.host_rows(destination), high_host_rows)
    prefix_plan, _ = target.plan_prefill_rows(destination, destination[:0])
    assert prefix_plan.tolist() == [[32, 8, 4]]
    # Freeing a cached reference cannot free the active reader's page.
    host.free(high_host_rows)
    assert host.available_size() == 0
    binding.release_device_pages(torch.tensor([2]))
    assert host.available_size() == 0
    runtime.device_events[-1].complete = True
    binding.reap()
    assert host.available_size() == 4
    assert binding.ensure_pages(torch.tensor([1]))


def test_mtp_rejects_different_host_capacities_before_publishing_binding(runtime):
    target, target_device = runtime.make(pages=3, host_tokens=32)
    draft, draft_device = runtime.make(pages=3, host_tokens=16, layers=1)
    host = target.bind_hicache(target_device)
    with pytest.raises(ValueError, match="geometry"):
        host.attach_mtp_drafts((draft_device,))
    assert host.mtp_draft_device_pools == ()
    assert draft.page_binding is None


def test_real_pool_constructor_expands_host_without_expanding_hbm(runtime, monkeypatch):
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0, raising=False)
    cls = production_pool_type()
    pools = []
    try:
        for host_tokens in (0, 128):
            pool = cls(
                size=16,
                page_size=4,
                dtype=torch.bfloat16,
                kv_lora_rank=8,
                qk_rope_head_dim=0,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                index_head_dim=4,
                index_kpool=2,
                index_kpool_compress=True,
                max_running_requests=2,
                prefill_layer_slots=1,
                decode_host_budget=1 << 20,
                integrated_offload=True,
                decode_host_token_capacity=host_tokens,
            )
            pools.append(pool)
        small, large = pools
        assert small.size == large.size == 16
        assert small.k_buffer.shape == large.k_buffer.shape == (1, 5, 4, 1, 8)
        assert small.index_k_buffer.shape == large.index_k_buffer.shape
        assert small.decode_host_backing.pages == 5
        assert large.decode_host_backing.pages == 33
        assert large.decode_host_backing.layer_views(0)[0].shape == (33, 4, 1, 8)
    finally:
        for pool in reversed(pools):
            pool.decode_host_backing.pool.close(drained=True)


@pytest.mark.parametrize("host_tokens", [-4, 17])
def test_invalid_host_capacity_rejected_before_physical_allocation(
    runtime, host_tokens
):
    before = list(runtime.events)
    with pytest.raises(ValueError, match="Host"):
        runtime.make(pages=3, host_tokens=host_tokens)
    assert runtime.events == before
