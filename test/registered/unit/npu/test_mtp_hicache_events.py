"""CPU stream simulation for target/draft Host writes and real HiCache backup."""

import importlib.util
import logging
from functools import cache
from inspect import getattr_static
from types import SimpleNamespace

import pytest
import torch
from test_hicache_offload_completion import SRT, Device, Gate, extract


@pytest.mark.parametrize("first_ready", ["target", "draft"])
def test_packed_backup_waits_for_both_runtime_dma_writes(monkeypatch, first_ready):
    device = Device()
    create_stream = device.Stream
    device.Stream = lambda device=None: create_stream()
    device.is_current_stream_capturing = lambda: False
    monkeypatch.setattr(torch, "npu", device, raising=False)
    namespace = dict(
        torch=torch,
        HostKVCache=object,
        device_module=device,
        cache=cache,
        getattr_static=getattr_static,
        logger=logging.getLogger(__name__),
    )
    from typing import NamedTuple

    namespace["NamedTuple"] = NamedTuple
    extract(
        "hardware_backend/npu/hicache_host_pool.py",
        namespace,
        {
            "MemFabricHiCachePool": {
                "_require_live",
                "_require_binding",
                "backup_from_device_all_layer",
            }
        },
    )
    extract(
        "mem_cache/l2_transfer.py",
        namespace,
        dict.fromkeys(
            (
                "_timing_events_supported",
                "make_timing_event_pair",
                "L2Transfer",
                "TransferCompletion",
                "L2TransferEngine",
            )
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "mtp_event_runtime", SRT / "hardware_backend/npu/prefill_overlap.py"
    )
    runtime_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_module)
    readers = []
    binding = SimpleNamespace(ready_event=None, record_reader=readers.append)
    rows = torch.tensor([2, 3])
    request_pool = SimpleNamespace(req_to_token=rows.reshape(1, -1))
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=rows,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )

    def submit_forward(value):
        host = torch.full((8, 1), -1.0)
        storage = torch.zeros(1, 8, 1)
        gate = Gate()

        def enqueue_save(layer, components, locations, valid, *, plan):
            def copy():
                host[locations] = components[0][locations]

            device.current_stream().wait_event(gate)
            device.current_stream().enqueue(copy)

        backing = SimpleNamespace(
            pages=4,
            page_size=2,
            page_binding=binding,
            _component_names=[("latent", None)],
            pool=SimpleNamespace(_closed=False),
            plan_prefill_rows=lambda prefix, new: (None, None),
            enqueue_prefill_rows_from_slot=enqueue_save,
            publish_prefill_writes=lambda: None,
        )
        pool = SimpleNamespace(
            integrated_offload=True,
            decode_host_backing=backing,
            prefill_layer_slots=1,
            start_layer=0,
            layer_num=1,
            kv_lora_rank=1,
            qk_rope_head_dim=0,
            k_buffer=storage,
            physical_layer_index=lambda layer: 0,
            raw_layer_components=lambda layer: (storage[0], torch.empty(0)),
        )
        runtime = runtime_module.PipelinedIntegratedOffloadRuntime(pool, request_pool)

        def scatter():
            storage[0, rows] = value

        with runtime.forward(batch):
            runtime.prepare_layer(0)
            device.current_stream().enqueue(scatter)
            runtime.on_kv_written(0, rows)
            runtime.on_compute_start(0)
            runtime.finish_layer(0)
        return SimpleNamespace(pool=pool, host=host, gate=gate, runtime=runtime)

    target = submit_forward(11.0)
    draft = submit_forward(29.0)
    anchor = namespace["MemFabricHiCachePool"]()
    anchor._destroyed = False
    anchor.backing = target.pool.decode_host_backing
    anchor.device_pool = target.pool
    anchor.mtp_draft_device_pools = (draft.pool,)
    anchor.binding = binding
    engine = namespace["L2TransferEngine"]("direct")
    # Scheduler submission has no implicit dependency on the forward stream.
    # The real Host backup method must carry the shared readiness dependency.
    with device.stream(device.Stream()):
        completion = engine.submit_device_to_host(
            [namespace["L2Transfer"](anchor, target.pool, rows, rows)]
        )

    device.progress()
    assert completion.start_event.query()
    assert not completion.finish_event.query()
    assert target.host[rows].flatten().tolist() == [-1.0, -1.0]
    assert draft.host[rows].flatten().tolist() == [-1.0, -1.0]

    {"target": target, "draft": draft}[first_ready].gate.open = True
    device.progress()
    assert not completion.finish_event.query()
    assert draft.host[rows].flatten().tolist() == [-1.0, -1.0]

    target.gate.open = draft.gate.open = True
    device.progress()
    assert completion.finish_event.query()
    assert target.host[rows].flatten().tolist() == [11.0, 11.0]
    assert draft.host[rows].flatten().tolist() == [29.0, 29.0]
    assert readers and all(event.query() for event in readers)
    assert not device.synchronized
