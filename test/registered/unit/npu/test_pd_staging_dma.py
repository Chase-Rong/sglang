"""Byte-oriented DMA descriptors for the bounded PD receive window."""

import contextlib
import importlib
import sys
from types import ModuleType, SimpleNamespace

import torch

import test_layer_staging as staging_tests


def test_mapped_host_batch_submits_one_dma_plan_without_host_payload_copy(monkeypatch):
    modules = staging_tests.modules.__wrapped__(monkeypatch)
    staging = importlib.import_module("sglang.srt.hardware_backend.npu.hisparse.layer_staging")
    original_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop("device", None)
        return original_empty(*args, **kwargs)

    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    event = lambda: SimpleNamespace(
        record=lambda current: None, synchronize=lambda: None, query=lambda: True
    )
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device=lambda device: contextlib.nullcontext(),
            stream=lambda current: contextlib.nullcontext(),
            Stream=lambda **kwargs: stream,
            current_stream=lambda: stream,
            Event=event,
        ),
        raising=False,
    )
    calls = []

    def dma(source, metadata, address, plan, to_host):
        calls.append((address, plan.tolist(), to_host))
        for source_offset, destination, count in plan.tolist():
            metadata[destination : destination + count].copy_(
                source[source_offset : source_offset + count]
            )

    native = ModuleType("sglang.srt.hardware_backend.npu.fsa_native")
    native.native_ops = lambda: SimpleNamespace(prefill_dma_copy=dma)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    receiver = staging.create_npu_layer_staging(64, 0)
    buffer = receiver.owner[0]
    buffer.copy_(torch.arange(64, dtype=torch.uint8))
    tensor = torch.zeros((8, 4), dtype=torch.bfloat16)
    region = SimpleNamespace(
        kind=modules[0].PdSourceKind.HOST_REGISTERED,
        owner=SimpleNamespace(tensor=tensor, device_ptr=123456),
    )
    receiver.copy_batch(((region, 16, 16, 0), (region, 48, 16, 16))).synchronize()
    assert calls == [(123456, [[0, 16, 16], [16, 48, 16]], True)]
    actual = tensor.view(torch.uint8).reshape(-1)
    assert torch.equal(actual[16:32], buffer[:16])
    assert torch.equal(actual[48:64], buffer[16:32])
    assert not actual[:16].any() and not actual[32:48].any()
