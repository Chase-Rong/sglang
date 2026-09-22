from __future__ import annotations

import logging
from functools import cache
from inspect import getattr_static
from typing import Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@cache
def _timing_events_supported() -> bool:
    try:
        device_module.Event(enable_timing=True)
        return True
    except (TypeError, NotImplementedError):
        logger.warning(
            "%s.Event does not support timing; L2 transfer timing is disabled",
            device_module.__name__,
        )
        return False


def make_timing_event_pair():
    timing_enabled = _timing_events_supported()
    kwargs = {"enable_timing": True} if timing_enabled else {}
    return device_module.Event(**kwargs), device_module.Event(**kwargs), timing_enabled


class L2Transfer(NamedTuple):
    host_pool: Any
    device_pool: Any
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    layer_mapper: Optional[Callable[[int], Optional[int]]] = None
    is_draft: bool = False


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
        """Join each pool's backup work into the transfer stream's completion.

        A pool adopting existing Host backing must enqueue waits for its backing's
        ready events on the current stream inside backup_from_device_all_layer.
        That method may omit its payload copy, but must not omit readiness waits.
        The finish event covers these waits and every side pool's actual copy.
        """
        reader_bindings = self._reader_bindings(transfers)
        start_event = self._start_event(None)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.device_to_host_stream):
            start_event.wait(self.device_to_host_stream)
            ack_start.record()
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
            ack_finish.record()
            for binding in reader_bindings:
                binding.record_reader(ack_finish)
            self._record_stream(transfers, self.device_to_host_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def drain_device_to_host(self) -> None:
        """Drain failed write submission before the caller releases its pages.

        Index preparation may have submitted work on the caller's stream before
        backup started. Both streams must finish successfully before rollback;
        a synchronization failure leaves ownership with the caller for recovery.
        """
        try:
            device_module.current_stream().synchronize()
        finally:
            self.device_to_host_stream.synchronize()

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        layer_num: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        """Invoke pool loads inline before publishing layer and transfer events.

        Pools binding Host mappings must update CPU-visible mappings inside
        load_to_device_per_layer before returning, so later CPU submission sees
        them. Device work must join the current stream before on_layer_done;
        a stream event alone cannot defer or publish CPU mapping updates.
        Optional prepare_load_to_device runs once on the caller stream and must
        return a callable taking the local layer id, with immutable copy plans.
        """
        reader_bindings = self._reader_bindings(transfers)
        prepared_loads = []
        for transfer in transfers:
            prepare = (
                getattr(transfer.host_pool, "prepare_load_to_device")
                if getattr_static(transfer.host_pool, "prepare_load_to_device", None)
                is not None
                else None
            )
            load_layer = None
            if callable(prepare):
                load_layer = prepare(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                    is_draft=transfer.is_draft,
                )
                if not callable(load_layer):
                    raise TypeError(
                        "prepare_load_to_device must return a layer callable"
                    )
            prepared_loads.append(load_layer)
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        primary = transfers[0] if transfers else None
        with device_module.stream(self.host_to_device_stream):
            start_event.wait(self.host_to_device_stream)
            ack_start.record()
            for layer_id in range(layer_num):
                for transfer, load_layer in zip(transfers, prepared_loads):
                    local_layer_id = (
                        transfer.layer_mapper(layer_id)
                        if transfer.layer_mapper is not None
                        else layer_id
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and layer_id >= transfer.host_pool.layer_num
                    ):
                        continue
                    if load_layer is not None:
                        load_layer(local_layer_id)
                    else:
                        transfer.host_pool.load_to_device_per_layer(
                            transfer.device_pool,
                            transfer.host_indices,
                            transfer.device_indices,
                            local_layer_id,
                            self.io_backend,
                            is_draft=transfer.is_draft,
                        )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
            ack_finish.record()
            for binding in reader_bindings:
                binding.record_reader(ack_finish)
            self._record_stream(transfers, self.host_to_device_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    @staticmethod
    def _reader_bindings(transfers: list[L2Transfer]) -> list[Any]:
        """Require adopted backing to retain all anchor and sidecar readers."""
        bindings = {}
        for transfer in transfers:
            binding = getattr(transfer.host_pool, "binding", None)
            if binding is None and (
                transfer is not transfers[0]
                or not callable(
                    getattr(transfer.host_pool, "acquire_device_pages", None)
                )
            ):
                continue
            if not callable(getattr(binding, "record_reader", None)):
                raise RuntimeError(
                    "Adopted HiCache backing requires binding.record_reader(event) "
                    "to protect transfer-stream readers."
                )
            bindings[id(binding)] = binding
        return list(bindings.values())

    @staticmethod
    def _start_event(start_event):
        if start_event is None:
            start_event = device_module.Event()
            start_event.record()
        return start_event

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream) -> None:
        for transfer in transfers:
            for indices in (transfer.host_indices, transfer.device_indices):
                if indices.device.type in (
                    "cuda",
                    "npu",
                    "privateuseone",
                    "privateuse1",
                ):
                    indices.record_stream(stream)
