"""Host ownership hooks shared by FSA capture and the NPU graph executor.

Capture discovers the bindings used by a graph. Only the outer replay owner can
record a fresh fence after all captured native reads and publication operations.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_CAPTURE_READERS = ContextVar("fsa_capture_readers", default=None)
class FsaGraphReaders:
    def __init__(self):
        self.readers = {}
        self.shared = {}
        self.retained = []
        self.states = {}
        self._acquired_guards = []

    def add(self, binding, epoch_input):
        self.readers[(id(binding), id(epoch_input))] = (binding, epoch_input)

    def add_shared(self, resident, guard, binding, epoch_input, retained, state=None):
        self.shared[id(guard)] = (resident, guard)
        self.retained.append(retained)
        if state is not None:
            self.states[id(state)] = state
        if binding is not None:
            self.add(binding, epoch_input)

    def finish_capture(self, device_module):
        if not self.readers and not self.shared:
            return
        try:
            # Capture may run on an internal stream; it is initialization, not
            # a replay reader fence. Drain before exposing the captured graph.
            from .shared_execution import drain_streams

            streams = []
            if self.shared:
                streams.append(device_module.current_stream())
            for _, guard in self.shared.values():
                main = getattr(guard, "main_stream", None)
                if main is not None and hasattr(main, "synchronize"):
                    streams.append(main)
                streams.extend(getattr(guard, "side_streams", ()))
            drain_streams(streams)
            device_module.synchronize()
            for retained in self.retained:
                if retained:
                    lease = retained[0]
                    lease.resident.cancel_execution(
                        lease.layer_id, lease.row_request_slots, execution_mask=lease
                    )
            # Cancel itself is device work; keep guards until it drains too.
            if self.shared:
                device_module.synchronize()
            for _, guard in self.shared.values():
                guard.drained()
        except Exception as error:
            for _, guard in self.shared.values():
                guard.fail(error)
            for binding, _ in self.readers.values():
                binding.mark_reader_failure(error)
            raise

    def before_replay(self):
        # Guard before epoch writes; all workspaces are acquired transactionally.
        acquired = self._acquired_guards = []
        try:
            for _, guard in self.shared.values():
                guard.acquire()
                acquired.append(guard)
        except Exception:
            # Outer replay owner drains any writes queued by earlier acquire/
            # refresh stages. Never expose a guard with pending writes.
            raise
        try:
            for state in self.states.values():
                state.refresh_generations()
            for binding, epoch_input in self.readers.values():
                binding.check_reader_health()
                if epoch_input is not None:
                    epoch_input.fill_(binding.mapping_epoch)
        except Exception:
            # Outer replay owner drains any writes queued by earlier acquire/
            # refresh stages. Never expose a guard with pending writes.
            raise

    def after_replay(self, device_module):
        if self.readers:
            event = device_module.Event()
            event.record(device_module.current_stream())
            bindings = {id(binding): binding for binding, _ in self.readers.values()}
            for binding in bindings.values():
                binding.record_reader(event)
        from .shared_execution import (
            register_shared_resident,
            check_shared_residents,
            _FORWARD,
        )

        for resident, guard in self.shared.values():
            guard.release()
            register_shared_resident(resident)
        if self.shared and _FORWARD.get() is None:
            check_shared_residents(resident for resident, _ in self.shared.values())

    def drain_failure(self, device_module):
        if not self.readers and not self.shared:
            return
        try:
            from .shared_execution import drain_streams

            drain_streams(
                (
                    device_module.current_stream(),
                    *(
                        stream
                        for _, guard in self.shared.values()
                        for stream in getattr(guard, "side_streams", ())
                    ),
                )
            )
            owned = {
                (id(resident), getattr(guard, "layer_id", None))
                for resident, guard in self.shared.values()
                if guard in self._acquired_guards
            }
            for retained in self.retained:
                if not retained:
                    continue
                lease = retained[0]
                if (id(lease.resident), lease.layer_id) in owned:
                    lease.resident.cancel_execution(
                        lease.layer_id, lease.row_request_slots, execution_mask=lease
                    )
            if self.shared:
                device_module.current_stream().synchronize()
            for guard in self._acquired_guards:
                guard.drained()
            self._acquired_guards = []
        except Exception as error:
            for guard in self._acquired_guards:
                guard.fail(error)
            for binding, _ in self.readers.values():
                binding.mark_reader_failure(error)
            raise


@contextmanager
def collect_fsa_graph_readers():
    readers = FsaGraphReaders()
    token = _CAPTURE_READERS.set(readers)
    try:
        yield readers
    finally:
        _CAPTURE_READERS.reset(token)


def register_capture_reader(binding, epoch_input):
    readers = _CAPTURE_READERS.get()
    if readers is None:
        raise RuntimeError(
            "external_plan Host capture requires NPUCudaGraphBackend replay ownership"
        )
    readers.add(binding, epoch_input)


def register_capture_shared(
    resident, guard, binding, epoch_input, retained, state=None
):
    readers = _CAPTURE_READERS.get()
    if readers is None:
        raise RuntimeError(
            "shared capture requires NPUCudaGraphBackend replay ownership"
        )
    readers.add_shared(resident, guard, binding, epoch_input, retained, state)
