"""FSA adapter contracts; CPU tensors, simulated native work and NPU fences.

The planner, row materialization, resident publication and wrapper are real.
Device execution/fence timing is checked separately by the NPU probes.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


















def load_graph_backend(monkeypatch):
    """Only import-time serving dependencies are stubbed; replay is production."""
    import importlib.util
    import os
    import sys
    from pathlib import Path

    names = {
        "sglang.srt.constants": dict(GPU_MEMORY_TYPE_CUDA_GRAPH="graph"),
        "sglang.srt.distributed.device_communicators.pynccl_allocator": dict(
            set_graph_pool_id=lambda x: None
        ),
        "sglang.srt.model_executor.runner.shape_key": dict(ShapeKey=object),
        "sglang.srt.model_executor.runner_backend.base_cuda_graph_backend": dict(
            BaseCudaGraphBackend=object
        ),
        "sglang.srt.runtime_context": dict(get_parallel=lambda: None),
        "sglang.srt.utils": dict(empty_context=None, get_bool_env_var=lambda x: False),
        "sglang.srt.utils.torch_memory_saver_adapter": dict(
            TorchMemorySaverAdapter=object
        ),
    }
    for name, attrs in names.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    root = Path(
        os.environ.get("FSA_BASELINE_ROOT", str(Path(__file__).resolve().parents[4]))
    )
    path = (
        root
        / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
    )
    spec = importlib.util.spec_from_file_location("fsa_lifecycle_backend", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NPUCudaGraphBackend


@pytest.mark.parametrize("update_inputs", [False, True])
def test_real_graph_backend_refreshes_epoch_and_records_each_replay(
    monkeypatch, update_inputs
):
    from concurrent.futures import ThreadPoolExecutor
    from sglang.srt.hardware_backend.npu.hisparse.fsa_graph_lifecycle import (
        collect_fsa_graph_readers,
        register_capture_reader,
    )

    cls = load_graph_backend(monkeypatch)
    backend = object.__new__(cls)
    order, events = [], []
    epoch = torch.tensor(1)

    class Event:
        def record(self, stream=None):
            order.append("record")

        def synchronize(self):
            pass

    binding = SimpleNamespace(
        mapping_epoch=1, record_reader=events.append, check_reader_health=lambda: None
    )
    with collect_fsa_graph_readers() as readers:
        register_capture_reader(binding, epoch)

    def replay():
        order.append("replay")
        assert int(epoch) == binding.mapping_epoch

    backend._graphs = {1: SimpleNamespace(replay=replay, update=lambda **kwargs: None)}
    backend._outputs = {1: "out"}
    backend._fsa_readers = {1: readers}
    backend._device_module = SimpleNamespace(Event=Event, current_stream=lambda: None)
    backend._rebind_fence = None
    backend._update_executor = ThreadPoolExecutor(max_workers=1)
    try:
        for new_epoch in (2, 3):
            binding.mapping_epoch = new_epoch
            if update_inputs:
                assert backend.replay_with_input_update(1, [4], "seq_lens") == "out"
            else:
                assert backend.replay(1, None) == "out"
        assert len(events) == 2
        assert events[0] is not events[1]
        assert order.index("replay") < order.index("record")
    finally:
        backend._update_executor.shutdown()








@pytest.mark.parametrize(
    "forward_fails,drain_fails,has_reader",
    [
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_graph_capture_registers_binding_and_drains_before_publishing(
    monkeypatch, forward_fails, drain_fails, has_reader
):
    import contextlib
    import sys
    from sglang.srt.hardware_backend.npu.hisparse.fsa_graph_lifecycle import (
        register_capture_reader,
    )

    cls = load_graph_backend(monkeypatch)
    backend = object.__new__(cls)
    order, capturing = [], [False]
    from threading import RLock
    from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import (
        OffloadPageBinding,
    )

    binding = object.__new__(OffloadPageBinding)
    binding.lock = RLock()
    binding.reader_failure = None
    epoch = torch.tensor(1)

    @contextlib.contextmanager
    def graph_context(*args, **kwargs):
        capturing[0] = True
        try:
            yield
        finally:
            capturing[0] = False
            order.append("capture_ended")

    def drain():
        after_capture = bool(order) and order[-1] == "capture_ended"
        order.append("drain")
        if after_capture and drain_fails:
            raise RuntimeError("injected capture drain failure")

    api = SimpleNamespace(NPUGraph=object, graph=graph_context, synchronize=drain)
    monkeypatch.setattr(torch, "npu", api, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    # load_graph_backend isolates import-time utilities, including empty_context.
    monkeypatch.setitem(
        cls.capture_one.__globals__, "empty_context", contextlib.nullcontext
    )
    backend._graphs, backend._outputs, backend._fsa_readers = {}, {}, {}
    backend._device_module = api
    backend._tp_group = SimpleNamespace(barrier=lambda: None)
    backend._pool = None
    backend._capture_stream = None
    backend._memory_saver_adapter = None
    backend._enable_torch_compile = False

    def forward():
        if capturing[0]:
            if has_reader:
                register_capture_reader(binding, epoch)
            if forward_fails:
                raise RuntimeError("injected capture failure")
        return "captured output"

    if forward_fails:
        with pytest.raises(RuntimeError, match="injected capture"):
            backend.capture_one(1, forward)
        assert not backend._graphs
    else:
        backend.capture_one(1, forward)
        assert backend._outputs[1] == "captured output"
        assert len(backend._fsa_readers[1].readers) == int(has_reader)
    if has_reader:
        assert order[-2:] == ["capture_ended", "drain"]
    else:
        assert order[-1] == "capture_ended"
    if drain_fails:
        with pytest.raises(RuntimeError, match="quarantined"):
            binding.check_reader_health()
    else:
        binding.check_reader_health()
