import ast
import importlib.util
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


ROOT = Path(__file__).parents[4]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    for package in (
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
    ):
        module = types.ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    prefix = "sglang.srt.hardware_backend.npu"
    for name in (f"{prefix}.prefill_offload", f"{prefix}.layer_transfer"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    protocol = load_module(f"{prefix}.prefill_offload", NPU / "prefill_offload.py")
    transfer = load_module(f"{prefix}.layer_transfer", NPU / "layer_transfer.py")
    name = f"{prefix}.layer_protocol"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    load_module(name, NPU / "layer_protocol.py")
    return protocol, transfer


def inputs(protocol, transfer):
    token = protocol.LayerSlotToken(1, "req", 4, 7)
    descriptors, sources, destinations = [], {}, {}
    for index, component in enumerate(protocol.KvComponent):
        descriptors.append(
            protocol.PdComponentDescriptor(
                request_id="req",
                logical_layer=4,
                generation=7,
                component=component,
                source_kind=protocol.PdSourceKind.NPU_REGISTERED,
                source_slot=1,
                source_page_ids=(2, 0),
                destination_page_ids=(1, 3),
                valid_range=protocol.KvPageRange(1, 3),
                bytes_per_page=16,
            )
        )
        sources[component] = transfer.RegisteredLayerRegion(
            1000 + 100 * index,
            64,
            16,
            f"source-{index}",
            protocol.PdSourceKind.NPU_REGISTERED,
            "bf16",
            object(),
        )
        destinations[component] = transfer.RegisteredLayerRegion(
            2000 + 100 * index,
            64,
            16,
            f"destination-{index}",
            protocol.PdSourceKind.NPU_REGISTERED
            if component == protocol.KvComponent.INDEXER_K
            else protocol.PdSourceKind.HOST_REGISTERED,
            "bf16",
            object(),
        )
    return protocol.LayerReadyRecord(token, tuple(descriptors)), sources, destinations


def test_transfer_copies_only_partial_page_range_and_independent_components(modules):
    protocol, transfer = modules
    record, sources, destinations = inputs(*modules)
    blocks = transfer.build_layer_transfer_blocks(
        record, sources, destinations, page_size=2
    )
    assert blocks == [
        (1040, 2024, 8),
        (1000, 2048, 8),
        (1140, 2124, 8),
        (1100, 2148, 8),
        (1240, 2224, 8),
        (1200, 2248, 8),
    ]


def test_transfer_rejects_unregistered_bounds_or_wrong_destination_kind(modules):
    protocol, transfer = modules
    record, sources, destinations = inputs(*modules)
    component = protocol.KvComponent.LATENT_K
    destinations[component] = transfer.RegisteredLayerRegion(
        2000,
        32,
        16,
        "short-region",
        protocol.PdSourceKind.HOST_REGISTERED,
        "bf16",
        object(),
    )
    with pytest.raises(ValueError, match="exceeds"):
        transfer.build_layer_transfer_blocks(record, sources, destinations, page_size=2)
    destinations[component] = sources[component]
    with pytest.raises(ValueError, match="address-space"):
        transfer.build_layer_transfer_blocks(record, sources, destinations, page_size=2)


def sender_methods():
    path = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sender = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MooncakeKVSender"
    )
    methods = [
        node
        for node in sender.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "send_layer_ready",
            "_execute_staged_layer",
            "acknowledge_layer_ready",
            "forget_layer_ready",
            "poll",
            "configure_prefill_layer_targets",
            "require_prefill_layer_targets",
            "prefill_layer_operations",
            "publish_prefill_layer",
            "send",
            "_load_negotiated_layer_targets",
        )
    ]
    for method in methods:
        method.decorator_list = []
    namespace = {
        "np": np,
        "envs": SimpleNamespace(SGLANG_NPU_PREFILL_LAYER_TIMEOUT_S=SimpleNamespace(get=lambda: 10)),
        "KVPoll": SimpleNamespace(
            Success="success",
            Failed="failed",
            Transferring="transferring",
            Bootstrapping="bootstrapping",
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        *methods,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


def manager(**kwargs):
    defaults = dict(_layer_send_lock=threading.RLock(), _layer_operations={},
                    check_status=lambda room: "transferring",
                    update_status=lambda *args: None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_real_sender_entry_keeps_owners_until_transfer_and_receiver_ack(modules):
    record, sources, destinations = inputs(*modules)
    calls = []
    methods = sender_methods()
    sender = SimpleNamespace(
        conclude_state=None,
        bootstrap_room=1,
        _layer_ready_operations={},
        kv_mgr=manager(_transfer_data=lambda *args: calls.append(args) or 0),
    )
    event = SimpleNamespace(query=lambda: False)
    kwargs = dict(
        source_regions=sources,
        destination_regions=destinations,
        destination_session_id="peer",
        page_size=2,
        source_ready_event=event,
    )
    with pytest.raises(modules[0].SlotBusyError):
        methods["send_layer_ready"](sender, record, **kwargs)
    assert not calls and not sender._layer_ready_operations
    event.query = lambda: True
    operation = methods["send_layer_ready"](sender, record, **kwargs)
    assert operation.state == "transferred" and len(calls) == 1
    with pytest.raises(RuntimeError, match="ACK"):
        methods["forget_layer_ready"](sender, record.token, "peer")
    with pytest.raises(RuntimeError, match="already submitted"):
        methods["send_layer_ready"](sender, record, **kwargs)
    methods["acknowledge_layer_ready"](sender, record.token, "peer")
    methods["forget_layer_ready"](sender, record.token, "peer")
    assert not sender._layer_ready_operations


def test_whole_request_success_waits_for_layer_receiver_ack(modules):
    record, sources, destinations = inputs(*modules)
    methods = sender_methods()
    sender = SimpleNamespace(
        conclude_state=None,
        bootstrap_room=1,
        _layer_ready_operations={},
        kv_mgr=manager(
            _transfer_data=lambda *args: 0,
            check_status=lambda room: "success",
            _staging_outstanding={},
        ),
        trace_ctx=SimpleNamespace(trace_req_finish=lambda: None),
    )
    methods["send_layer_ready"](
        sender,
        record,
        source_regions=sources,
        destination_regions=destinations,
        destination_session_id="peer",
        page_size=2,
        source_ready_event=SimpleNamespace(query=lambda: True),
    )
    assert methods["poll"](sender) == "transferring"
    methods["acknowledge_layer_ready"](sender, record.token, "peer")
    assert methods["poll"](sender) == "success"
    methods["forget_layer_ready"](sender, record.token, "peer")
    assert not sender._layer_ready_operations


def test_failed_transfer_does_not_release_registration_owners(modules):
    record, sources, destinations = inputs(*modules)
    methods = sender_methods()
    sender = SimpleNamespace(
        conclude_state=None,
        bootstrap_room=1,
        _layer_ready_operations={},
        kv_mgr=manager(_transfer_data=lambda *args: -1),
    )
    with pytest.raises(RuntimeError, match="status -1"):
        methods["send_layer_ready"](
            sender,
            record,
            source_regions=sources,
            destination_regions=destinations,
            destination_session_id="peer",
            page_size=2,
            source_ready_event=SimpleNamespace(query=lambda: True),
        )
    operation = sender._layer_ready_operations[record.token, "peer"]
    assert operation.state == "failed" and operation.sources
    with pytest.raises(RuntimeError):
        methods["forget_layer_ready"](sender, record.token, "peer")
    operation.confirm_failed_drain()
    methods["forget_layer_ready"](sender, record.token, "peer")
    assert methods["poll"](sender) == "failed"


@pytest.mark.parametrize("layer_offset", [0, 64])
def test_prefill_publisher_uses_registered_device_sources_and_peer_page_leases(modules, layer_offset):
    protocol, transfer = modules
    methods = sender_methods()
    sender_type = type(
        "Sender",
        (),
        {
            name: value
            for name, value in methods.items()
            if isinstance(value, types.FunctionType)
        },
    )
    sender = sender_type()
    sender.bootstrap_room = 1
    sender.conclude_state = None
    sender._layer_ready_operations = {}
    sender._prefill_layer_targets = {}
    tensors, regions = {}, {}
    for index, component in enumerate(protocol.KvComponent):
        width = (8, 4, 6)[index]
        address = 1000 + index * 1000
        tensors[component] = SimpleNamespace(
            shape=(3, 2, 1, width),
            dtype="bf16",
            nbytes=6 * width * 2,
            data_ptr=lambda address=address: address,
            element_size=lambda: 2,
        )
        regions[component] = transfer.RegisteredLayerRegion(
            10000 + index * 1000,
            8 * width * 2,
            2 * width * 2,
            f"receiver-{index}",
            protocol.PdSourceKind.NPU_REGISTERED
            if component == protocol.KvComponent.INDEXER_K
            else protocol.PdSourceKind.HOST_REGISTERED,
            "bf16",
            object(),
        )
    calls = []
    sender.kv_mgr = manager(
        kv_args=SimpleNamespace(prefill_layerwise_aux_kv=False),
        _registerable_regions=lambda: [
            (tensor.data_ptr(), tensor.nbytes) for tensor in tensors.values()
        ],
        _transfer_data=lambda session, blocks: calls.append((session, blocks)) or 0,
    )
    sender.configure_prefill_layer_targets(
        4 + layer_offset,
        [
            transfer.PrefillLayerTarget(
                "peer",
                regions,
                {component: (0, 1, 2, 3) for component in protocol.KvComponent},
                lambda record, operation: operation.acknowledge(),
            )
        ],
    )
    work = protocol.LayerWork(
        protocol.LayerSlotToken(0, "req", 4, 1),
        protocol.HostPageAllocation("req", (2, 3), 2, 1),
        protocol.KvPageRange(0, 3),
    )
    binding = SimpleNamespace(
        latent_k=tensors[protocol.KvComponent.LATENT_K],
        rope_k=tensors[protocol.KvComponent.ROPE_K],
        indexer_k=tensors[protocol.KvComponent.INDEXER_K],
        source_page_ids=(1, 2),
    )
    operations = sender.publish_prefill_layer(
        work, binding, SimpleNamespace(query=lambda: True), layer_offset=layer_offset,
    )
    assert len(calls) == len(operations) == 1
    assert operations[0].state == "acknowledged"
    assert work.token.logical_layer == 4
    assert operations[0].record.token.logical_layer == 4 + layer_offset
    assert sum(block[2] for block in calls[0][1]) == 3 * (8 + 4 + 6) * 2
    assert all(
        descriptor.destination_page_ids == (0, 1)
        for descriptor in operations[0].record.components
    )


@pytest.mark.parametrize("last_chunk", [False, True])
def test_layerwise_whole_request_send_does_not_reread_reused_kv_slots(last_chunk):
    calls = []
    methods = sender_methods()
    indices = np.array([1, 2], dtype=np.int32)
    sender = SimpleNamespace(
        _prepare_send_indices=lambda *args: (indices, slice(0, 2), last_chunk, False),
        _record_transfer_indices=lambda *args: None,
        _layer_ready_operations={"layer": SimpleNamespace(state="acknowledged")},
        kv_mgr=SimpleNamespace(
            kv_args=SimpleNamespace(prefill_layerwise_offload=True),
            add_transfer_request=lambda *args, **kwargs: calls.append((args, kwargs)),
        ),
        bootstrap_room=1,
        aux_index=9,
        trace_ctx=SimpleNamespace(copy_for_thread=lambda: None),
    )
    methods["send"](sender, indices, state_indices=[indices], num_kv_tokens=3)
    assert len(calls) == int(last_chunk)
    if last_chunk:
        args, kwargs = calls[0]
        assert args[1].size == 0 and args[2] == slice(0, 0)
        assert kwargs["aux_index"] == 9
        assert np.array_equal(kwargs["state_indices"][0], indices)
        assert kwargs["num_kv_tokens"] == 0
