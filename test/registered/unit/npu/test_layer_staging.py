"""Bounded layer-window data mapping, credit ownership and failed drain."""

import ast
import importlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

import test_prefill_layer_transfer as transfer_tests


@pytest.fixture
def modules(monkeypatch):
    loaded = transfer_tests.modules.__wrapped__(monkeypatch)
    name = "sglang.srt.hardware_backend.npu.layer_staging"
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    transfer_tests.load_module(name, transfer_tests.NPU / "layer_staging.py")
    return loaded


def case(modules):
    protocol, transfer = modules
    wire = importlib.import_module("sglang.srt.hardware_backend.npu.layer_protocol")
    staging = importlib.import_module("sglang.srt.hardware_backend.npu.layer_staging")
    record, sources, destinations = transfer_tests.inputs(*modules)
    record = replace(record, components=tuple(replace(item, bytes_per_page=256) for item in record.components))
    for index, component in enumerate(protocol.KvComponent):
        source_bytes = bytearray((position + index * 37) % 256 for position in range(1024))
        sources[component] = replace(sources[component], address=10000 + index * 10000,
                                     nbytes=1024, bytes_per_page=256, owner=source_bytes)
        destinations[component] = replace(destinations[component], address=100000 + index * 10000,
                                          nbytes=1024, bytes_per_page=256, owner=bytearray([255] * 1024))
    window = bytearray(32)
    copies = []
    drains = []

    def copy(region, offset, count):
        def complete():
            region.owner[offset:offset + count] = window[:count]
            copies.append((region.registration_id, offset, count))

        return SimpleNamespace(synchronize=complete, query=lambda: True)

    receiver = staging.LayerStagingReceiver(500000, 32, owner=window, copy=copy,
                                           drain=lambda: drains.append("drained"))
    lease = wire.LayerReceiveLease(
        5, "peer", 2, {4: destinations}, {component: (1, 3) for component in destinations},
        lambda *args: None, prefix=1, writer_endpoint=("p", 100), staging=receiver.manifest(),
    )
    operation = transfer.LayerTransferOperation(record, sources, destinations, ())
    payload = wire.ready_payload(5, "peer", lease.lease_id, record, operation)
    payload.update(reply_ip="p", reply_port=100)
    spans = transfer.build_layer_transfer_spans(record, sources, destinations, page_size=2)
    return SimpleNamespace(
        protocol=protocol, transfer=transfer, wire=wire, staging=staging, receiver=receiver,
        lease=lease, operation=operation, payload=payload, spans=spans, sources=sources,
        destinations=destinations, window=window, copies=copies, drains=drains,
    )


def chunk(context, **overrides):
    return {
        **context.payload, "component": "latent_k", "offset": 0, "bytes": 32,
        "chunk": "chunk-0", "staging_registration": context.receiver.registration, **overrides,
    }


def test_staged_client_transfers_three_components_through_one_fixed_window(modules):
    context = case(modules)
    transferred = []

    def transfer(block):
        source, destination, count = block
        assert destination == context.receiver.address and count <= 32
        assert context.receiver.active is not None
        source_region = next(region for region in context.sources.values()
                             if region.address <= source < region.address + region.nbytes)
        begin = source - source_region.address
        context.window[:count] = source_region.owner[begin:begin + count]
        transferred.append(block)
        return 0

    def rpc(header, message, timeout):
        assert timeout > 0
        if header == b"FSA_STAGE_ACQUIRE":
            return context.receiver.acquire(context.lease, message)
        result = context.receiver.commit(context.lease, message)
        assert len(context.copies) == len(transferred)
        return result

    context.staging.execute_staged_layer(
        context.operation, context.spans, context.receiver.manifest(), context.payload,
        rpc=rpc, transfer=transfer, timeout_s=10,
    )
    assert len(transferred) == 24
    assert len(context.window) == 32 and context.receiver.active is None
    assert context.operation.state == "transferred"
    for span in context.spans:
        source = context.sources[span.component]
        destination = context.destinations[span.component]
        begin = span.source_address - source.address
        assert destination.owner[span.destination_offset:span.destination_offset + span.nbytes] == (
            source.owner[begin:begin + span.nbytes]
        )
        assert destination.owner[:256] == bytearray([255] * 256)
    context.lease.receive(context.payload)
    assert context.lease.prefixes[4] == 3


def test_window_stays_owned_until_local_copy_event_completes(modules):
    context = case(modules)
    message = chunk(context)
    grant = context.receiver.acquire(context.lease, message)
    order = []

    def complete():
        assert context.receiver.active is not None
        assert context.receiver.acquire(context.lease, chunk(context, component="rope_k", chunk="other"))["busy"]
        order.append("copy_complete")

    context.receiver.copy = lambda *args: SimpleNamespace(synchronize=complete, query=lambda: True)
    result = context.receiver.commit(context.lease, {**message, "grant": grant["grant"]})
    assert order == ["copy_complete"] and result["committed"]
    assert context.receiver.active is None


def test_stale_grant_cannot_release_another_chunks_credit(modules):
    context = case(modules)
    first = chunk(context)
    grant = context.receiver.acquire(context.lease, first)
    assert context.receiver.acquire(context.lease, first) == grant
    completed = {**first, "grant": grant["grant"]}
    context.receiver.commit(context.lease, completed)
    second = chunk(context, offset=32, chunk="chunk-1")
    second_grant = context.receiver.acquire(context.lease, second)
    assert second_grant["grant"] > grant["grant"]
    assert context.receiver.commit(context.lease, completed)["committed"]
    assert context.receiver.active["grant"] == second_grant["grant"]
    with pytest.raises(ValueError, match="credit"):
        context.receiver.commit(context.lease, {**second, "grant": grant["grant"]})


def test_copy_failure_quarantines_window_until_transport_and_local_drain(modules):
    context = case(modules)
    message = chunk(context)
    grant = context.receiver.acquire(context.lease, message)

    def fail(*args):
        raise RuntimeError("copy enqueue failed after queued work")

    context.receiver.copy = fail
    with pytest.raises(RuntimeError, match="copy enqueue"):
        context.receiver.commit(context.lease, {**message, "grant": grant["grant"]})
    assert context.receiver.active is not None and context.lease.failed
    with pytest.raises(RuntimeError, match="remote writers"):
        context.receiver.abort(context.lease, transport_drained=False)
    context.receiver.drain = fail
    with pytest.raises(RuntimeError, match="copy enqueue"):
        context.receiver.abort(context.lease, transport_drained=True)
    assert context.receiver.active is not None
    context.receiver.drain = lambda: context.drains.append("drained")
    context.receiver.abort(context.lease, transport_drained=True)
    assert context.receiver.active is None and context.drains == ["drained"]
    context.receiver.forget(context.lease)
    assert not context.receiver.layer_owners and not context.receiver.last_commits


@pytest.mark.parametrize("change", [
    {"offset": 1}, {"offset": 120}, {"bytes": 33}, {"staging_registration": "old"},
    {"reply_port": 200}, {"lease": "old"},
])
def test_invalid_chunk_rejected_before_window_write(modules, change):
    context = case(modules)
    with pytest.raises(ValueError):
        context.receiver.acquire(context.lease, chunk(context, **change))
    assert context.receiver.active is None and not context.copies


def test_partial_staging_cannot_publish_a_complete_layer_or_change_range(modules):
    context = case(modules)
    message = chunk(context)
    grant = context.receiver.acquire(context.lease, message)
    context.receiver.commit(context.lease, {**message, "grant": grant["grant"]})
    with pytest.raises(RuntimeError, match="all staged bytes"):
        context.lease.receive(context.payload)
    with pytest.raises(ValueError, match="valid interval"):
        context.receiver.acquire(context.lease, chunk(context, offset=32, range=[1, 4]))
    assert context.lease.prefixes[4] == 1


def test_staging_manifest_hides_local_addresses_and_requires_version_three(modules):
    context = case(modules)
    manifest = context.wire.decode(context.lease.manifest())
    assert manifest["version"] == 3
    assert all(region["address"] == 1 for region in manifest["layers"]["4"].values())
    target = context.wire.parse_targets(context.lease.manifest(), room=5, session_id="peer",
                                        page_size=2, notify=lambda *args: None)[4]
    assert target.staging == context.receiver.manifest()
    manifest["version"] = 2
    with pytest.raises(ValueError, match="staging"):
        context.wire.parse_targets(context.wire.encode(manifest), room=5, session_id="peer",
                                   page_size=2, notify=lambda *args: None)


def test_staging_credit_timeout_preserves_quiescent_state_for_abort(modules):
    context = case(modules)
    with pytest.raises(TimeoutError):
        context.staging.execute_staged_layer(
            context.operation, context.spans, context.receiver.manifest(), context.payload,
            rpc=lambda *args: {"busy": True}, transfer=lambda block: pytest.fail("unexpected write"),
            timeout_s=0,
        )
    assert context.operation.state == "failed" and context.operation.transport_drained


def test_actual_sender_uses_staging_rpc_without_holding_control_lock(modules, monkeypatch):
    context = case(modules)
    methods = transfer_tests.sender_methods()
    sender_type = type("Sender", (), {name: methods[name] for name in ("send_layer_ready", "_execute_staged_layer")})
    sender = sender_type()
    sender.bootstrap_room = 5
    sender.conclude_state = None
    sender._layer_ready_operations = {}
    peer = SimpleNamespace(layer_manifest=context.lease.manifest(), endpoint="decode", dst_port=200)
    manager = transfer_tests.manager(local_ip="p", rank_port=100, transfer_infos={5: {"peer": peer}})
    sender.kv_mgr = manager

    def rpc(manager, peer, header, payload, timeout):
        assert not manager._layer_send_lock._is_owned()
        payload = {**payload, "reply_ip": "p", "reply_port": 100}
        if header == b"FSA_STAGE_ACQUIRE":
            return context.receiver.acquire(context.lease, payload)
        return context.receiver.commit(context.lease, payload)

    def transfer(session, blocks):
        assert manager._layer_send_lock._is_owned()
        assert session == "peer" and len(blocks) == 1
        source, destination, count = blocks[0]
        assert destination == context.receiver.address
        region = next(region for region in context.sources.values()
                      if region.address <= source < region.address + region.nbytes)
        offset = source - region.address
        context.window[:count] = region.owner[offset:offset + count]
        return 0

    monkeypatch.setattr(context.staging, "request_layer_staging", rpc)
    manager._transfer_data = transfer
    operation = sender.send_layer_ready(
        context.operation.record, source_regions=context.sources, destination_regions=context.destinations,
        destination_session_id="peer", page_size=2, source_ready_event=SimpleNamespace(query=lambda: True),
        staging=context.receiver.manifest(),
    )
    assert operation.state == "transferred" and operation.transport_drained
    assert operation.blocks == () and len(context.copies) == 24


def test_cancelled_staged_operation_does_not_overwrite_confirmed_drain(modules):
    context = case(modules)

    def abort(*args):
        context.operation.state = "drained"
        raise RuntimeError("aborted")

    with pytest.raises(RuntimeError, match="aborted"):
        context.staging.execute_staged_layer(
            context.operation, context.spans, context.receiver.manifest(), context.payload,
            rpc=abort, transfer=lambda block: pytest.fail("write after abort"), timeout_s=10,
        )
    assert context.operation.state == "drained"


def test_actual_manager_registers_window_but_not_staged_host_backing():
    source = transfer_tests.ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "MooncakeKVManager")
    method = next(node for node in manager.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_registerable_regions")
    namespace = {}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    pool = SimpleNamespace(
        kv_args=SimpleNamespace(kv_data_ptrs=[100, 200], kv_data_lens=[64, 64],
                                aux_data_ptrs=[400], aux_data_lens=[16],
                                state_data_ptrs=[[300]], state_data_lens=[[64]]),
        _layer_staged_host_regions={(100, 64), (200, 64)},
        _layer_staging=SimpleNamespace(address=500, window_bytes=32),
    )
    assert set(namespace["_registerable_regions"](pool)) == {(300, 64), (400, 16), (500, 32)}
