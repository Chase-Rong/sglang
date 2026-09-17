"""Layer bootstrap and receiver state tests; transport is exercised separately."""

import importlib
from dataclasses import replace

import pytest

import test_prefill_layer_transfer as transfer_tests

inputs = transfer_tests.inputs
sender_methods = transfer_tests.sender_methods


@pytest.fixture
def modules(monkeypatch):
    return transfer_tests.modules.__wrapped__(monkeypatch)


def wire_module():
    return importlib.import_module("sglang.srt.hardware_backend.npu.layer_protocol")


def lease_inputs(modules):
    record, sources, destinations = inputs(*modules)
    wire = wire_module()
    visible = []
    lease = wire.LayerReceiveLease(
        5, "decode-session", 2, {4: destinations},
        {component: (1, 3) for component in destinations},
        lambda *args: visible.append(args), prefix=1,
    )
    operation = modules[1].LayerTransferOperation(record, sources, destinations, ())
    payload = wire.ready_payload(5, "decode-session", lease.lease_id, record, operation)
    return lease, record, operation, payload, visible


def test_manifest_roundtrip_retains_address_kinds_and_independent_pages(modules):
    lease, record, operation, _, _ = lease_inputs(modules)
    component = modules[0].KvComponent.INDEXER_K
    lease.page_ids[component] = (0, 2)
    notifications = []
    targets = wire_module().parse_targets(
        lease.manifest(), room=5, session_id="decode-session", page_size=2,
        notify=lambda *args: notifications.append(args),
    )
    target = targets[4]
    assert target.page_ids[component] == (0, 2)
    assert target.page_ids[modules[0].KvComponent.LATENT_K] == (1, 3)
    assert target.regions[component].kind == modules[0].PdSourceKind.NPU_REGISTERED
    target.notify(record, operation)
    assert notifications == [(lease.lease_id, record, operation)]


@pytest.mark.parametrize("writer", [True, False])
def test_shared_host_manifest_sends_host_once_and_indexer_to_each_rank(modules, writer):
    lease, record, operation, _, _ = lease_inputs(modules)
    lease.shared_group_id = "tp-local-group"
    lease.shared_host_writer = writer
    wire = wire_module()
    manifest = lease.manifest()
    assert wire.decode(manifest)["version"] == 2
    target = wire.parse_targets(
        manifest, room=5, session_id="decode-session", page_size=2,
        notify=lambda *args: None,
    )[4]
    blocks = modules[1].build_layer_transfer_blocks(
        record, operation.sources, operation.destinations, page_size=2,
        transfer_components=target.transfer_components,
    )
    assert len(blocks) == (6 if writer else 2)
    if not writer:
        assert blocks == [(1240, 2224, 8), (1200, 2248, 8)]


def test_shared_host_receiver_rejects_old_wire_version(modules):
    lease, _, _, _, _ = lease_inputs(modules)
    lease.shared_group_id = "tp-local-group"
    wire = wire_module()
    manifest = wire.decode(lease.manifest())
    manifest["version"] = 1
    with pytest.raises(ValueError, match="ownership"):
        wire.parse_targets(wire.encode(manifest), room=5, session_id="decode-session",
                           page_size=2, notify=lambda *args: None)


def test_visibility_precedes_ack_and_duplicate_ready_is_idempotent(modules):
    lease, _, _, payload, visible = lease_inputs(modules)
    wire = wire_module()
    lease.receive(wire.decode(wire.encode(payload)))
    assert lease.prefixes[4] == 3 and visible == [(4, 1, 3)]
    lease.receive(payload)
    assert visible == [(4, 1, 3)]


@pytest.mark.parametrize("mutation", ["lease", "session", "version", "pages", "kind"])
def test_bad_manifest_is_rejected_before_transfer(modules, mutation):
    lease, _, _, _, _ = lease_inputs(modules)
    wire = wire_module()
    payload = wire.decode(lease.manifest())
    if mutation == "version":
        payload["version"] = 99
    elif mutation == "pages":
        payload["pages"]["latent_k"] = [1, 1000]
    elif mutation == "kind":
        payload["layers"]["4"]["indexer_k"]["kind"] = "host_registered"
    else:
        payload[mutation] = "" if mutation == "lease" else "stale-session"
    with pytest.raises(ValueError):
        wire.parse_targets(wire.encode(payload), room=5, session_id="decode-session",
                           page_size=2, notify=lambda *args: None)


@pytest.mark.parametrize("mutation", ["lease", "range", "registration", "request"])
def test_receiver_rejects_stale_or_inconsistent_ready(modules, mutation):
    lease, _, _, payload, visible = lease_inputs(modules)
    if mutation == "lease":
        payload["lease"] = "stale-lease"
    elif mutation == "range":
        payload["range"] = [2, 3]
    elif mutation == "registration":
        payload["registrations"]["latent_k"] = "stale-registration"
    else:
        lease.request_id = "another-request"
    with pytest.raises(ValueError):
        lease.receive(payload)
    assert visible == [] and lease.prefixes[4] == 1


def test_visibility_failure_does_not_publish_prefix(modules):
    lease, _, _, payload, _ = lease_inputs(modules)

    def fail(*args):
        raise RuntimeError("visibility did not complete")

    lease.visibility = fail
    with pytest.raises(RuntimeError, match="visibility"):
        lease.receive(payload)
    assert lease.prefixes[4] == 1 and not lease.completed


def test_second_chunk_uses_same_lease_with_a_new_slot_generation(modules):
    lease, record, operation, payload, visible = lease_inputs(modules)
    lease.receive(payload)
    second_token = replace(record.token, generation=record.token.generation + 1)
    second = replace(record, token=second_token, components=tuple(
        replace(entry, generation=second_token.generation,
                valid_range=modules[0].KvPageRange(3, 4)) for entry in record.components
    ))
    lease.receive(wire_module().ready_payload(5, "decode-session", lease.lease_id, second, operation))
    assert visible == [(4, 1, 3), (4, 3, 4)]


def test_sender_automatically_loads_targets_from_bootstrap(modules):
    from types import SimpleNamespace

    lease, _, _, _, _ = lease_inputs(modules)
    methods = sender_methods()
    sender_type = type("Sender", (), {name: methods[name] for name in (
        "require_prefill_layer_targets", "_load_negotiated_layer_targets",
        "configure_prefill_layer_targets",
    )})
    sender = sender_type()
    sender.bootstrap_room = 5
    sender._prefill_layer_targets = {}
    sender._layer_ready_operations = {}
    sender._notify_layer_receiver = lambda *args: None
    sender.kv_mgr = SimpleNamespace(
        kv_args=SimpleNamespace(page_size=2),
        transfer_infos={5: {"decode-session": SimpleNamespace(
            is_dummy=False, layer_manifest=lease.manifest(), required_dst_info_num=1
        )}},
    )
    sender.require_prefill_layer_targets([4])
    assert sender._prefill_layer_targets[4][0].session_id == "decode-session"


def test_replica_bootstrap_selects_only_the_non_dummy_writer(modules):
    wire = wire_module()
    peers = [{"is_dummy": True, "rank_ip": "peer-a", "rank_port": 10},
             {"is_dummy": False, "rank_ip": "peer-b", "rank_port": 11}]
    assert wire.select_layer_writer(peers) == ("peer-b", 11)
    peers[0]["is_dummy"] = False
    with pytest.raises(ValueError, match="one data writer"):
        wire.select_layer_writer(peers)


def test_receiver_rejects_ready_from_an_unselected_tp_rank(modules):
    lease, _, _, payload, visible = lease_inputs(modules)
    lease.writer_endpoint = ("selected", 10)
    payload.update(reply_ip="other", reply_port=10)
    with pytest.raises(ValueError, match="negotiated writer"):
        lease.receive(payload)
    assert not visible
    payload["reply_ip"] = "selected"
    lease.receive(payload)
    assert visible == [(4, 1, 3)]


@pytest.mark.parametrize("dummy_only", [False, True])
def test_bootstrap_fanout_and_dummy_rank_targets(modules, dummy_only):
    from types import SimpleNamespace

    lease, _, _, _, _ = lease_inputs(modules)
    methods = sender_methods()
    sender_type = type("Sender", (), {name: methods[name] for name in (
        "require_prefill_layer_targets", "_load_negotiated_layer_targets",
        "configure_prefill_layer_targets",
    )})
    sender = sender_type()
    sender.bootstrap_room = 5
    sender._prefill_layer_targets = {}
    sender._layer_ready_operations = {}
    sender._notify_layer_receiver = lambda *args: None
    peers = {}
    for session in ("decode-0", "decode-1"):
        lease.session_id = session
        peers[session] = SimpleNamespace(
            is_dummy=dummy_only, layer_manifest=lease.manifest(), required_dst_info_num=2,
        )
    sender.kv_mgr = SimpleNamespace(kv_args=SimpleNamespace(page_size=2), transfer_infos={5: peers})
    sender.require_prefill_layer_targets([4])
    assert len(sender._prefill_layer_targets[4]) == (0 if dummy_only else 2)
    assert sender._prefill_layer_dummy_only == dummy_only


def test_pp_writers_cover_disjoint_global_layer_intervals(modules):
    wire = wire_module()
    peers = [dict(is_dummy=False, rank_ip="stage-0", rank_port=10,
                  layer_start=0, layer_end=4),
             dict(is_dummy=False, rank_ip="stage-1", rank_port=11,
                  layer_start=4, layer_end=7),
             dict(is_dummy=True, rank_ip="tp-dummy", rank_port=12)]
    selected = wire.select_layer_writers(peers, range(7))
    assert selected == {layer: ("stage-0", 10) if layer < 4 else ("stage-1", 11)
                        for layer in range(7)}
    assert wire.select_layer_writers(peers[1:], range(4, 7)) == {
        layer: ("stage-1", 11) for layer in range(4, 7)
    }


@pytest.mark.parametrize("interval", [(3, 7), (5, 7), (None, None)])
def test_pp_writer_overlap_gap_and_missing_metadata_reject(modules, interval):
    peers = [dict(is_dummy=False, rank_ip="stage-0", rank_port=10,
                  layer_start=0, layer_end=4),
             dict(is_dummy=False, rank_ip="stage-1", rank_port=11,
                  layer_start=interval[0], layer_end=interval[1])]
    with pytest.raises(ValueError):
        wire_module().select_layer_writers(peers, range(7))


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("shared_writer", [None, False, True])
def test_pp_manifest_combines_writer_ownership_with_shared_and_staged_receive(
    modules, staged, shared_writer,
):
    lease, _, _, payload, visible = lease_inputs(modules)
    lease.layers[5] = lease.layers[4]
    lease.prefixes[5] = 1
    lease.layer_writers = {4: ("stage-0", 10), 5: ("stage-1", 11)}
    if staged:
        lease.staging = dict(window_bytes=32, registration="window")
    if shared_writer is not None:
        lease.shared_group_id = "tp-group"
        lease.shared_host_writer = shared_writer
    wire = wire_module()
    assert wire.decode(lease.manifest())["version"] == 4
    targets = wire.parse_targets(lease.manifest(), room=5, session_id="decode-session",
                                 page_size=2, notify=lambda *args: None)
    assert set(targets) == {4, 5}
    local_targets = wire.parse_targets(
        lease.manifest(), room=5, session_id="decode-session", page_size=2,
        notify=lambda *args: None, writer_endpoint=("stage-1", 11),
    )
    assert set(local_targets) == {5}
    assert targets[4].staging == lease.staging
    if shared_writer is False:
        assert targets[4].transfer_components == {modules[0].KvComponent.INDEXER_K}
    payload.update(reply_ip="stage-1", reply_port=11)
    with pytest.raises(ValueError, match="layer writer"):
        lease.receive(payload)
    assert not visible
    payload.update(reply_ip="stage-0", reply_port=10)
    lease.validate_notification(payload)
    first = dict(reply_ip="stage-0", reply_port=10)
    assert not lease.note_writer_drain(first)
    assert not lease.note_writer_drain(first)
    with pytest.raises(ValueError, match="negotiated writer"):
        lease.note_writer_drain(dict(reply_ip="dummy", reply_port=12))
    assert lease.note_writer_drain(dict(reply_ip="stage-1", reply_port=11))


def test_dummy_cp_runtime_needs_no_receiver_metadata(modules):
    from types import SimpleNamespace

    methods = sender_methods()
    sender = SimpleNamespace(kv_mgr=SimpleNamespace(is_dummy_cp_rank=True),
                             _prefill_layer_targets={})
    methods["require_prefill_layer_targets"](sender, iter([4, 5]))
    assert sender._prefill_layer_targets == {4: (), 5: ()}
    assert sender._prefill_layer_dummy_only


def test_actual_abort_handler_waits_for_all_pp_writers_before_staging_drain(modules):
    import ast
    from types import SimpleNamespace

    lease, _, _, _, _ = lease_inputs(modules)
    lease.layer_writers = {4: ("stage-0", 10), 5: ("stage-1", 11)}
    source = transfer_tests.ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "MooncakeKVManager")
    method = next(node for node in manager.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_handle_layer_message")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    drains = []
    manager = SimpleNamespace(
        _layer_leases={5: lease},
        _layer_staging=SimpleNamespace(abort=lambda lease, **kwargs: drains.append(kwargs)),
    )
    payload = dict(room=5, session=lease.session_id, lease=lease.lease_id)
    handler = namespace["_handle_layer_message"]
    for stage in (0, 0, 1):
        handler(manager, [b"FSA_LAYER_ABORT_ACK", wire_module().encode(
            {**payload, "reply_ip": f"stage-{stage}", "reply_port": 10 + stage}
        )])
        assert lease.drained == (stage == 1)
        assert len(drains) == (1 if stage == 1 else 0)
    assert drains == [{"transport_drained": True}]
