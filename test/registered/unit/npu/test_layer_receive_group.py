"""Target/draft lease namespaces and shared staging inventory."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest

import test_layer_receive_protocol as protocol_tests


@pytest.fixture
def inputs(monkeypatch):
    modules = protocol_tests.modules.__wrapped__(monkeypatch)
    wire = protocol_tests.wire_module()
    source = protocol_tests.transfer_tests.ROOT / "python/sglang/srt/hardware_backend/npu/layer_receive_group.py"
    config_name = "sglang.srt.hardware_backend.npu.offload_config"
    config_spec = importlib.util.spec_from_file_location(config_name, source.with_name("offload_config.py"))
    config_module = importlib.util.module_from_spec(config_spec)
    monkeypatch.setitem(sys.modules, config_name, config_module)
    config_spec.loader.exec_module(config_module)
    name = "sglang.srt.hardware_backend.npu.layer_receive_group"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    lease, _, _, _, _ = protocol_tests.lease_inputs(modules)
    events = []

    def pool(start, count, name):
        def receive(room, session, host_pages, device_pages, *, prefix):
            pages = {component: tuple(device_pages if component == modules[0].KvComponent.INDEXER_K
                                       else host_pages) for component in lease.page_ids}
            return wire.LayerReceiveLease(
                room, session, 2, {layer: lease.layers[4] for layer in range(start, start + count)},
                pages, lambda *args: events.append((name, *args)), prefix=prefix,
            )

        return SimpleNamespace(start_layer=start, layer_num=count, page_size=2,
                                size=16, kv_lora_rank=512, qk_rope_head_dim=64,
                                index_head_dim=128, store_dtype="bf16",
                                decode_host_backing=SimpleNamespace(pool=SimpleNamespace(device_id=0)),
                                layer_receive_lease=receive)

    return module, pool, events


def test_draft_local_layer_zero_is_distinct_from_target_layer_zero(inputs):
    module, pool, events = inputs
    group = module.LayerReceivePoolGroup(pool(0, 4, "target"), pool(0, 1, "draft"), 4)
    lease = group.layer_receive_lease(5, "session", (1, 3), (0, 2), prefix=1)
    assert set(lease.layers) == {0, 1, 2, 3, 4}
    assert len(group.host_backings) == 2
    lease.visibility(4, 1, 3)
    lease.visibility(0, 1, 3)
    assert events == [("draft", 0, 1, 3), ("target", 0, 1, 3)]


def test_nonzero_draft_local_start_is_mapped_to_the_negotiated_base(inputs):
    module, pool, _ = inputs
    group = module.LayerReceivePoolGroup(pool(4, 4, "target"), pool(16, 2, "draft"), 8)
    lease = group.layer_receive_lease(5, "session", (1, 3), (0, 2), prefix=1)
    assert set(lease.layers) == set(range(4, 10))


@pytest.mark.parametrize("mismatch", ["overlap", "device", "pages", "host", "writer"])
def test_incompatible_draft_receive_inventory_rejects_before_publication(inputs, mismatch):
    module, pool, _ = inputs
    target, draft = pool(0, 4, "target"), pool(0, 1, "draft")
    base = 4
    if mismatch == "overlap":
        base = 3
    elif mismatch == "device":
        draft.decode_host_backing.pool.device_id = 1
    elif mismatch == "pages":
        draft.page_size = 4
    elif mismatch == "host":
        draft.decode_host_backing = None
    else:
        draft.decode_host_backing.pool.shared_group = SimpleNamespace(
            writer_rank=0, group=SimpleNamespace(ranks=[0, 1]),
        )
    with pytest.raises(ValueError):
        module.LayerReceivePoolGroup(target, draft, base)
