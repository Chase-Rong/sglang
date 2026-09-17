import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "prefill_offload.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prefill_offload_under_test", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

KvComponent = _MODULE.KvComponent
KvPageRange = _MODULE.KvPageRange
LayerComponentManifest = _MODULE.LayerComponentManifest
HostKvPagePool = _MODULE.HostKvPagePool
LayerReadyRecord = _MODULE.LayerReadyRecord
LayerReadyTransfer = _MODULE.LayerReadyTransfer
LayerReadyTransportAdapter = _MODULE.LayerReadyTransportAdapter
LayerSlotToken = _MODULE.LayerSlotToken
PdComponentDescriptor = _MODULE.PdComponentDescriptor
PdTransferState = _MODULE.PdTransferState
PdSourceKind = _MODULE.PdSourceKind
PrefillLayerwiseOffload = _MODULE.PrefillLayerwiseOffload
PrefillLayerSlotManager = _MODULE.PrefillLayerSlotManager
SlotBusyError = _MODULE.SlotBusyError
StaleGenerationError = _MODULE.StaleGenerationError


def complete(manager, token, pd_readers=0):
    manager.set_pd_readers(token, pd_readers)
    manager.mark_host_saved(token)
    manager.mark_attention_done(token)
    if pd_readers:
        manager.mark_pd_read_done(token, pd_readers)


def test_four_layers_reuse_two_slots_after_all_completion_gates():
    manager = PrefillLayerSlotManager(logical_layers=4, physical_slots=2)
    first = [manager.acquire("request-0", layer) for layer in (0, 1)]

    with pytest.raises(SlotBusyError):
        manager.acquire("request-0", 2)

    complete(manager, first[0], pd_readers=2)
    complete(manager, first[1], pd_readers=1)
    manager.release(first[0])
    manager.release(first[1])

    second = [manager.acquire("request-0", layer) for layer in (2, 3)]
    assert [token.slot_id for token in second] == [0, 1]
    assert second[0].generation == first[0].generation + 1


def test_pd_completion_is_required_before_reuse():
    manager = PrefillLayerSlotManager(logical_layers=2, physical_slots=1)
    token = manager.acquire("request-0", 0)
    manager.set_pd_readers(token, 1)
    manager.mark_host_saved(token)
    manager.mark_attention_done(token)

    with pytest.raises(SlotBusyError):
        manager.acquire("request-1", 1)

    manager.mark_pd_read_done(token)
    manager.release(token)
    replacement = manager.acquire("request-1", 1)
    assert replacement.generation == token.generation + 1


def test_failed_transfer_can_be_discarded_but_not_reused_by_stale_token():
    manager = PrefillLayerSlotManager(logical_layers=2, physical_slots=1)
    token = manager.acquire("request-0", 0)
    manager.set_pd_readers(token, 1)
    manager.mark_failed(token)

    with pytest.raises(SlotBusyError):
        manager.release(token)

    manager.mark_pd_read_done(token)
    with pytest.raises(SlotBusyError):
        manager.release(token)
    manager.mark_failed_local_work_drained(token)
    manager.release(token)
    replacement = manager.acquire("request-1", 1)

    with pytest.raises(StaleGenerationError):
        manager.mark_host_saved(token)
    assert replacement.generation == token.generation + 1


def test_component_manifest_preserves_partial_ranges_and_separate_layouts():
    manifest = LayerComponentManifest()
    manifest.publish(KvComponent.LATENT_K, KvPageRange(0, 130))
    manifest.publish(KvComponent.ROPE_K, KvPageRange(0, 130))
    manifest.publish(KvComponent.INDEXER_K, KvPageRange(0, 128))

    assert manifest.get(KvComponent.LATENT_K).end_token == 130
    assert manifest.get(KvComponent.INDEXER_K).end_token == 128
    assert manifest.components() == {
        KvComponent.LATENT_K,
        KvComponent.ROPE_K,
        KvComponent.INDEXER_K,
    }


def test_host_pool_allocates_pages_and_preserves_partial_tail():
    pool = HostKvPagePool(page_count=4, page_size=128)
    allocation = pool.allocate("request-0", token_count=257)
    assert allocation.page_ids == (0, 1, 2)
    assert allocation.page_for_token(256) == 2
    assert pool.free_page_count == 1

    pool.release(allocation)
    assert pool.free_page_count == 4


def test_stale_host_allocation_cannot_release_reused_request_pages():
    pool = HostKvPagePool(page_count=2, page_size=128)
    old = pool.allocate("request-0", 128)
    pool.release(old)
    current = pool.allocate("request-0", 128)
    assert old.page_ids == current.page_ids
    with pytest.raises(ValueError, match="stale"):
        pool.release(old)
    assert pool.allocation("request-0") is current


def test_transfer_can_complete_synchronously_during_publication():
    manager = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    token = manager.acquire("request-0", 0)
    transfer = LayerReadyTransfer(lambda record: transfer.complete(record.token))
    transfer.publish(LayerReadyRecord(token, ()))
    assert transfer.state(token) == PdTransferState.COMPLETED
    transfer.forget(token)
    with pytest.raises(StaleGenerationError):
        transfer.complete(token)


def test_layer_ready_transport_requires_explicit_ack_before_forget():
    transport = LayerReadyTransportAdapter(lambda record: "handle")
    token = LayerSlotToken(0, "request-0", 0, 1)
    record = LayerReadyRecord(token, ())
    assert transport.publish(record) == "handle"
    assert transport.state(token) == "published"
    with pytest.raises(SlotBusyError):
        transport.forget(token)
    assert transport.acknowledge(token) == "handle"
    assert transport.state(token) == "completed"
    assert transport.forget(token) == "handle"


def test_layer_ready_transport_rejects_duplicate_or_stale_ack():
    transport = LayerReadyTransportAdapter(lambda record: "handle")
    token = LayerSlotToken(0, "request-0", 0, 1)
    transport.publish(LayerReadyRecord(token, ()))
    transport.acknowledge(token, success=False)
    with pytest.raises(ValueError, match="acknowledged twice"):
        transport.acknowledge(token)
    transport.forget(token)
    with pytest.raises(StaleGenerationError):
        transport.acknowledge(token)


def test_publication_exception_keeps_failed_generation_until_drained():
    manager = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    token = manager.acquire("request-0", 0)

    def send(record):
        raise RuntimeError("transport failed")

    transfer = LayerReadyTransfer(send)
    with pytest.raises(RuntimeError, match="transport failed"):
        transfer.publish(LayerReadyRecord(token, ()))
    assert transfer.state(token) == PdTransferState.FAILED


def test_invalid_layer_range_does_not_acquire_a_slot():
    slots = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    host_pool = HostKvPagePool(page_count=2, page_size=128)
    coordinator = PrefillLayerwiseOffload(slots, host_pool)
    coordinator.begin_request("request-0", 128)
    with pytest.raises(ValueError, match="valid token count"):
        coordinator.begin_layer("request-0", 0, 1, KvPageRange(0, 2))
    assert slots.active_tokens() == ()


def test_layerwise_lifecycle_publishes_npu_source_without_raw_pointers():
    slot_manager = PrefillLayerSlotManager(logical_layers=4, physical_slots=2)
    host_pool = HostKvPagePool(page_count=8, page_size=128)
    host_pool.allocate("request-0", token_count=257)
    restored = []
    published = []
    coordinator = PrefillLayerwiseOffload(
        slot_manager,
        host_pool,
        restore=restored.append,
        publish=published.append,
    )

    work = coordinator.begin_layer(
        "request-0",
        logical_layer=0,
        token_count=257,
        valid_range=KvPageRange(0, 257),
        pd_readers=1,
    )
    assert restored == [work]
    coordinator.mark_kv_written(work)
    record = coordinator.publish_layer_ready(
        work,
        [
            PdComponentDescriptor(
                request_id="request-0",
                logical_layer=0,
                generation=work.token.generation,
                component=KvComponent.LATENT_K,
                source_kind=PdSourceKind.NPU_REGISTERED,
                source_slot=work.token.slot_id,
                source_page_ids=(10, 11, 12),
                destination_page_ids=work.host_allocation.page_ids,
                valid_range=KvPageRange(0, 257),
                bytes_per_page=4096,
            )
        ],
    )
    assert isinstance(record, LayerReadyRecord)
    assert published == [record]
    assert (
        record.component(KvComponent.LATENT_K).source_kind
        == PdSourceKind.NPU_REGISTERED
    )

    with pytest.raises(SlotBusyError):
        coordinator.release_layer(work)
    coordinator.start_host_save(work)
    coordinator.complete_host_save(work)
    coordinator.mark_attention_done(work)
    coordinator.mark_pd_read_done(work)
    coordinator.release_layer(work)


def test_layerwise_save_and_pd_publication_require_written_kv():
    slot_manager = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    host_pool = HostKvPagePool(page_count=2, page_size=128)
    host_pool.allocate("request-0", token_count=1)
    coordinator = PrefillLayerwiseOffload(slot_manager, host_pool)
    work = coordinator.begin_layer("request-0", 0, 1, KvPageRange(0, 1), pd_readers=0)

    with pytest.raises(RuntimeError, match="written KV"):
        coordinator.start_host_save(work)
    with pytest.raises(RuntimeError, match="written KV"):
        coordinator.publish_layer_ready(work, [])


def test_layer_ready_transfer_ack_is_separate_from_publication():
    slot_manager = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    host_pool = HostKvPagePool(page_count=2, page_size=128)
    host_pool.allocate("request-0", token_count=1)
    sent = []
    transfer = LayerReadyTransfer(sent.append)
    coordinator = PrefillLayerwiseOffload(slot_manager, host_pool, pd_transfer=transfer)
    work = coordinator.begin_layer("request-0", 0, 1, KvPageRange(0, 1), pd_readers=1)
    coordinator.mark_kv_written(work)
    record = coordinator.publish_layer_ready(work, [])

    assert sent == [record]
    assert transfer.state(work.token) == PdTransferState.PUBLISHED
    coordinator.acknowledge_pd_read(work)
    assert transfer.state(work.token) == PdTransferState.COMPLETED

    coordinator.start_host_save(work)
    coordinator.complete_host_save(work)
    coordinator.mark_attention_done(work)
    coordinator.release_layer(work)


def test_failed_pd_ack_blocks_until_reader_is_drained_then_allows_release():
    slot_manager = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    host_pool = HostKvPagePool(page_count=2, page_size=128)
    host_pool.allocate("request-0", token_count=1)
    transfer = LayerReadyTransfer(lambda record: None)
    coordinator = PrefillLayerwiseOffload(slot_manager, host_pool, pd_transfer=transfer)
    work = coordinator.begin_layer("request-0", 0, 1, KvPageRange(0, 1), pd_readers=1)
    coordinator.mark_kv_written(work)
    coordinator.publish_layer_ready(work, [])
    coordinator.acknowledge_pd_read(work, success=False)
    coordinator.start_host_save(work)
    coordinator.complete_host_save(work)
    coordinator.mark_attention_done(work)
    coordinator.release_layer(work)


@pytest.mark.parametrize("first_success", [True, False])
def test_multiple_pd_readers_must_all_drain_before_release(first_success):
    slots = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    host_pool = HostKvPagePool(page_count=2, page_size=128)
    transfer = LayerReadyTransfer(lambda record: None)
    coordinator = PrefillLayerwiseOffload(slots, host_pool, pd_transfer=transfer)
    coordinator.begin_request("request-0", 1)
    work = coordinator.begin_layer("request-0", 0, 1, KvPageRange(0, 1), pd_readers=2)
    coordinator.mark_kv_written(work)
    coordinator.start_host_save(work)
    coordinator.complete_host_save(work)
    coordinator.mark_attention_done(work)
    coordinator.publish_layer_ready(work, [])
    coordinator.acknowledge_pd_read(work, success=first_success)
    with pytest.raises(SlotBusyError):
        coordinator.release_layer(work)
    coordinator.acknowledge_pd_read(work)
    coordinator.release_layer(work)
    assert slots.active_tokens() == ()
    with pytest.raises(StaleGenerationError):
        transfer.state(work.token)


def test_two_query_chunks_reuse_layer_slots_with_generation_changes():
    slot_manager = PrefillLayerSlotManager(logical_layers=4, physical_slots=2)
    host_pool = HostKvPagePool(page_count=16, page_size=128)
    allocation = host_pool.allocate("request-0", token_count=257)
    coordinator = PrefillLayerwiseOffload(slot_manager, host_pool)

    generations = []
    for chunk_start, chunk_end in ((0, 128), (128, 257)):
        for logical_layer in range(4):
            work = coordinator.begin_layer(
                "request-0",
                logical_layer,
                token_count=chunk_end,
                valid_range=KvPageRange(chunk_start, chunk_end),
                pd_readers=1,
            )
            generations.append(work.token.generation)
            coordinator.mark_kv_written(work)
            coordinator.publish_layer_ready(
                work,
                [
                    PdComponentDescriptor(
                        request_id="request-0",
                        logical_layer=logical_layer,
                        generation=work.token.generation,
                        component=KvComponent.INDEXER_K,
                        source_kind=PdSourceKind.NPU_REGISTERED,
                        source_slot=work.token.slot_id,
                        source_page_ids=(work.token.slot_id,),
                        destination_page_ids=(allocation.page_ids[0],),
                        valid_range=KvPageRange(chunk_start, chunk_end),
                        bytes_per_page=1024,
                    )
                ],
            )
            coordinator.acknowledge_pd_read(work)
            coordinator.start_host_save(work)
            coordinator.complete_host_save(work)
            coordinator.mark_attention_done(work)
            coordinator.release_layer(work)

    assert generations == [1, 1, 2, 2, 3, 3, 4, 4]


def test_out_of_order_pd_ack_does_not_release_the_wrong_generation():
    slot_manager = PrefillLayerSlotManager(logical_layers=2, physical_slots=1)
    host_pool = HostKvPagePool(page_count=4, page_size=128)
    host_pool.allocate("request-0", token_count=1)
    coordinator = PrefillLayerwiseOffload(slot_manager, host_pool)
    first = coordinator.begin_layer("request-0", 0, 1, KvPageRange(0, 1), pd_readers=1)
    coordinator.mark_kv_written(first)
    coordinator.publish_layer_ready(first, [])
    coordinator.mark_attention_done(first)
    coordinator.start_host_save(first)
    coordinator.complete_host_save(first)

    with pytest.raises(SlotBusyError):
        slot_manager.acquire("request-0", 1)

    coordinator.acknowledge_pd_read(first)
    coordinator.release_layer(first)
    second = coordinator.begin_layer("request-0", 1, 1, KvPageRange(0, 1), pd_readers=0)
    with pytest.raises(StaleGenerationError):
        coordinator.mark_attention_done(first)
    assert second.token.generation == first.token.generation + 1


def test_cancelled_layer_keeps_host_allocation_until_explicit_release():
    slot_manager = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    host_pool = HostKvPagePool(page_count=2, page_size=128)
    allocation = host_pool.allocate("request-0", token_count=1)
    coordinator = PrefillLayerwiseOffload(slot_manager, host_pool)
    work = coordinator.begin_layer("request-0", 0, 1, KvPageRange(0, 1), pd_readers=0)
    coordinator.fail_layer(work)
    with pytest.raises(SlotBusyError):
        coordinator.release_layer(work)
    slot_manager.mark_failed_local_work_drained(work.token)
    coordinator.release_layer(work)
    assert host_pool.allocation("request-0") == allocation
    host_pool.release(allocation)
    assert host_pool.free_page_count == 2


def test_request_allocation_is_released_only_after_layer_release():
    slot_manager = PrefillLayerSlotManager(logical_layers=2, physical_slots=1)
    host_pool = HostKvPagePool(page_count=4, page_size=128)
    coordinator = PrefillLayerwiseOffload(slot_manager, host_pool)
    allocation = coordinator.begin_request("request-0", token_count=129)
    work = coordinator.begin_layer(
        "request-0", 0, 129, KvPageRange(0, 129), pd_readers=0
    )
    coordinator.mark_kv_written(work)
    coordinator.start_host_save(work)
    coordinator.complete_host_save(work)
    coordinator.mark_attention_done(work)

    with pytest.raises(SlotBusyError):
        coordinator.end_request("request-0")

    coordinator.release_layer(work)
    coordinator.end_request("request-0")
    assert allocation.page_ids == (0, 1)
    assert host_pool.free_page_count == 4


def test_chunked_prefill_grows_host_pages_without_remapping_prefix():
    pool = HostKvPagePool(page_count=5, page_size=2)
    first = pool.allocate("first", 2)
    other = pool.allocate("other", 2)
    grown = pool.grow("first", 5)
    assert grown.page_ids == (0, 2, 3)
    assert grown.generation == first.generation
    assert pool.allocation("other") == other
    with pytest.raises(ValueError, match="stale"):
        pool.release(first)
    with pytest.raises(MemoryError):
        pool.grow("first", 10)
    assert pool.allocation("first") == grown
    assert pool.free_page_count == 1
    pool.release(grown)
    pool.release(other)
    assert pool.free_page_count == 5


def test_request_growth_waits_for_previous_layer_to_drain():
    slots = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    pool = HostKvPagePool(page_count=4, page_size=2)
    coordinator = PrefillLayerwiseOffload(slots, pool)
    coordinator.begin_request("req", 1)
    work = coordinator.begin_layer("req", 0, 1, KvPageRange(0, 1))
    with pytest.raises(SlotBusyError, match="drain"):
        coordinator.grow_request("req", 4)
    coordinator.mark_kv_written(work)
    coordinator.start_host_save(work)
    coordinator.complete_host_save(work)
    coordinator.mark_attention_done(work)
    coordinator.release_layer(work)
    assert len(coordinator.grow_request("req", 4).page_ids) == 2


def test_partial_host_save_failure_keeps_slot_until_explicit_device_drain():
    def failing_save(work):
        raise RuntimeError("second component launch failed")

    slots = PrefillLayerSlotManager(logical_layers=1, physical_slots=1)
    coordinator = PrefillLayerwiseOffload(
        slots, HostKvPagePool(page_count=2, page_size=2), save=failing_save
    )
    coordinator.begin_request("req", 2)
    work = coordinator.begin_layer("req", 0, 2, KvPageRange(0, 2))
    coordinator.mark_kv_written(work)
    with pytest.raises(RuntimeError, match="second component"):
        coordinator.start_host_save(work)
    with pytest.raises(SlotBusyError):
        coordinator.release_layer(work)
    slots.mark_failed_local_work_drained(work.token)
    coordinator.release_layer(work)
