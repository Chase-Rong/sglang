import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def coordinator(monkeypatch):
    source = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/hisparse/runtime.py"
    )
    spec = importlib.util.spec_from_file_location("rebase_npu_coordinator", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    context = ModuleType("sglang.srt.runtime_context")
    context.get_spec = lambda: SimpleNamespace(
        speculative_eagle_topk=1, speculative_num_draft_tokens=6
    )
    monkeypatch.setitem(sys.modules, context.__name__, context)
    pool = SimpleNamespace(req_generation=torch.tensor([0, 1, 1]))
    runner = SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=object(),
        token_to_kv_pool=SimpleNamespace(decode_host_backing=None),
        device="cpu",
        prefill_offload_runtime=None,
    )
    return module.NpuHiSparseCoordinator(runner)


def make_batch():
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        req_pool_indices=torch.tensor([1, 2]),
        out_cache_loc=torch.tensor(
            [127, 256, 257, 258, 259, 260, 511, 640, 641, 642, 643, 644]
        ),
    )


@pytest.mark.parametrize("accepted", [0, 2, 5])
def test_chain_verify_preserves_canonical_slots_and_published_kv(coordinator, accepted):
    batch = make_batch()
    addresses = batch.out_cache_loc.clone()
    host = torch.zeros(768)
    values = torch.arange(12, dtype=torch.float32) + 1
    coordinator.prepare_spec_verify(batch)
    host[batch.out_cache_loc] = values
    accepts = torch.full((2, 6), -1)
    for row in range(2):
        accepts[row, : accepted + 1] = row * 6 + torch.arange(accepted + 1)
    coordinator.commit_spec_accept_tokens(batch, accepts)
    torch.testing.assert_close(batch.out_cache_loc, addresses)
    for row in range(2):
        committed = batch.out_cache_loc[row * 6 : row * 6 + accepted + 1]
        torch.testing.assert_close(
            host[committed], values[row * 6 : row * 6 + accepted + 1]
        )
        next_location = batch.out_cache_loc[row * 6 + min(accepted + 1, 5)]
        host[next_location] = 99
        assert host[next_location] == 99


def test_verify_rejects_incomplete_reserved_window(coordinator):
    batch = make_batch()
    batch.out_cache_loc = batch.out_cache_loc[:-1]
    with pytest.raises(ValueError, match="chain width"):
        coordinator.prepare_spec_verify(batch)


def test_accept_requires_one_row_per_request(coordinator):
    with pytest.raises(ValueError, match="active requests"):
        coordinator.commit_spec_accept_tokens(make_batch(), torch.zeros(1, 6))


def test_idle_verify_does_not_require_reservations(coordinator):
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: True))
    coordinator.prepare_spec_verify(batch)
    coordinator.commit_spec_accept_tokens(batch, None)


def test_completed_layer_records_do_not_keep_integrated_runtime_busy(coordinator):
    event = SimpleNamespace(query=Mock(return_value=False))
    coordinator.bind_prefill_runtime(
        SimpleNamespace(
            active=False, failed=False, layers={0: object()}, host_visible=event
        )
    )
    assert coordinator.has_ongoing_staging()
    event.query.return_value = True
    assert not coordinator.has_ongoing_staging()


if __name__ == "__main__":
    run_pytest(__file__)
