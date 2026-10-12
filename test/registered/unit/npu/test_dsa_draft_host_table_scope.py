"""Exercise draft preparation lifecycle without importing NPU model dependencies."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[4]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsa_backend.py"


def load_definitions(namespace):
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definitions = []
    for node in source.body:
        if isinstance(node, ast.ClassDef) and node.name == "AscendDSAAttnBackend":
            definitions.extend(
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and method.name == "_page_table_max_seq_len"
            )
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "AscendDSAAttnMultiStepDraftBackend"
        ):
            definitions.append(node)
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("base_length", [124, 125, 127, 128, 131071])
def test_draft_pages_cover_last_step_before_first_consumer(base_length):
    namespace = load_definitions(
        {"AscendAttnMultiStepDraftBackend": object, "AscendDSAAttnBackend": object}
    )
    method = namespace["_page_table_max_seq_len"]
    for step in range(4):
        backend = SimpleNamespace(
            speculative_step_id=step,
            _draft_host_table_scope=SimpleNamespace(last_step=3),
            _cache_seqlens_cpu_max=lambda _: base_length + step + 1,
        )
        assert method(backend, None) == base_length + 4
        backend._draft_host_table_scope = None
        assert method(backend, None) == base_length + step + 1


@pytest.mark.parametrize("fail", [False, True])
def test_draft_scope_is_fresh_and_cleared_on_error(monkeypatch, fail):
    calls = []

    class Parent:
        def init_forward_metadata_out_graph(self, batch, in_capture=False):
            scopes = [
                getattr(backend, "_draft_host_table_scope", None)
                for backend in self.attn_backends
            ]
            assert scopes[0] is scopes[1]
            assert scopes[2] is None
            assert scopes[3] is scopes[0]
            calls.append(scopes[0])
            if fail:
                raise RuntimeError("preparation failed")

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.hicache_offload",
        SimpleNamespace(
            DraftHostTableScope=lambda last_step: SimpleNamespace(last_step=last_step),
            get_offload_backing=lambda pool: pool,
        ),
    )
    namespace = load_definitions(
        {"AscendAttnMultiStepDraftBackend": Parent, "AscendDSAAttnBackend": object}
    )
    backend = namespace["AscendDSAAttnMultiStepDraftBackend"]()
    backend.speculative_num_steps = 5
    backend.attn_backends = [
        SimpleNamespace(token_to_kv_pool=object()) for _ in range(5)
    ]
    backend.attn_backends[2].token_to_kv_pool = None
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="preparation failed"):
                backend.init_forward_metadata_out_graph(None)
        else:
            backend.init_forward_metadata_out_graph(None)
        assert all(
            getattr(item, "_draft_host_table_scope", None) is None
            for item in backend.attn_backends
        )
    assert calls[0] is not calls[1]
