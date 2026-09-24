"""CPU regression for DSA graph input aliasing; no NPU graph execution.

Extract the production metadata methods to avoid importing the NPU runtime.
The base metadata update and DSA effective-length calculation run unchanged.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "python/sglang/srt/hardware_backend/npu/attention"


def metadata_backend():
    classes = []
    for filename, class_name, methods in (
        (
            "ascend_backend.py",
            "AscendAttnBackend",
            {"_init_cuda_graph_metadata", "_apply_cuda_graph_metadata"},
        ),
        (
            "ascend_dsa_backend.py",
            "AscendDSAAttnBackend",
            {"_init_cuda_graph_metadata", "_cache_seqlens"},
        ),
    ):
        source = ast.parse((BACKEND / filename).read_text(encoding="utf-8"))
        cls = next(
            node for node in source.body if getattr(node, "name", None) == class_name
        )
        cls.decorator_list = []
        cls.bases = (
            [ast.Name(id="AscendAttnBackend", ctx=ast.Load())] if classes else []
        )
        cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
        classes.append(cls)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *classes,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "_is_dflash_verify": lambda spec: False}
    exec(compile(ast.fix_missing_locations(tree), str(BACKEND), "exec"), namespace)
    return namespace["AscendDSAAttnBackend"]


def make_backend(cls, step):
    backend = cls()
    backend._create_forward_metadata = SimpleNamespace
    backend.graph_metadata = {"block_tables": torch.zeros((2, 16), dtype=torch.int32)}
    backend.is_hybrid_swa = False
    backend.use_sliding_window_kv_pool = False
    backend.q_head_num_padding = None
    backend.speculative_num_draft_tokens = 6
    backend.speculative_step_id = step
    backend.speculative_step_offset_npu = torch.tensor(step + 1, dtype=torch.int32)
    backend.page_size = 4
    backend.req_to_token = torch.arange(2 * 64).reshape(2, 64)
    return backend


@pytest.mark.parametrize(
    "mode_name,step", [("verify", 0), ("draft", 0), ("draft", 4), ("decode", 0)]
)
def test_graph_updates_preserve_raw_lengths_and_capture_storage(mode_name, step):
    mode = SimpleNamespace(
        is_target_verify=lambda: mode_name == "verify",
        is_draft_extend_v2=lambda: False,
        is_decode_or_idle=lambda: mode_name != "verify",
        is_dllm_extend=lambda: False,
    )
    spec = None if mode_name == "decode" else object()
    offset = 6 if mode_name == "verify" else step + 1 if spec is not None else 0
    cls = metadata_backend()
    backends = [make_backend(cls, step) for _ in range(2)]
    raw = torch.tensor([12, 20], dtype=torch.int32)
    pointers = set()
    for backend in backends:
        for bs in (1, 2):
            metadata = backend._init_cuda_graph_metadata(bs, mode, raw[:bs])
            assert metadata is backend.graph_metadata[bs]
            pointers.add(metadata.seq_lens.data_ptr())
    assert len(pointers) == 4
    assert raw.data_ptr() not in pointers

    for replay in range(3):
        raw.copy_(torch.tensor([12 + replay, 20 + replay]))
        before = raw.clone()
        for backend in backends:
            for bs in (1, 2):
                metadata = backend.graph_metadata[bs]
                address = metadata.seq_lens.data_ptr()
                backend._apply_cuda_graph_metadata(
                    bs, torch.arange(bs), raw[:bs], before[:bs], mode, spec
                )
                assert metadata.seq_lens.data_ptr() == address
                torch.testing.assert_close(raw, before)
                torch.testing.assert_close(metadata.seq_lens[:bs], before[:bs] + offset)
                effective = backend._cache_seqlens(
                    SimpleNamespace(
                        batch_size=bs,
                        seq_lens=raw,
                        forward_mode=mode,
                        spec_info=spec,
                    )
                )
                torch.testing.assert_close(effective, before[:bs] + offset)
