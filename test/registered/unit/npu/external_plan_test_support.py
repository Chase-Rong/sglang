"""CPU test-only ABI3 planner oracle and loader; never imported by production."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

_spec = importlib.util.spec_from_file_location("protected_hash_contract_oracle", Path(__file__).resolve().parents[4] / "native/fsa/tests/planner_reference.py")
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)

def install_planner_double(monkeypatch, attention=None):
    import sglang.srt.hardware_backend.npu.fsa_native as native

    calls = []

    def planner(i, o, C, R, host_tokens, version):
        calls.append("planner")
        assert version == 3 and len(i) == 14 and len(o) == 23
        assert o[18].shape[1] == 8 and o[15].shape[1] % 16 == 0
        T, K = i[0].shape
        for n in [0, 4, 5, 6, 7, 8, 9, 17, 19]:
            o[n].fill_(-1)
        for n in [1, 2, 3, 10, 11, 12, 13, 15, 16, 18, 20, 21, 22]:
            o[n].zero_()
        begin = 0
        for b, end in enumerate(i[1].tolist()):
            owner = int(i[2][b])
            if owner <= 0:
                begin = end
                continue
            expected = _reference.reference(
                i[0][begin:end].tolist(),
                i[3][begin:end].tolist(),
                i[6][owner].tolist(),
                i[7][owner].tolist(),
                i[8][owner].tolist(),
                i[9][owner].tolist(),
                length=int(i[5][b]),
            )
            i[9][owner].copy_(torch.tensor(expected["reusable"]))
            count = len(expected["keys"])
            o[12][b] = count
            for n, name in [(7, "keys"), (8, "slots"), (10, "miss"), (11, "cacheable")]:
                o[n][b, :count] = torch.tensor(expected[name], dtype=o[n].dtype)
            o[9][b, :count] = (
                torch.tensor(expected["owner"], dtype=torch.int32) + begin * K
            )
            for local, u in enumerate(expected["row_union"]):
                row, col = begin + local // K, local % K
                o[5][row] = b
                if u < 0:
                    continue
                slot, miss, cacheable = (
                    expected["slots"][u],
                    expected["miss"][u],
                    expected["cacheable"][u],
                )
                o[0][row, col] = slot
                o[1][row, col] = miss
                o[2][row, col] = miss and expected["owner"][u] == local
                o[3][row, col] = cacheable
                o[4][row] = owner
                o[6][row, col] = u
                o[15][b, u] = 1 if miss else 2
                o[19][row, 0, col] = i[4][row, 0, col]
            begin = end

    try:
        existing = native.native_ops()
    except RuntimeError:
        existing = SimpleNamespace()
    if attention is not None:
        existing.npu_fused_sparse_attention_overlap = attention
    if not hasattr(existing, "external_plan_prepare"):
        existing.external_plan_prepare = planner
    monkeypatch.setattr(native, "native_ops", lambda: existing)
    return calls
