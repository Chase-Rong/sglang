"""Registration failures must not expose partially registered PD regions."""

import ast
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest


def engine_type():
    source = (
        Path(__file__).parents[4]
        / "python/sglang/srt/disaggregation/ascend/transfer_engine.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("batch_register", "batch_deregister")
    ]
    namespace = {"List": list, "logger": logging.getLogger(__name__)}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return type(
        "Engine", (), {method.name: namespace[method.name] for method in methods}
    )


def make_engine():
    engine = engine_type()()
    engine.engine = Mock()
    engine.engine.batch_register_memory.return_value = 0
    engine.engine.unregister_memory.return_value = 0
    return engine


def test_duplicate_registration_and_deregistration_are_idempotent():
    engine = make_engine()
    engine.batch_register([100, 100, 200], [32, 32, 64])
    engine.batch_register([100], [32])
    engine.engine.batch_register_memory.assert_called_once_with([100, 200], [32, 64])
    engine.batch_deregister([100, 100, 200])
    engine.batch_deregister([100])
    assert engine.engine.unregister_memory.call_count == 2


@pytest.mark.parametrize("failure", [-1, RuntimeError("native registration failed")])
def test_partial_failure_rolls_back_new_regions_and_blocks_reuse(failure):
    engine = make_engine()
    engine.batch_register([100], [32])
    if isinstance(failure, Exception):
        engine.engine.batch_register_memory.side_effect = failure
    else:
        engine.engine.batch_register_memory.return_value = failure
    with pytest.raises(RuntimeError):
        engine.batch_register([100, 200, 300], [32, 64, 32])
    assert [
        call.args[0] for call in engine.engine.unregister_memory.call_args_list
    ] == [200, 300]
    assert engine._registered_regions == {100: 32}
    with pytest.raises(RuntimeError, match="restart"):
        engine.batch_register([400], [32])


def test_changed_length_is_rejected_before_native_call():
    engine = make_engine()
    engine.batch_register([100], [32])
    with pytest.raises(ValueError, match="length"):
        engine.batch_register([100], [64])
    assert engine.engine.batch_register_memory.call_count == 1
