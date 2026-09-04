"""Regression tests for the direct OpenVLA attack entrypoint bootstrap."""

import ast

from pathlib import Path


ATTACK_ENTRYPOINT = (
    Path(__file__).resolve().parents[2]
    / "openvla/experiments/robot/libero/attack_openvla.py"
)


def test_robot_sibling_path_is_configured_before_libero_utils_import() -> None:
    source = ATTACK_ENTRYPOINT.read_text()

    robot_path_setup = source.index(
        "sys.path.append(str(Path(__file__).parent.parent))"
    )
    libero_utils_import = source.index("from libero_utils import (")

    assert robot_path_setup < libero_utils_import


def _load_function(name: str):
    tree = ast.parse(ATTACK_ENTRYPOINT.read_text(encoding="utf-8"))
    node = next(
        child
        for child in tree.body
        if isinstance(child, ast.FunctionDef) and child.name == name
    )
    namespace = {}
    code = compile(
        ast.Module(body=[node], type_ignores=[]),
        str(ATTACK_ENTRYPOINT),
        "exec",
    )
    exec(code, namespace)
    return namespace[name]


def test_rollout_summary_does_not_report_zero_episode_rate() -> None:
    format_summary = _load_function("_format_rollout_summary")

    console, log = format_summary(total_successes=0, total_episodes=0)

    assert console == (
        "[DONE] Episodes: 0 | Task success rate: NOT EVALUATED "
        "(rollout disabled)"
    )
    assert log == "FINAL TASK SUCCESS RATE: NOT EVALUATED (0 episodes)"


def test_rollout_summary_labels_observed_successes_as_task_success_rate() -> None:
    format_summary = _load_function("_format_rollout_summary")

    console, log = format_summary(total_successes=1, total_episodes=4)

    assert console == "[DONE] Episodes: 4 | Task success rate: 25.00%"
    assert log == "FINAL TASK SUCCESS RATE: 25.00% (1/4)"
