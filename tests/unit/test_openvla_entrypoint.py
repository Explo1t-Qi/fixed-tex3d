"""Regression tests for the direct OpenVLA attack entrypoint bootstrap."""

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
