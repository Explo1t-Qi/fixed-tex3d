"""Tests for XML-reference-based runtime texture activation."""

from __future__ import annotations

import sys
from pathlib import Path


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from openvla_runtime_assets import (  # noqa: E402
    activate_runtime_texture,
    resolve_runtime_texture_binding,
    temporary_runtime_texture,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    original_texture = tmp_path / "texture_map.png"
    original_texture.write_bytes(b"clean texture")
    active_texture = tmp_path / "adversarial.png"
    active_texture.write_bytes(b"active texture")
    xml_path = tmp_path / "hope_object.xml"
    xml_path.write_text(
        """<mujoco>
  <asset>
    <texture name="tex-textured" type="2d" file="texture_map.png"/>
    <material name="textured" texture="tex-textured"/>
  </asset>
</mujoco>
"""
    )
    return xml_path, original_texture, active_texture


def test_runtime_texture_is_resolved_from_file_and_material_relations(
    tmp_path: Path,
) -> None:
    xml_path, original_texture, _ = _write_fixture(tmp_path)

    binding = resolve_runtime_texture_binding(xml_path, original_texture)

    assert binding.texture_name == "tex-textured"
    assert binding.material_names == ("textured",)
    assert binding.used_name_fallback is False


def test_activation_targets_resolved_nodes_instead_of_object_name_guess(
    tmp_path: Path,
) -> None:
    xml_path, original_texture, active_texture = _write_fixture(tmp_path)
    binding = resolve_runtime_texture_binding(xml_path, original_texture)

    activate_runtime_texture(xml_path, binding, active_texture)

    updated = xml_path.read_text()
    assert str(active_texture.resolve()) in updated
    assert 'name="tex-textured"' in updated
    assert 'name="textured"' in updated
    assert 'texuniform="false"' in updated


def test_temporary_activation_restores_xml_and_never_overwrites_clean_texture(
    tmp_path: Path,
) -> None:
    xml_path, original_texture, active_texture = _write_fixture(tmp_path)
    original_xml = xml_path.read_bytes()
    original_texture_bytes = original_texture.read_bytes()

    with temporary_runtime_texture(
        xml_path,
        original_texture,
        active_texture,
    ) as binding:
        assert binding.texture_name == "tex-textured"
        assert xml_path.read_bytes() != original_xml
        assert original_texture.read_bytes() == original_texture_bytes

    assert xml_path.read_bytes() == original_xml
    assert original_texture.read_bytes() == original_texture_bytes
    assert list(tmp_path.glob("*backup*")) == []
