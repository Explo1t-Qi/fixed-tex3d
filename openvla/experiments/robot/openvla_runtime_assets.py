"""Resolve and activate MuJoCo textures through XML relationships."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RuntimeTextureBinding:
    texture_name: str
    texture_file: str
    material_names: tuple[str, ...]
    used_name_fallback: bool


def _texture_file_path(
    xml_path: Path, root: ET.Element, file_reference: str
) -> Path:
    candidate = Path(file_reference)
    if candidate.is_absolute():
        return candidate.resolve()
    compiler = root.find("compiler")
    texture_dir = compiler.get("texturedir") if compiler is not None else None
    base = xml_path.parent / texture_dir if texture_dir else xml_path.parent
    return (base / candidate).resolve()


def resolve_runtime_texture_binding(
    xml_path: str | Path,
    target_texture_path: str | Path,
    *,
    object_name: str | None = None,
) -> RuntimeTextureBinding:
    """Resolve texture and materials by file/reference, with name fallback last."""
    xml = Path(xml_path)
    target = Path(target_texture_path).resolve()
    root = ET.parse(xml).getroot()
    textures = list(root.findall(".//texture"))

    exact = [
        element
        for element in textures
        if element.get("file")
        and _texture_file_path(xml, root, element.get("file", "")) == target
    ]
    if not exact:
        basename_matches = [
            element
            for element in textures
            if element.get("file")
            and Path(element.get("file", "")).name == target.name
        ]
        if len(basename_matches) == 1:
            exact = basename_matches

    used_name_fallback = False
    if not exact and object_name:
        expected_name = f"tex-{object_name}"
        exact = [
            element for element in textures if element.get("name") == expected_name
        ]
        used_name_fallback = bool(exact)
    if len(exact) != 1:
        raise ValueError(
            "runtime texture file reference must resolve to exactly one XML node"
        )

    texture = exact[0]
    texture_name = texture.get("name")
    texture_file = texture.get("file")
    if not texture_name or not texture_file:
        raise ValueError("runtime texture node requires name and file attributes")
    material_names = tuple(
        material.get("name", "")
        for material in root.findall(".//material")
        if material.get("texture") == texture_name and material.get("name")
    )
    if not material_names:
        raise ValueError("no XML material references the resolved runtime texture")
    return RuntimeTextureBinding(
        texture_name=texture_name,
        texture_file=texture_file,
        material_names=material_names,
        used_name_fallback=used_name_fallback,
    )


def activate_runtime_texture(
    xml_path: str | Path,
    binding: RuntimeTextureBinding,
    active_texture_path: str | Path,
) -> None:
    """Point the resolved texture node at an existing active texture file."""
    xml = Path(xml_path)
    active = Path(active_texture_path).resolve()
    if not active.is_file():
        raise FileNotFoundError(f"active texture does not exist: {active}")
    tree = ET.parse(xml)
    root = tree.getroot()
    textures = [
        element
        for element in root.findall(".//texture")
        if element.get("name") == binding.texture_name
    ]
    if len(textures) != 1:
        raise ValueError("resolved runtime texture node is no longer unique")
    textures[0].set("file", str(active))
    textures[0].set("type", "2d")

    found_materials: set[str] = set()
    for material in root.findall(".//material"):
        name = material.get("name")
        if name in binding.material_names and material.get("texture") == binding.texture_name:
            material.set("texuniform", "false")
            found_materials.add(name)
    if found_materials != set(binding.material_names):
        raise ValueError("resolved runtime material relation changed before activation")
    tree.write(xml)


@contextmanager
def temporary_runtime_texture(
    xml_path: str | Path,
    target_texture_path: str | Path,
    active_texture_path: str | Path,
    *,
    object_name: str | None = None,
) -> Iterator[RuntimeTextureBinding]:
    """Activate a texture and restore the XML bytes without backup artifacts."""
    xml = Path(xml_path)
    original_xml = xml.read_bytes()
    binding = resolve_runtime_texture_binding(
        xml, target_texture_path, object_name=object_name
    )
    try:
        activate_runtime_texture(xml, binding, active_texture_path)
        yield binding
    finally:
        xml.write_bytes(original_xml)
