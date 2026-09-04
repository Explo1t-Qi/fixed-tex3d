"""Dependency-light renderer scene and visibility composition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation


DEFAULT_RENDERER_POSITION_OFFSET: Final[tuple[float, float, float]] = (
    0.0,
    0.0,
    0.0,
)


@dataclass(frozen=True)
class TargetBodyPose:
    model_matrix: torch.Tensor
    body_id: int
    body_name: str


def _simulation(env: Any) -> Any:
    return env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim


def _native_mujoco_object_type(kind: str) -> Any:
    import mujoco

    return getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")


def _raw_mujoco_model(model: Any) -> Any:
    """Unwrap robosuite's binding_utils.MjModel when present."""
    return getattr(model, "_model", model)


def resolve_mujoco_object_id(model: Any, name: str, kind: str) -> int:
    """Resolve names across mujoco-py wrappers and native mujoco.MjModel."""
    for method_name in (f"{kind[:3]}_name2id", f"{kind}_name2id"):
        if hasattr(model, method_name):
            object_id = int(getattr(model, method_name)(name))
            if object_id < 0:
                raise KeyError(name)
            return object_id
    if hasattr(model, "name2id"):
        object_id = int(model.name2id(name, kind))
        if object_id < 0:
            raise KeyError(name)
        return object_id

    import mujoco

    object_id = int(
        mujoco.mj_name2id(
            _raw_mujoco_model(model),
            _native_mujoco_object_type(kind),
            name,
        )
    )
    if object_id < 0:
        raise KeyError(name)
    return object_id


def _resolve_compiled_texture_ids(model: Any, source_name: str) -> tuple[int, ...]:
    """Resolve one source texture across robosuite-prefixed MJCF instances.

    ``MujocoXMLObject`` prefixes copied asset names with the object instance's
    naming prefix.  The compiled model can therefore contain multiple texture
    IDs such as ``akita_black_bowl_1_tex-akita_black_bowl`` while the source
    XML calls the texture ``tex-akita_black_bowl``.
    """
    raw_model = _raw_mujoco_model(model)
    compiled_names: list[tuple[int, str]] = []
    try:
        import mujoco

        texture_type = _native_mujoco_object_type("texture")
        for texture_id in range(int(raw_model.ntex)):
            compiled_name = mujoco.mj_id2name(
                raw_model, texture_type, texture_id
            )
            if compiled_name is not None:
                compiled_names.append((texture_id, compiled_name))
    except (ImportError, AttributeError, TypeError, ValueError):
        # Legacy mujoco-py models expose tex_name2id but cannot be enumerated
        # with native mujoco module functions.
        compiled_names = []

    suffix = f"_{source_name}"
    matches = tuple(
        texture_id
        for texture_id, compiled_name in compiled_names
        if compiled_name == source_name or compiled_name.endswith(suffix)
    )
    if matches:
        return matches

    try:
        return (resolve_mujoco_object_id(model, source_name, "texture"),)
    except KeyError as error:
        available = tuple(name for _, name in compiled_names)
        raise KeyError(
            f"{source_name!r}; compiled texture names={available!r}"
        ) from error


def _body_name(model: Any, body_id: int) -> str | None:
    if hasattr(model, "body_id2name"):
        try:
            return model.body_id2name(body_id)
        except Exception:
            pass
    if hasattr(model, "id2name"):
        try:
            return model.id2name(body_id, "body")
        except Exception:
            pass
    try:
        import mujoco

        return mujoco.mj_id2name(
            _raw_mujoco_model(model),
            _native_mujoco_object_type("body"),
            body_id,
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return None


def _pose_from_id(simulation: Any, body_id: int, body_name: str, device: Any) -> TargetBodyPose:
    data = simulation.data
    position_field = "body_xpos" if hasattr(data, "body_xpos") else "xpos"
    quaternion_field = "body_xquat" if hasattr(data, "body_xquat") else "xquat"
    position = np.asarray(
        getattr(data, position_field)[body_id], dtype=np.float32
    )
    quaternion = np.asarray(
        getattr(data, quaternion_field)[body_id], dtype=np.float32
    )
    rotation = Rotation.from_quat(
        (quaternion[1], quaternion[2], quaternion[3], quaternion[0])
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation.as_matrix().astype(np.float32)
    matrix[:3, 3] = position
    return TargetBodyPose(torch.from_numpy(matrix).to(device), body_id, body_name)


def find_target_body_poses(
    env: Any,
    search_keywords: Sequence[Sequence[str]],
    *,
    device: Any,
    texture_name: str | None = None,
) -> tuple[TargetBodyPose, ...]:
    """Return every body matching the first successful keyword group.

    All returned instances share the caller's one texture parameterization;
    unlike the baseline helper, this function never stops at the first body.
    """
    simulation = _simulation(env)
    texture_body_ids: set[int] | None = None
    if texture_name is not None:
        model = simulation.model
        try:
            texture_ids = _resolve_compiled_texture_ids(model, texture_name)
        except Exception as error:
            raise ValueError(
                f"MuJoCo model does not contain source texture "
                f"{texture_name!r}: {error}"
            ) from error
        material_texture_ids = np.asarray(model.mat_texid)
        material_ids = set(
            int(index)
            for index in np.flatnonzero(
                np.isin(material_texture_ids, texture_ids).reshape(
                    material_texture_ids.shape[0], -1
                ).any(axis=1)
            )
        )
        geometry_material_ids = np.asarray(model.geom_matid, dtype=np.int64)
        geometry_body_ids = np.asarray(model.geom_bodyid, dtype=np.int64)
        texture_body_ids = {
            int(geometry_body_ids[geometry_id])
            for geometry_id, material_id in enumerate(geometry_material_ids)
            if int(material_id) in material_ids
        }
    for keyword_group in search_keywords:
        matches: list[TargetBodyPose] = []
        for body_id in range(int(simulation.model.nbody)):
            name = _body_name(simulation.model, body_id)
            if not name or "vis" in name or "site" in name:
                continue
            if all(keyword in name for keyword in keyword_group) and (
                texture_body_ids is None
                or bool(_descendant_bodies(simulation.model, body_id) & texture_body_ids)
            ):
                matches.append(
                    _pose_from_id(simulation, body_id, name, device)
                )
        if matches:
            return tuple(matches)
    return ()


def _descendant_bodies(model: Any, root_body_id: int) -> set[int]:
    descendants = {root_body_id}
    if not hasattr(model, "body_parentid"):
        return descendants
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(np.asarray(model.body_parentid)):
            if int(parent_id) in descendants and body_id not in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def build_frontmost_instance_masks(
    segmentation: np.ndarray,
    *,
    model: Any,
    body_ids: Sequence[int],
    geom_object_type: int,
) -> torch.Tensor:
    """Split one MuJoCo front-most geom ID image into per-instance masks."""
    if (
        not isinstance(segmentation, np.ndarray)
        or segmentation.ndim != 3
        or segmentation.shape[2] != 2
        or not np.issubdtype(segmentation.dtype, np.integer)
    ):
        raise ValueError("segmentation must be an integer [H, W, 2] array")
    if not body_ids or len(set(int(item) for item in body_ids)) != len(body_ids):
        raise ValueError("body_ids must be non-empty and unique")
    geom_body_ids = np.asarray(model.geom_bodyid, dtype=np.int64)
    if geom_body_ids.shape != (int(model.ngeom),):
        raise ValueError("geom_bodyid shape does not match ngeom")

    height, width = segmentation.shape[:2]
    masks = np.zeros((len(body_ids), 1, height, width), dtype=np.float32)
    object_types = segmentation[..., 0]
    object_ids = segmentation[..., 1]
    geom_pixels = object_types == geom_object_type
    for instance_index, body_id in enumerate(body_ids):
        descendants = _descendant_bodies(model, int(body_id))
        geom_ids = np.flatnonzero(
            np.isin(geom_body_ids, np.fromiter(descendants, dtype=np.int64))
        )
        masks[instance_index, 0] = geom_pixels & np.isin(object_ids, geom_ids)
    result = torch.from_numpy(masks)
    if bool((result.sum(dim=0) > 1.0).any()):
        raise RuntimeError("front-most instance masks must be mutually exclusive")
    return result


def _resolve_geom_object_type() -> int:
    try:
        import mujoco

        return int(mujoco.mjtObj.mjOBJ_GEOM)
    except (ImportError, AttributeError):
        try:
            from mujoco_py import const

            return int(const.OBJ_GEOM)
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "MuJoCo GEOM object type is unavailable in this runtime"
            ) from error


def capture_frontmost_instance_masks(
    env: Any,
    *,
    body_ids: Sequence[int],
    resolution: int,
    camera_name: str = "agentview",
    geom_object_type: int | None = None,
) -> torch.Tensor:
    """Capture one oriented MuJoCo segmentation image and split its instances."""
    if resolution <= 0:
        raise ValueError("segmentation resolution must be positive")
    simulation = _simulation(env)
    segmentation = np.asarray(
        simulation.render(
            width=resolution,
            height=resolution,
            camera_name=camera_name,
            mode="offscreen",
            segmentation=True,
        )
    )
    if segmentation.shape != (resolution, resolution, 2):
        raise RuntimeError(
            "MuJoCo segmentation must return [resolution, resolution, 2]"
        )
    oriented = segmentation[::-1, ::-1].copy()
    return build_frontmost_instance_masks(
        oriented,
        model=simulation.model,
        body_ids=body_ids,
        geom_object_type=(
            _resolve_geom_object_type()
            if geom_object_type is None
            else geom_object_type
        ),
    )


def compose_visibility_masked_renderer_delta(
    mujoco_clean_rgb: torch.Tensor,
    mujoco_instance_masks: torch.Tensor,
    renderer_adversarial_rgb: torch.Tensor,
    renderer_clean_rgb: torch.Tensor,
    renderer_valid_masks: torch.Tensor,
) -> torch.Tensor:
    """Apply only visible renderer color deltas to the clean MuJoCo frame."""
    if mujoco_instance_masks.ndim != 4 or mujoco_instance_masks.shape[1] != 1:
        raise ValueError("MuJoCo masks must have shape [instances, 1, H, W]")
    instances, _, height, width = mujoco_instance_masks.shape
    expected_rgb = (instances, 3, height, width)
    if mujoco_clean_rgb.shape != (1, 3, height, width):
        raise ValueError("clean MuJoCo RGB shape does not match instance masks")
    if renderer_adversarial_rgb.shape != expected_rgb:
        raise ValueError("adversarial renderer RGB shape does not match masks")
    if renderer_clean_rgb.shape != expected_rgb:
        raise ValueError("clean renderer RGB shape does not match masks")
    if renderer_valid_masks.shape != (instances, 1, height, width):
        raise ValueError("renderer masks do not match MuJoCo masks")
    if not bool(
        ((mujoco_instance_masks == 0.0) | (mujoco_instance_masks == 1.0)).all()
    ):
        raise ValueError("MuJoCo front-most masks must be hard 0/1 values")
    if bool((mujoco_instance_masks.sum(dim=0) > 1.0).any()):
        raise ValueError("MuJoCo front-most masks must be mutually exclusive")
    values = (
        mujoco_clean_rgb,
        mujoco_instance_masks,
        renderer_adversarial_rgb,
        renderer_clean_rgb,
        renderer_valid_masks,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("composition inputs must be finite")

    renderer_delta = renderer_adversarial_rgb - renderer_clean_rgb.detach()
    visible_delta = (
        mujoco_instance_masks.to(renderer_delta.device)
        * renderer_valid_masks
        * renderer_delta
    )
    return torch.clamp(
        mujoco_clean_rgb + visible_delta.sum(dim=0, keepdim=True), 0.0, 1.0
    )
