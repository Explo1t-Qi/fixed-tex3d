"""Collect frozen-texture Step 1 clean/adversarial MuJoCo observation pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla" / "experiments" / "robot"
LIBERO_ROBOT_ROOT = ROBOT_ROOT / "libero"
for source_root in (ROBOT_ROOT, LIBERO_ROBOT_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from openvla_runtime_assets import (  # noqa: E402
    resolve_runtime_texture_binding,
    temporary_runtime_texture,
)
from step1_o2_p2 import RAW_RGB_SHAPE, sha256_bytes, sha256_rgb  # noqa: E402


SUITE_NAME = "libero_spatial"
TASK_ID = 0
OBJECT_NAME = "akita_black_bowl"
HELDOUT_STATE_IDS = tuple(range(10, 20))
CAMERA_FIELD = "agentview_image"
PI05_CAMERA_FIELD = "base_0_rgb"
WRIST_FIELD = "robot0_eye_in_hand_image"


@dataclass(frozen=True)
class _Capture:
    base_rgb: np.ndarray
    wrist_rgb: np.ndarray
    robot_state: np.ndarray
    scene_state: np.ndarray
    camera_state: np.ndarray
    task_description: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Step 1 held-out MuJoCo Active Texture pairs."
    )
    parser.add_argument("--attack-texture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=Path("/opt/libero"))
    parser.add_argument("--xml-path", type=Path)
    parser.add_argument("--clean-texture-path", type=Path)
    parser.add_argument(
        "--state-ids",
        default="10-19",
        help="Inclusive range (for example 10-19) or comma-separated IDs.",
    )
    return parser.parse_args(argv)


def _parse_state_ids(value: str) -> tuple[int, ...]:
    try:
        if "-" in value:
            start_text, end_text = value.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            result = tuple(range(start, end + 1))
        else:
            result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"invalid state IDs: {value!r}") from error
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"state IDs must be unique non-negative integers: {value!r}")
    return result


def _canonical_robot_state(observation: Mapping[str, Any]) -> np.ndarray:
    from libero_utils import quat2axisangle

    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"]),
            quat2axisangle(np.asarray(observation["robot0_eef_quat"]).copy()),
            np.asarray(observation["robot0_gripper_qpos"]),
        )
    ).astype(np.float64, copy=False)


def _copy_rgb(name: str, value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != RAW_RGB_SHAPE or image.dtype != np.uint8:
        raise RuntimeError(
            f"{name} must be a uint8 MuJoCo RGB frame with shape {RAW_RGB_SHAPE}, "
            f"got shape={image.shape}, dtype={image.dtype}"
        )
    return np.ascontiguousarray(image.copy())


def _sim_state(sim: Any) -> np.ndarray:
    arrays = (
        np.asarray(sim.data.qpos).reshape(-1),
        np.asarray(sim.data.qvel).reshape(-1),
        np.asarray(sim.data.body_xpos).reshape(-1),
        np.asarray(sim.data.body_xquat).reshape(-1),
    )
    return np.concatenate(arrays).astype(np.float64, copy=True)


def _camera_state(sim: Any) -> np.ndarray:
    try:
        camera_id = sim.model.camera_name2id("agentview")
    except AttributeError:
        import mujoco

        camera_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview"
        )
    if int(camera_id) < 0:
        raise RuntimeError("agentview camera is unavailable")
    return np.concatenate(
        (
            np.asarray(sim.data.cam_xpos[camera_id]).reshape(-1),
            np.asarray(sim.data.cam_xmat[camera_id]).reshape(-1),
            np.asarray([sim.model.cam_fovy[camera_id]], dtype=np.float64),
        )
    ).astype(np.float64, copy=True)


def _capture_state(task: Any, initial_state: Any) -> _Capture:
    from libero_utils import get_libero_env

    env, task_description = get_libero_env(task, "openvla", resolution=512)
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        env.env.sim.forward()
        sim = env.env.sim
        return _Capture(
            base_rgb=_copy_rgb(CAMERA_FIELD, observation[CAMERA_FIELD]),
            wrist_rgb=_copy_rgb(WRIST_FIELD, observation[WRIST_FIELD]),
            robot_state=_canonical_robot_state(observation),
            scene_state=_sim_state(sim),
            camera_state=_camera_state(sim),
            task_description=str(task_description),
        )
    finally:
        env.close()


def _validate_scene_pair(clean: _Capture, adversarial: _Capture) -> None:
    checks = {
        "robot state": np.array_equal(clean.robot_state, adversarial.robot_state),
        "scene state": np.array_equal(clean.scene_state, adversarial.scene_state),
        "camera state": np.array_equal(clean.camera_state, adversarial.camera_state),
        "task language": clean.task_description == adversarial.task_description,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "clean/adversarial pair changed non-texture scene inputs: "
            + ", ".join(failures)
        )


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path)
    restored = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if sha256_rgb(restored) != sha256_rgb(image):
        raise RuntimeError(f"lossless RGB round-trip failed: {path}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    state_ids = _parse_state_ids(args.state_ids)
    libero_root = args.libero_root.expanduser().resolve()
    if not libero_root.is_dir():
        raise FileNotFoundError(f"LIBERO root not found: {libero_root}")
    os.environ["LIBERO_ROOT"] = str(libero_root)
    if str(libero_root) not in sys.path:
        sys.path.append(str(libero_root))

    asset_root = libero_root / "libero/libero/assets/stable_scanned_objects/akita_black_bowl"
    xml_path = (args.xml_path or asset_root / "akita_black_bowl.xml").resolve()
    clean_texture = (args.clean_texture_path or asset_root / "texture.png").resolve()
    attack_texture = args.attack_texture.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, name in (
        (xml_path, "object XML"),
        (clean_texture, "clean texture"),
        (attack_texture, "attack texture"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    if output_dir.exists():
        raise FileExistsError(f"pair output directory must be fresh: {output_dir}")

    original_xml = xml_path.read_bytes()
    original_texture = clean_texture.read_bytes()
    texture_sha256 = sha256_bytes(attack_texture.read_bytes())
    binding = resolve_runtime_texture_binding(
        xml_path, clean_texture, object_name=OBJECT_NAME
    )
    if binding.used_name_fallback:
        raise RuntimeError(
            "clean texture must resolve by its XML file relationship, not name fallback"
        )

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    task = suite.get_task(TASK_ID)
    init_states = suite.get_task_init_states(TASK_ID)
    if len(init_states) <= max(state_ids):
        raise RuntimeError(f"LIBERO task does not contain requested states {state_ids}")

    clean_captures = {
        state_id: _capture_state(task, init_states[state_id])
        for state_id in state_ids
    }
    with temporary_runtime_texture(
        xml_path,
        clean_texture,
        attack_texture,
        object_name=OBJECT_NAME,
    ):
        adversarial_captures = {
            state_id: _capture_state(task, init_states[state_id])
            for state_id in state_ids
        }

    if xml_path.read_bytes() != original_xml:
        raise RuntimeError("object XML was not restored after pair collection")
    if clean_texture.read_bytes() != original_texture:
        raise RuntimeError("clean texture was modified during pair collection")
    backup_files = sorted(xml_path.parent.glob("*clean_backup*"))
    if backup_files:
        raise RuntimeError(f"unexpected asset backup files remain: {backup_files}")

    output_dir.mkdir(parents=True)
    manifest_records: list[dict[str, Any]] = []
    for state_id in state_ids:
        clean = clean_captures[state_id]
        adversarial = adversarial_captures[state_id]
        _validate_scene_pair(clean, adversarial)
        sample_id = f"libero_spatial_task00_state{state_id:02d}"
        state_dir = output_dir / f"state_{state_id}"
        state_dir.mkdir()
        clean_path = state_dir / "clean.png"
        adversarial_path = state_dir / "adversarial.png"
        _save_rgb(clean_path, clean.base_rgb)
        _save_rgb(adversarial_path, adversarial.base_rgb)
        np.savez_compressed(
            state_dir / "fixed_inputs.npz",
            wrist_rgb_raw=clean.wrist_rgb,
            robot_state=clean.robot_state,
            scene_state=clean.scene_state,
            camera_state=clean.camera_state,
        )
        metadata = {
            "sample_id": sample_id,
            "task_suite": SUITE_NAME,
            "task_id": TASK_ID,
            "state_id": state_id,
            "clean_rgb_sha256": sha256_rgb(clean.base_rgb),
            "adv_rgb_sha256": sha256_rgb(adversarial.base_rgb),
            "texture_sha256": texture_sha256,
            "camera": CAMERA_FIELD,
            "source_resolution": [512, 512],
            "openvla_attacked_camera_field": CAMERA_FIELD,
            "pi05_corresponding_image_field": PI05_CAMERA_FIELD,
            "pi05_fixed_wrist_field": WRIST_FIELD,
            "pi05_wrist_rgb_source": "clean_capture",
            "pi05_fixed_wrist_rgb_sha256": sha256_rgb(clean.wrist_rgb),
            "pi05_wrist_rgb_held_fixed": True,
            "captured_clean_wrist_rgb_sha256": sha256_rgb(clean.wrist_rgb),
            "captured_adv_wrist_rgb_sha256": sha256_rgb(adversarial.wrist_rgb),
            "captured_wrist_rgb_identical": bool(
                np.array_equal(clean.wrist_rgb, adversarial.wrist_rgb)
            ),
            "task_description": clean.task_description,
            "no_policy_action_between_observations": True,
            "scene_state_identical": True,
            "camera_state_identical": True,
            "robot_state_identical": True,
            "clean_path": "clean.png",
            "adversarial_path": "adversarial.png",
            "fixed_inputs_path": "fixed_inputs.npz",
        }
        (state_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_records.append(
            {
                **metadata,
                "metadata_path": f"state_{state_id}/metadata.json",
            }
        )

    manifest = {
        "schema_version": "step1_heldout_pairs_v1",
        "texture_sha256": texture_sha256,
        "heldout_state_ids": list(state_ids),
        "num_pairs": len(manifest_records),
        "xml_restored": True,
        "clean_texture_unchanged": True,
        "records": manifest_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    manifest = _run(_parse_args(argv))
    print(
        json.dumps(
            {
                "status": "held-out pair collection complete",
                "num_pairs": manifest["num_pairs"],
                "texture_sha256": manifest["texture_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
