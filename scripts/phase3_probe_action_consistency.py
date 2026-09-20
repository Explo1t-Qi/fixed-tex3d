#!/usr/bin/env python3
"""Read-only probe-vs-deployed-action check on a frozen Phase 3 texture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
OPENVLA_ROOT = PROJECT_ROOT / "openvla"
STATE_IDS = tuple(range(10))
UNNORM_KEY = "libero_spatial_no_noops"


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-param", type=Path, required=True)
    parser.add_argument("--probe-artifact-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--openvla-checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--openvla-device", default="cuda:0")
    parser.add_argument("--pi05-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "texture": args.texture_param.expanduser().resolve(strict=True),
        "probes": args.probe_artifact_dir.expanduser().resolve(strict=True),
        "openpi": args.openpi_root.expanduser().resolve(strict=True),
        "openvla_checkpoint": args.openvla_checkpoint.expanduser().resolve(strict=True),
        "pi05_checkpoint": args.pi05_checkpoint.expanduser().resolve(strict=True),
        "libero": args.libero_root.expanduser().resolve(strict=True),
    }
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")
    paths["output"] = output
    return paths


def _add_source_paths(openpi_root: Path) -> None:
    paths = (
        PROJECT_ROOT,
        ROBOT_ROOT / "libero",
        ROBOT_ROOT,
        OPENVLA_ROOT,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required source paths are missing: {missing}")
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _uint8_rgb(image: torch.Tensor) -> np.ndarray:
    if (
        not isinstance(image, torch.Tensor)
        or tuple(image.shape) != (1, 3, 512, 512)
        or not bool(torch.isfinite(image).all())
    ):
        raise RuntimeError("canonical image must be finite [1,3,512,512]")
    return (
        image[0]
        .detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _validate_paths(args)
    os.environ["LIBERO_ROOT"] = str(paths["libero"])
    if str(paths["libero"]) not in sys.path:
        sys.path.insert(0, str(paths["libero"]))
    _add_source_paths(paths["openpi"])

    from scripts.phase2_action_representation_extract import _noise
    from phase2_action_representation import (
        extract_openvla_action_representation,
        extract_pi05_action_representation,
    )
    from phase2_multilevel_gradient_diagnostic import load_texture_parameter
    from phase2_shared_gradient_smoke import (
        _client_image,
        _load_pi05,
        _robot_state,
    )
    from phase2_shared_optimization import SharedOptimizationProtocol
    from phase3_action_predictive_objective import load_frozen_primary_probes
    from phase3_probe_action_consistency import ConsistencyInputs, consistency_metrics
    from attack_openvla import (
        DifferentiableRenderer,
        OBJECTS,
        _build_adv_samples,
        capture_frontmost_instance_masks,
        find_target_body_poses,
        freeze_openvla_for_o2,
        get_libero_env,
        get_libero_image,
        get_render_mvp_from_matrix,
        parse_mesh_scale,
        resolve_runtime_texture_binding,
    )
    from openvla_model_inputs import ensure_trailing_empty_token
    from openvla_policy_view import (
        deployment_center_crop_uint8,
        resize_policy_pre_crop_canvas,
    )
    from openvla_utils import (
        OPENVLA_V01_SYSTEM_PROMPT,
        get_processor,
        get_vla,
    )
    from robot_utils import invert_gripper_action, normalize_gripper_action
    from libero.libero import benchmark
    from openpi_client import image_tools

    protocol = SharedOptimizationProtocol()
    protocol.validate_frozen_pilot()
    if args.seed != protocol.seed:
        raise ValueError(f"diagnostic seed is frozen to {protocol.seed}")
    openvla_device = torch.device(args.openvla_device)
    pi05_device = torch.device(args.pi05_device)
    if (
        openvla_device.type != "cuda"
        or pi05_device.type != "cuda"
        or openvla_device == pi05_device
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        raise RuntimeError("consistency diagnostic requires two CUDA devices")
    torch.cuda.set_device(openvla_device)
    torch.manual_seed(protocol.seed)
    torch.cuda.manual_seed_all(protocol.seed)
    np.random.seed(protocol.seed)

    output = paths["output"]
    output.mkdir(parents=True)
    probes = load_frozen_primary_probes(
        paths["probes"],
        openvla_device=openvla_device,
        pi05_device=pi05_device,
    )
    openvla_cfg = SimpleNamespace(
        pretrained_checkpoint=paths["openvla_checkpoint"],
        load_in_8bit=False,
        load_in_4bit=False,
    )
    openvla_model = get_vla(openvla_cfg)
    if torch.device(openvla_model.device) != openvla_device:
        raise RuntimeError("OpenVLA loaded on the wrong device")
    freeze_openvla_for_o2(openvla_model)
    processor = get_processor(openvla_cfg)
    pi05 = _load_pi05(paths["openpi"], paths["pi05_checkpoint"], pi05_device)
    torch.cuda.set_device(openvla_device)

    object_config = OBJECTS[protocol.object_name]
    mesh_path = Path(object_config["mesh"]).resolve(strict=True)
    clean_texture_path = Path(object_config["texture"]).resolve(strict=True)
    xml_path = Path(object_config["xml"]).resolve(strict=True)
    texture_binding = resolve_runtime_texture_binding(
        xml_path, clean_texture_path, object_name=protocol.object_name
    )
    renderer = DifferentiableRenderer(
        mesh_path=str(mesh_path),
        orig_texture_path=str(clean_texture_path),
        device=str(openvla_device),
        scale_xyz=parse_mesh_scale(str(xml_path)),
    ).to(openvla_device)
    texture_metadata = load_texture_parameter(renderer.adv_noise, paths["texture"])
    theta_before = renderer.adv_noise.detach().clone()

    checkpoint_name = str(paths["openvla_checkpoint"])

    def openvla_extract(source_rgb: np.ndarray, prompt_text: str):
        pre_crop = resize_policy_pre_crop_canvas(source_rgb)
        policy_rgb = deployment_center_crop_uint8(pre_crop)
        image = Image.fromarray(policy_rgb).convert("RGB")
        if "openvla-v01" in checkpoint_name:
            prompt = (
                f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot "
                f"take to {prompt_text.lower()}? ASSISTANT:"
            )
        else:
            prompt = (
                f"In: What action should the robot take to {prompt_text.lower()}?\nOut:"
            )
        inputs = processor(prompt, image, return_tensors="pt").to(
            openvla_device, dtype=torch.bfloat16
        )
        inputs = ensure_trailing_empty_token(inputs)

        def deploy(raw: np.ndarray) -> np.ndarray:
            return invert_gripper_action(normalize_gripper_action(raw, binarize=True))

        result = extract_openvla_action_representation(
            model=openvla_model,
            model_inputs=inputs,
            unnorm_key=UNNORM_KEY,
            deploy_action=deploy,
        )
        with torch.no_grad():
            coordinates = (
                probes.openvla(result.projected).detach().float().cpu().numpy()[0]
            )
        return coordinates, result.deployed_action.copy()

    suite = benchmark.get_benchmark_dict()[protocol.task_suite]()
    task = suite.get_task(protocol.task_id)
    init_states = suite.get_task_init_states(protocol.task_id)
    if len(init_states) < len(STATE_IDS):
        raise RuntimeError("LIBERO does not provide all frozen training states")
    records: list[dict[str, Any]] = []
    values: dict[str, dict[str, list[np.ndarray]]] = {
        model: {key: [] for key in ("z_clean", "z_adv", "action_clean", "action_adv")}
        for model in ("openvla", "pi05")
    }
    calibration_count = 0
    for state_id in STATE_IDS:
        env, task_description = get_libero_env(task, "openvla", resolution=512)
        try:
            env.reset()
            observation = env.set_init_state(init_states[state_id])
            env.env.sim.forward()
            clean_rgb = get_libero_image(observation, 512)
            clean_image = (
                torch.from_numpy(clean_rgb.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(openvla_device, dtype=torch.float32)
                .div(255.0)
            )
            wrist_rgb = np.ascontiguousarray(
                observation["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            poses = find_target_body_poses(
                env,
                object_config["search"],
                device=openvla_device,
                texture_name=texture_binding.texture_name,
            )
            if not poses:
                raise RuntimeError(f"state {state_id}: no target object instance")
            mvps = tuple(
                get_render_mvp_from_matrix(
                    env, pose.model_matrix, resolution=(512, 512)
                )
                for pose in poses
            )
            rotations = tuple(pose.model_matrix[:3, :3] for pose in poses)
            visibility = capture_frontmost_instance_masks(
                env,
                body_ids=tuple(pose.body_id for pose in poses),
                resolution=512,
            ).to(openvla_device)
            if calibration_count < 5:
                renderer.calibrate_lighting(
                    mvps[0],
                    clean_image,
                    ema=(0.8 if calibration_count else 0.0),
                    model_rot=rotations[0],
                )
                calibration_count += 1
            with torch.no_grad():
                rendered = _build_adv_samples(
                    renderer,
                    {
                        "bg_tensor": clean_image,
                        "mvps": mvps,
                        "model_rotations": rotations,
                        "instance_visibility": visibility,
                    },
                    512,
                )
            if len(rendered) != 1:
                raise RuntimeError("renderer must produce one adversarial image")
            adv_rgb = _uint8_rgb(rendered[0])

            z_o_clean, action_o_clean = openvla_extract(clean_rgb, task_description)
            z_o_adv, action_o_adv = openvla_extract(adv_rgb, task_description)

            frame_id = f"task00-state{state_id:02d}-frame00"
            noise, noise_seed = _noise(protocol.seed, frame_id)

            def pi_raw(base_rgb: np.ndarray) -> dict[str, Any]:
                return {
                    "observation/image": _client_image(base_rgb, image_tools),
                    "observation/wrist_image": _client_image(wrist_rgb, image_tools),
                    "observation/state": _robot_state(observation),
                    "prompt": str(task_description),
                }

            clean_pi = extract_pi05_action_representation(
                policy=pi05.policy,
                model=pi05.model,
                raw_observation=pi_raw(clean_rgb),
                noise=noise.copy(),
            )
            with torch.no_grad():
                z_p_clean = (
                    probes.pi05(clean_pi.projected).detach().float().cpu().numpy()[0]
                )
            adv_pi = extract_pi05_action_representation(
                policy=pi05.policy,
                model=pi05.model,
                raw_observation=pi_raw(adv_rgb),
                noise=noise.copy(),
            )
            with torch.no_grad():
                z_p_adv = (
                    probes.pi05(adv_pi.projected).detach().float().cpu().numpy()[0]
                )

            paired = {
                "openvla": (z_o_clean, z_o_adv, action_o_clean, action_o_adv),
                "pi05": (
                    z_p_clean,
                    z_p_adv,
                    clean_pi.deployed_action.copy(),
                    adv_pi.deployed_action.copy(),
                ),
            }
            for model, model_values in paired.items():
                for key, value in zip(values[model], model_values, strict=True):
                    values[model][key].append(np.asarray(value, dtype=np.float32))
            records.append(
                {
                    "frame_id": frame_id,
                    "state_id": state_id,
                    "pi05_noise_seed": noise_seed,
                    "visible_pixel_counts": [
                        int(visibility[index].sum()) for index in range(len(poses))
                    ],
                }
            )
        finally:
            env.close()

    if len(records) != 10 or calibration_count != 5:
        raise RuntimeError("frozen ten-frame substrate was not reproduced")
    if not torch.equal(theta_before, renderer.adv_noise.detach()):
        raise RuntimeError("consistency diagnostic mutated the texture parameter")
    if any(parameter.grad is not None for parameter in openvla_model.parameters()):
        raise RuntimeError("OpenVLA parameters received gradients")
    if any(parameter.grad is not None for parameter in pi05.model.parameters()):
        raise RuntimeError("PI0Pytorch parameters received gradients")

    results: dict[str, Any] = {}
    for model, probe in (("openvla", probes.openvla), ("pi05", probes.pi05)):
        arrays = {key: np.stack(value) for key, value in values[model].items()}
        results[model] = consistency_metrics(
            ConsistencyInputs(
                **arrays,
                action_mean=probe.action_mean.detach().cpu().numpy(),
                action_safe_std=probe.safe_action_std.detach().cpu().numpy(),
            )
        )

    report = {
        "schema_version": "phase3_probe_action_consistency_v1",
        "status": "COMPLETE_PENDING_SCIENTIFIC_DECISION",
        "scope": "read-only probe-vs-deployed-action consistency; no texture update or rollout",
        "commit": _git_head(),
        "task_suite": protocol.task_suite,
        "task_id": protocol.task_id,
        "state_ids": list(STATE_IDS),
        "seed": protocol.seed,
        "texture_artifact": texture_metadata,
        "probe_artifact": str(paths["probes"]),
        "probe_hashes": probes.hashes,
        "pi05_noise_contract": (
            "same deterministic sha256-derived [10,32] noise for clean/adv per frame"
        ),
        "texture_parameter_unchanged": True,
        "model_parameters_with_grad": {"openvla": 0, "pi05": 0},
        "frames": records,
        "results": results,
    }
    config = {
        "commit": report["commit"],
        "texture_param": str(paths["texture"]),
        "texture_sha256": _sha256(paths["texture"]),
        "probe_artifact_dir": str(paths["probes"]),
        "openvla_checkpoint": str(paths["openvla_checkpoint"]),
        "pi05_checkpoint": str(paths["pi05_checkpoint"]),
        "openvla_device": str(openvla_device),
        "pi05_device": str(pi05_device),
        "optimization_performed": False,
    }
    (output / "diagnostic_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (output / "probe_action_consistency.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        report = _run(args)
    except Exception as error:
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "Phase 3 Probe-Action Consistency — BLOCKED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        traceback.print_exc()
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
