#!/usr/bin/env python3
"""Run explicit OpenVLA-only seed-7 O2 action-aware texture ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
OPENVLA_ROOT = PROJECT_ROOT / "openvla"
OBJECTIVE = "openvla_action_predictive"


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-artifact-dir", type=Path, required=True)
    parser.add_argument("--openvla-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--openvla-device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--pgd-step", type=float, default=0.05)
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument(
        "--smoke", action="store_true", help="Engineering smoke, at most 10 steps"
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "probe": args.probe_artifact_dir.expanduser().resolve(strict=True),
        "openvla_checkpoint": args.openvla_checkpoint.expanduser().resolve(strict=True),
        "libero": args.libero_root.expanduser().resolve(strict=True),
        "output": args.output_dir.expanduser().resolve(),
    }
    if paths["output"].exists():
        raise FileExistsError(f"output directory must be fresh: {paths['output']}")
    if args.smoke and (args.iterations < 1 or args.iterations > 10):
        raise ValueError("engineering smoke requires 1–10 iterations")
    return paths


def _source_paths(libero_root: Path) -> None:
    paths = (ROBOT_ROOT / "libero", ROBOT_ROOT, OPENVLA_ROOT, libero_root)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"OpenVLA-only source paths missing: {missing}")
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _parse_checkpoint_steps(value: str, *, iterations: int) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        steps = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("checkpoint steps must be comma-separated integers") from error
    if not steps or any(step < 1 or step > iterations for step in steps):
        raise ValueError(f"checkpoint steps must be within [1,{iterations}]")
    return steps


def _validate_checkpoint_provenance(checkpoint: Path, metadata: dict[str, Any]) -> None:
    if str(checkpoint) != metadata["models"]["openvla"]["checkpoint_path"]:
        raise RuntimeError("OpenVLA checkpoint differs from frozen probe provenance")


def _save_checkpoint(
    renderer: Any, output: Path, step: int, row: dict[str, Any]
) -> dict[str, Any]:
    directory = output / "checkpoints" / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    parameter = directory / "vertex_noise.pt"
    texture = directory / "attack_texture.png"
    torch.save(renderer.get_texture_param().detach().cpu(), parameter)
    with torch.no_grad():
        baked = renderer.get_baked_adv_texture()[0].cpu().numpy()
    Image.fromarray((baked * 255).round().clip(0, 255).astype(np.uint8)).save(texture)
    result = {
        "completed_step": step,
        "diagnostics": row,
        "vertex_noise": str(parameter.resolve()),
        "vertex_noise_sha256": _sha256(parameter),
        "baked_uv_texture": str(texture.resolve()),
        "baked_uv_texture_sha256": _sha256(texture),
    }
    (directory / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _validate_paths(args)
    os.environ["LIBERO_ROOT"] = str(paths["libero"])
    _source_paths(paths["libero"])

    from attack_openvla import (
        DEFAULT_DEPLOYMENT_VIEW,
        OBJECTS,
        DifferentiableRenderer,
        ExactForwardSurrogateBackwardOpenVLAImageProcessor,
        PolicyViewTransform,
        _build_adv_samples,
        capture_frontmost_instance_masks,
        find_target_body_poses,
        freeze_openvla_for_o2,
        get_libero_env,
        get_libero_image,
        get_render_mvp_from_matrix,
        initialize_o2_texture_parameter,
        parse_mesh_scale,
        resolve_runtime_texture_binding,
    )
    from libero.libero import benchmark
    from openvla_utils import get_processor, get_vla
    from phase2_shared_optimization import (
        SharedOptimizationProtocol,
        validate_texture_budget,
    )
    from phase3_action_predictive_objective import load_frozen_openvla_probe
    from phase3_openvla_action_predictive import (
        OpenVLAReference,
        OpenVLATrainingFrame,
        calibrate_openvla_lambda,
        openvla_action_losses,
        train_openvla_action_predictive,
    )
    from step1_o2_p2 import extract_openvla_o2

    protocol = SharedOptimizationProtocol(
        attack_iterations=args.iterations, pgd_step=args.pgd_step
    )
    run_configuration = protocol.validate_training_configuration()
    if args.smoke and args.iterations != 10:
        raise ValueError("controlled 10-step smoke must run exactly 10 iterations")
    checkpoints = _parse_checkpoint_steps(
        args.checkpoint_steps, iterations=args.iterations
    )
    device = torch.device(args.openvla_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("OpenVLA-only Phase 3 requires a CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(protocol.seed)
    torch.cuda.manual_seed_all(protocol.seed)
    np.random.seed(protocol.seed)
    artifact = load_frozen_openvla_probe(paths["probe"], device=device)
    _validate_checkpoint_provenance(paths["openvla_checkpoint"], artifact.metadata)
    weight_snapshot = artifact.openvla.weight.detach().clone()

    output = paths["output"]
    output.mkdir(parents=True)
    config_path = output / "training_config.json"
    config = {
        **vars(protocol),
        "objective": OBJECTIVE,
        "run_configuration": run_configuration,
        "mode": "fixed-seed-7 exploratory OpenVLA-only action-aware ablation",
        "smoke": args.smoke,
        "checkpoint_steps": list(checkpoints),
        "effective_frame_pool": 10,
        "effective_batch_size": 10,
        "optimizer": "sign_pgd",
        "renderer_epsilon": 128 / 255,
        "texture_parameterization": "tanh(adv_noise) * epsilon",
        "tex3d_commit": _git_head(PROJECT_ROOT),
        "openvla_checkpoint": str(paths["openvla_checkpoint"]),
        "openvla_device": str(device),
        "probe_artifact": str(paths["probe"]),
        "probe_hashes": artifact.hashes,
        "coordinates": "mean_token(O2) @ W_O2.T",
        "magnitude_loss": "-MSE(Z_adv, Z_clean)",
        "direction_loss": "cosine_similarity(Z_adv, Z_clean, eps=1e-8)",
        "combined_loss": "L_mag + lambda_dir * L_dir",
        "gradient_aggregation": "mean frame OpenVLA gradients; no model normalization or ensemble",
        "pi05_loaded": False,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    openvla_cfg = SimpleNamespace(
        pretrained_checkpoint=paths["openvla_checkpoint"],
        load_in_8bit=False,
        load_in_4bit=False,
    )
    model = get_vla(openvla_cfg)
    if torch.device(model.device) != device:
        raise RuntimeError("OpenVLA loaded on wrong device")
    freeze_openvla_for_o2(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("OpenVLA parameters are not frozen")
    processor = get_processor(openvla_cfg)
    image_processor = (
        ExactForwardSurrogateBackwardOpenVLAImageProcessor.from_checkpoint(
            model=model, processor=processor
        )
    )
    policy_view = PolicyViewTransform(DEFAULT_DEPLOYMENT_VIEW)

    object_config = OBJECTS[protocol.object_name]
    mesh = Path(object_config["mesh"]).resolve(strict=True)
    clean_texture = Path(object_config["texture"]).resolve(strict=True)
    xml = Path(object_config["xml"]).resolve(strict=True)
    binding = resolve_runtime_texture_binding(
        xml, clean_texture, object_name=protocol.object_name
    )
    renderer = DifferentiableRenderer(
        mesh_path=str(mesh),
        orig_texture_path=str(clean_texture),
        device=str(device),
        scale_xyz=parse_mesh_scale(str(xml)),
    ).to(device)
    if float(renderer.epsilon) != 128 / 255:
        raise RuntimeError("renderer epsilon differs from current Tex3D")
    initialize_o2_texture_parameter(
        renderer.get_texture_param(), scale=protocol.pgd_step, seed=protocol.seed
    )

    suite = benchmark.get_benchmark_dict()[protocol.task_suite]()
    task = suite.get_task(protocol.task_id)
    init_states = suite.get_task_init_states(protocol.task_id)
    if len(init_states) < protocol.num_train_init_states:
        raise RuntimeError("LIBERO does not provide all frozen training states")

    def o2_path(image: torch.Tensor) -> torch.Tensor:
        pixels = image_processor(policy_view(image)).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return extract_openvla_o2(model, pixels)

    frames = []
    frame_contract = []
    calibration_count = 0
    for state_id in range(protocol.num_train_init_states):
        env, _ = get_libero_env(task, "openvla", resolution=512)
        try:
            env.reset()
            observation = env.set_init_state(init_states[state_id])
            env.env.sim.forward()
            clean_rgb = get_libero_image(observation, 512)
            clean_image = (
                torch.from_numpy(clean_rgb.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device, dtype=torch.float32)
                .div(255.0)
            )
            with torch.no_grad():
                clean = OpenVLAReference.capture(o2_path(clean_image), artifact.openvla)
            poses = find_target_body_poses(
                env,
                object_config["search"],
                device=device,
                texture_name=binding.texture_name,
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
                env, body_ids=tuple(pose.body_id for pose in poses), resolution=512
            ).to(device)
            if calibration_count < 5:
                renderer.calibrate_lighting(
                    mvps[0],
                    clean_image,
                    ema=(0.8 if calibration_count else 0.0),
                    model_rot=rotations[0],
                )
                calibration_count += 1
            frame_id = f"task00-state{state_id:02d}-frame00"
            frames.append(
                OpenVLATrainingFrame(
                    frame_id=frame_id,
                    state_id=state_id,
                    clean=clean,
                    payload={
                        "bg_tensor": clean_image,
                        "mvps": mvps,
                        "model_rotations": rotations,
                        "instance_visibility": visibility,
                    },
                )
            )
            frame_contract.append(
                {
                    "frame_id": frame_id,
                    "state_id": state_id,
                    "instance_names": [pose.body_name for pose in poses],
                    "visible_pixel_counts": [
                        int(visibility[i].sum()) for i in range(len(poses))
                    ],
                    "clean_reference_detached": True,
                    "clean_reference_space": "native_o2_action_predictive_coordinates",
                    "base_image_sha256": hashlib.sha256(
                        np.ascontiguousarray(clean_rgb).tobytes()
                    ).hexdigest(),
                }
            )
        finally:
            env.close()
    if len(frames) != 10 or calibration_count != 5:
        raise RuntimeError("frozen frame-pool/calibration contract not materialized")
    (output / "frame_contract.json").write_text(
        json.dumps(frame_contract, indent=2, sort_keys=True) + "\n"
    )

    def forward_frame(frame: Any, lambda_dir: float) -> Any:
        images = _build_adv_samples(renderer, frame.payload, 512)
        if len(images) != 1 or not bool(torch.isfinite(images[0]).all()):
            raise RuntimeError(f"{frame.frame_id}: invalid renderer image")
        return openvla_action_losses(
            o2_path(images[0]),
            frame.clean,
            probe=artifact.openvla,
            lambda_dir=lambda_dir,
        )

    calibration = calibrate_openvla_lambda(
        renderer=renderer, frames=frames, forward_frame=forward_frame
    )
    (output / "lambda_calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n"
    )
    config["lambda_dir"] = calibration["lambda_dir"]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    saved = []

    def progress(row: dict[str, Any]) -> None:
        completed = int(row["iteration"]) + 1
        if completed in checkpoints:
            saved.append(_save_checkpoint(renderer, output, completed, row))
        print(json.dumps(row, sort_keys=True), flush=True)

    history = train_openvla_action_predictive(
        renderer=renderer,
        frames=frames,
        forward_frame=forward_frame,
        lambda_dir=float(calibration["lambda_dir"]),
        iterations=protocol.attack_iterations,
        requested_batch_size=protocol.num_frames_to_attack,
        pgd_step=protocol.pgd_step,
        seed=protocol.seed,
        metrics_path=output / "step_metrics.jsonl",
        progress=progress,
    )
    maximum = validate_texture_budget(renderer)
    if not torch.equal(weight_snapshot, artifact.openvla.weight):
        raise RuntimeError("frozen OpenVLA O2 probe weight changed during training")
    parameter = output / "final_vertex_noise.pt"
    texture = output / "final_attack_texture.png"
    torch.save(renderer.get_texture_param().detach().cpu(), parameter)
    with torch.no_grad():
        baked = renderer.get_baked_adv_texture()[0].cpu().numpy()
    Image.fromarray((baked * 255).round().clip(0, 255).astype(np.uint8)).save(texture)
    np.save(output / "loss_history.npy", np.asarray([row["loss"] for row in history]))
    summary = {
        "status": "SMOKE_COMPLETE"
        if args.smoke
        else "TRAINING_COMPLETE_PENDING_OPENVLA_ROLLOUT",
        "objective": OBJECTIVE,
        "scope": "fixed-seed-7 exploratory OpenVLA-only source-policy ablation; no transfer claim",
        "iterations_completed": len(history),
        "frame_pool_size": len(frames),
        "effective_batch_size": len(frames),
        "lambda_dir": calibration["lambda_dir"],
        "probe_weights_unchanged": True,
        "pi05_loaded": False,
        "maximum_texture_perturbation": maximum,
        "renderer_epsilon": float(renderer.epsilon),
        "texture_budget_respected": maximum <= float(renderer.epsilon) + 1e-6,
        "final_diagnostics": history[-1],
        "checkpoints": saved,
        "artifacts": {
            "vertex_noise": str(parameter.resolve()),
            "vertex_noise_sha256": _sha256(parameter),
            "baked_uv_texture": str(texture.resolve()),
            "baked_uv_texture_sha256": _sha256(texture),
        },
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    metadata = {
        "schema_version": "phase3_openvla_action_predictive_texture_v1",
        "status": summary["status"],
        "objective": OBJECTIVE,
        "tex3d_commit": _git_head(PROJECT_ROOT),
        "phase2_probe_artifact": str(paths["probe"]),
        "phase2_probe_hashes": artifact.hashes,
        "phase2_probe_schema": artifact.metadata["schema_version"],
        "lambda_calibration": calibration,
        "training_summary": summary,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    inventory = {
        str(path.relative_to(output)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    (output / "artifact_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    output_preexisted = output.exists()
    try:
        report = _run(args)
    except Exception as error:  # noqa: BLE001 - write resumeless failure provenance
        if not output_preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "OPENVLA_ONLY_ACTION_AWARE_BLOCKED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        traceback.print_exc()
        return 1
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
