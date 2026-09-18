#!/usr/bin/env python3
"""Run read-only O1/O2/P1/P2 gradient decomposition on a frozen texture."""

from __future__ import annotations

import argparse
import csv
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
OPENVLA_ROOT = PROJECT_ROOT / "openvla"
STATE_IDS = tuple(range(10))


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-param", type=Path, required=True)
    parser.add_argument("--label", default="texture")
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


def _validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    if not args.label.strip():
        raise ValueError("diagnostic label must be non-empty")
    paths = {
        "texture": args.texture_param.expanduser().resolve(strict=True),
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
        ROBOT_ROOT / "libero",
        ROBOT_ROOT,
        OPENVLA_ROOT,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"required diagnostic source paths are missing: {missing}"
        )
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _validate_paths(args)
    os.environ["LIBERO_ROOT"] = str(paths["libero"])
    if str(paths["libero"]) not in sys.path:
        sys.path.insert(0, str(paths["libero"]))
    _add_source_paths(paths["openpi"])

    from phase2_shared_gradient_smoke import (
        _client_image,
        _load_pi05,
        _robot_state,
    )
    from phase2_multilevel_gradient_diagnostic import (
        COMPONENTS,
        GradientDiagnosticFrame,
        component_losses,
        load_texture_parameter,
        run_gradient_decomposition,
    )
    from phase2_multilevel_native_gradient_ensemble import (
        DualVLAMultiLevelNativeFeatureAdapter,
        extract_openvla_o1_o2_autograd,
        extract_pi05_p1_p2_autograd,
    )
    from phase2_shared_gradient import Pi05BaseImageAdapter
    from phase2_shared_optimization import SharedOptimizationProtocol
    from attack_openvla import (
        DEFAULT_DEPLOYMENT_VIEW,
        DifferentiableRenderer,
        ExactForwardSurrogateBackwardOpenVLAImageProcessor,
        OBJECTS,
        PolicyViewTransform,
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
    from openvla_utils import get_processor, get_vla
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
        raise RuntimeError("gradient diagnostic requires two distinct CUDA devices")
    torch.cuda.set_device(openvla_device)
    torch.manual_seed(protocol.seed)
    torch.cuda.manual_seed_all(protocol.seed)
    np.random.seed(protocol.seed)

    output = paths["output"]
    output.mkdir(parents=True)

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
    image_processor = (
        ExactForwardSurrogateBackwardOpenVLAImageProcessor.from_checkpoint(
            model=openvla_model,
            processor=processor,
        )
    )
    policy_view = PolicyViewTransform(DEFAULT_DEPLOYMENT_VIEW)
    pi05 = _load_pi05(paths["openpi"], paths["pi05_checkpoint"], pi05_device)
    torch.cuda.set_device(openvla_device)

    object_config = OBJECTS[protocol.object_name]
    mesh_path = Path(object_config["mesh"]).resolve(strict=True)
    clean_texture_path = Path(object_config["texture"]).resolve(strict=True)
    xml_path = Path(object_config["xml"]).resolve(strict=True)
    texture_binding = resolve_runtime_texture_binding(
        xml_path,
        clean_texture_path,
        object_name=protocol.object_name,
    )
    renderer = DifferentiableRenderer(
        mesh_path=str(mesh_path),
        orig_texture_path=str(clean_texture_path),
        device=str(openvla_device),
        scale_xyz=parse_mesh_scale(str(xml_path)),
    ).to(openvla_device)
    parameter = renderer.adv_noise
    renderer_parameter = renderer.get_texture_param()
    if parameter.data_ptr() != renderer_parameter.data_ptr():
        raise RuntimeError("renderer texture parameter is not renderer.adv_noise")
    texture_metadata = load_texture_parameter(parameter, paths["texture"])
    theta_loaded = parameter.detach().clone()

    def openvla_path(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = image_processor(policy_view(image)).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return extract_openvla_o1_o2_autograd(openvla_model, pixels)

    suite = benchmark.get_benchmark_dict()[protocol.task_suite]()
    task = suite.get_task(protocol.task_id)
    init_states = suite.get_task_init_states(protocol.task_id)
    if len(init_states) <= STATE_IDS[-1]:
        raise RuntimeError("LIBERO does not provide all diagnostic states")

    frames: list[GradientDiagnosticFrame] = []
    frame_contract: list[dict[str, Any]] = []
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
            pi_inputs = Pi05BaseImageAdapter(
                policy=pi05.policy,
                observation_type=pi05.observation_type,
                policy_input={
                    "observation/image": _client_image(clean_rgb, image_tools),
                    "observation/wrist_image": _client_image(wrist_rgb, image_tools),
                    "observation/state": _robot_state(observation),
                    "prompt": str(task_description),
                },
                device=pi05_device,
            )

            def pi05_path(
                image: torch.Tensor,
                adapter: Pi05BaseImageAdapter = pi_inputs,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                model_observation = adapter.observation_for_base_image(image)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return extract_pi05_p1_p2_autograd(pi05.model, model_observation)

            pipeline = DualVLAMultiLevelNativeFeatureAdapter(
                openvla_path=openvla_path,
                pi05_path=pi05_path,
                openvla_device=openvla_device,
                pi05_device=pi05_device,
            )
            clean = pipeline.clean_reference(clean_image)
            if any(
                value.requires_grad
                for value in (clean.o1, clean.o2, clean.p1, clean.p2)
            ):
                raise RuntimeError("clean multi-level reference is not detached")

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

            frame_id = f"task00-state{state_id:02d}-frame00"
            frames.append(
                GradientDiagnosticFrame(
                    frame_id=frame_id,
                    payload={
                        "pipeline": pipeline,
                        "clean": clean,
                        "renderer_frame": {
                            "bg_tensor": clean_image,
                            "mvps": mvps,
                            "model_rotations": rotations,
                            "instance_visibility": visibility,
                        },
                    },
                )
            )
            frame_contract.append(
                {
                    "frame_id": frame_id,
                    "state_id": state_id,
                    "pi05_template_id": f"pi05-state-{state_id:02d}",
                    "clean_reference_detached": True,
                    "visible_pixel_counts": [
                        int(visibility[index].sum()) for index in range(len(poses))
                    ],
                }
            )
        finally:
            env.close()

    if len(frames) != len(STATE_IDS) or calibration_count != 5:
        raise RuntimeError("frozen diagnostic frame substrate was not materialized")
    if not torch.equal(theta_loaded, parameter.detach()):
        raise RuntimeError("frame materialization mutated the texture parameter")

    def forward_frame(frame: GradientDiagnosticFrame):
        images = _build_adv_samples(renderer, frame.payload["renderer_frame"], 512)
        if len(images) != 1 or not bool(torch.isfinite(images[0]).all()):
            raise RuntimeError(f"{frame.frame_id}: invalid rendered image")
        features = frame.payload["pipeline"].features(images[0])
        return component_losses(features, frame.payload["clean"])

    result = run_gradient_decomposition(
        parameter=parameter,
        frames=frames,
        forward_frame=forward_frame,
    )
    if not torch.equal(theta_loaded, parameter.detach()):
        raise RuntimeError("gradient diagnostic mutated the loaded texture parameter")

    config = {
        "status": "Phase 2 Multi-Level Gradient Diagnostic — CONFIGURED",
        "commit": _git_head(),
        "texture_artifact": texture_metadata["absolute_path"],
        "texture_sha256": texture_metadata["sha256"],
        "texture_shape": texture_metadata["shape"],
        "texture_dtype": texture_metadata["dtype"],
        "label": args.label,
        "task_suite": protocol.task_suite,
        "task_id": protocol.task_id,
        "object_name": protocol.object_name,
        "state_ids": list(STATE_IDS),
        "batch_size": len(frames),
        "seed": protocol.seed,
        "openvla_checkpoint": str(paths["openvla_checkpoint"]),
        "pi05_checkpoint": str(paths["pi05_checkpoint"]),
        "openvla_device": str(openvla_device),
        "pi05_device": str(pi05_device),
        "component_losses": {
            "O1": "-MSE(O1_adv, O1_clean)",
            "O2": "-MSE(O2_adv, O2_clean)",
            "P1": "-MSE(P1_adv, P1_clean)",
            "P2": "-MSE(P2_adv, P2_clean)",
        },
        "frame_aggregation": "raw component gradient mean over all 10 frames",
        "optimization_performed": False,
    }
    report = {
        "status": "Phase 2 Multi-Level Gradient Diagnostic — COMPLETE",
        "commit": config["commit"],
        "texture_artifact": config["texture_artifact"],
        "texture_sha256": config["texture_sha256"],
        "label": args.label,
        "task_suite": protocol.task_suite,
        "task_id": protocol.task_id,
        "state_ids": list(STATE_IDS),
        **result,
    }

    (output / "diagnostic_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (output / "gradient_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "frame_contract.json").write_text(
        json.dumps(frame_contract, indent=2, sort_keys=True) + "\n"
    )
    with (output / "cosine_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component", *COMPONENTS])
        for name in COMPONENTS:
            writer.writerow(
                [name, *(report["cosine_matrix"][name][other] for other in COMPONENTS)]
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        report = _run(args)
    except Exception as error:
        output = args.output_dir.expanduser().resolve()
        if output.exists():
            failure_path = output.parent / f"{output.name}.failure.json"
        else:
            output.mkdir(parents=True, exist_ok=True)
            failure_path = output / "failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "status": "Phase 2 Multi-Level Gradient Diagnostic — FAILED",
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
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
