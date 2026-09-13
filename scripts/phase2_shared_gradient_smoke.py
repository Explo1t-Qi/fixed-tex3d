#!/usr/bin/env python3
"""Run the authoritative Phase 2.3 dual-VLA gradient-closure smoke.

Gate A checks a real image leaf through both VLA feature branches.  Gate B
uses the existing Tex3D renderer and performs exactly one optimizer update.
The script does not run policy inference, rollouts, or attack evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
OPENVLA_ROOT = PROJECT_ROOT / "openvla"
EXPECTED_SHARED_COMMIT = "176c46c2b13e8a647a7f513ea7583cefe0ee44ba"
EXPECTED_MAPPING_SHA256 = (
    "572d4772432025f130ecf0403562bab20a20d4bec008c778985b2b9aee28caec"
)
PI05_CONFIG = "pi05_libero"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.3 real dual-VLA and one-step renderer smoke"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-feature-root", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--openvla-checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--openvla-device", default="cuda:0")
    parser.add_argument("--pi05-device", default="cuda:1")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--optimizer-lr", type=float, default=0.05)
    return parser.parse_args(argv)


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_rgb(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _add_source_paths(openpi_root: Path, shared_root: Path) -> None:
    paths = (
        ROBOT_ROOT / "libero",
        ROBOT_ROOT,
        OPENVLA_ROOT,
        shared_root,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required source paths are missing: {missing}")
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _float(value: torch.Tensor) -> float:
    return float(value.detach().item())


def _tensor_description(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(torch.isfinite(value).all()),
        "requires_grad": bool(value.requires_grad),
    }


def _gradient_description(value: torch.Tensor) -> dict[str, Any]:
    return {
        "exists": True,
        "finite": bool(torch.isfinite(value).all()),
        "nonzero": bool(torch.any(value != 0)),
        "norm": float(value.norm().detach().item()),
    }


def _loss_description(result: Any) -> dict[str, float]:
    return {
        "loss_shared": _float(result.loss),
        "shared_mse": _float(result.shared_mse),
        "o2_mse": _float(result.o2_mse),
        "p2_mse": _float(result.p2_mse),
        "displacement_cosine_mean": _float(
            result.displacement_cosine_mean
        ),
        "o2_to_p2_mse_ratio": _float(result.o2_to_p2_mse_ratio),
    }


def _require_tensor_gradient(value: torch.Tensor | None, label: str) -> torch.Tensor:
    from phase2_shared_gradient import Phase2GradientClosureError

    if value is None:
        raise Phase2GradientClosureError(f"{label} gradient is missing")
    if not bool(torch.isfinite(value).all()):
        raise Phase2GradientClosureError(f"{label} gradient is non-finite")
    if not bool(torch.any(value != 0)):
        raise Phase2GradientClosureError(f"{label} gradient is zero")
    return value


def _client_image(oriented_rgb: np.ndarray, image_tools: Any) -> np.ndarray:
    """Reproduce Phase 1 client preprocessing from an already oriented view."""

    resized = image_tools.resize_with_pad(oriented_rgb, 224, 224)
    result = image_tools.convert_to_uint8(resized)
    if result.shape != (224, 224, 3) or result.dtype != np.uint8:
        raise RuntimeError("OpenPI client preprocessing produced malformed RGB")
    return result


def _robot_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    from libero_utils import quat2axisangle

    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat2axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise RuntimeError("LIBERO robot state must be finite [8]")
    return state


def _load_pi05(openpi_root: Path, checkpoint: Path, device: torch.device) -> Any:
    del openpi_root
    from openpi.models import model as openpi_model
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    from openpi_client import image_tools

    train_config = training_config.get_config(PI05_CONFIG)
    config = train_config.model
    if (
        train_config.name != PI05_CONFIG
        or config.pi05 is not True
        or config.action_horizon != 10
        or config.action_dim != 32
        or config.discrete_state_input is not False
        or config.max_token_len != 200
    ):
        raise RuntimeError("pi05_libero configuration violates Phase 1 semantics")
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        pytorch_device=str(device),
    )
    if getattr(policy, "_is_pytorch_model", None) is not True:
        raise RuntimeError("PI0Pytorch checkpoint did not load with the Torch backend")
    model = policy._model
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("PI0Pytorch model is not a torch.nn.Module")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("PI0Pytorch model is not frozen in eval mode")
    return SimpleNamespace(
        policy=policy,
        model=model,
        observation_type=openpi_model.Observation,
        image_tools=image_tools,
    )


def _gate_a(pipeline: Any, clean_image: torch.Tensor, seed: int) -> dict[str, Any]:
    from phase2_shared_gradient import checked_gradient, gradient_cosine

    clean = pipeline.clean_reference(clean_image)
    generator = torch.Generator(device=clean_image.device).manual_seed(seed)
    perturbation = torch.empty_like(clean_image).uniform_(
        -1.0 / 255.0, 1.0 / 255.0, generator=generator
    )
    adversarial = (clean_image + perturbation).clamp(0.0, 1.0).detach()
    adversarial.requires_grad_(True)
    result, features = pipeline.loss(adversarial, clean)

    openvla_gradient = checked_gradient(
        result.o2_mse,
        adversarial,
        label="OpenVLA branch image",
        retain_graph=True,
    )
    pi05_gradient = checked_gradient(
        result.p2_mse,
        adversarial,
        label="PI0Pytorch branch image",
        retain_graph=True,
    )
    result.loss.backward()
    joint_gradient = _require_tensor_gradient(
        adversarial.grad, "joint shared-loss image"
    )
    if clean.h_o.requires_grad or clean.h_p.requires_grad or clean_image.grad is not None:
        raise RuntimeError("clean reference branch received an optimization gradient")

    report = {
        "status": "PASS",
        "same_image_before_model_preprocessing": True,
        "source_image_device": str(adversarial.device),
        "features": {
            "o2": _tensor_description(features.o2),
            "p2": _tensor_description(features.p2),
            "h_o": _tensor_description(features.h_o),
            "h_p": _tensor_description(features.h_p),
        },
        "diagnostics": _loss_description(result),
        "gradients": {
            "openvla_branch_image": _gradient_description(openvla_gradient),
            "pi05_branch_image": _gradient_description(pi05_gradient),
            "joint_shared_loss_image": _gradient_description(joint_gradient),
            "branch_image_gradient_cosine": gradient_cosine(
                openvla_gradient, pi05_gradient
            ),
        },
        "clean_requires_grad": {
            "h_o": bool(clean.h_o.requires_grad),
            "h_p": bool(clean.h_p.requires_grad),
        },
    }
    del result, features, adversarial
    return {"clean": clean, "report": report}


def _gate_b(
    *,
    pipeline: Any,
    clean: Any,
    renderer: Any,
    frame_data: dict[str, Any],
    optimizer_lr: float,
    output_dir: Path,
) -> dict[str, Any]:
    from attack_openvla import _build_adv_samples
    from phase2_shared_gradient import checked_gradient, gradient_cosine

    optimizer = torch.optim.Adam([renderer.get_texture_param()], lr=optimizer_lr)
    texture_before = renderer.get_texture_param().detach().clone()
    torch.save(texture_before.cpu(), output_dir / "texture_before.pt")

    optimizer.zero_grad(set_to_none=True)
    samples = _build_adv_samples(renderer, frame_data, 512)
    if len(samples) != 1:
        raise RuntimeError("Phase 2.3 renderer smoke requires one adversarial image")
    adversarial = samples[0]
    adversarial.retain_grad()
    if not bool(torch.isfinite(adversarial).all()):
        raise RuntimeError("renderer adversarial image is non-finite")
    result, features = pipeline.loss(adversarial, clean)

    openvla_gradient = checked_gradient(
        result.o2_mse,
        adversarial,
        label="Gate B OpenVLA branch image",
        retain_graph=True,
    )
    pi05_gradient = checked_gradient(
        result.p2_mse,
        adversarial,
        label="Gate B PI0Pytorch branch image",
        retain_graph=True,
    )
    result.loss.backward()
    image_gradient = _require_tensor_gradient(
        adversarial.grad, "Gate B shared-loss image"
    )
    texture_gradient = _require_tensor_gradient(
        renderer.get_texture_param().grad, "Gate B renderer texture"
    )
    texture_gradient_norm = float(texture_gradient.norm().detach().item())

    optimizer.step()
    texture_after = renderer.get_texture_param().detach().clone()
    update_norm = float((texture_after - texture_before).norm().item())
    if not np.isfinite(update_norm) or update_norm <= 0.0:
        raise RuntimeError("one optimizer step did not change the texture parameter")
    torch.save(texture_after.cpu(), output_dir / "texture_after.pt")

    if clean.h_o.requires_grad or clean.h_p.requires_grad:
        raise RuntimeError("clean reference branch received an optimization gradient")
    return {
        "status": "PASS",
        "optimizer": "torch.optim.Adam",
        "optimizer_steps": 1,
        "optimizer_lr": optimizer_lr,
        "adv_image": _tensor_description(adversarial),
        "features": {
            "o2": _tensor_description(features.o2),
            "p2": _tensor_description(features.p2),
            "h_o": _tensor_description(features.h_o),
            "h_p": _tensor_description(features.h_p),
        },
        "diagnostics_before_step": _loss_description(result),
        "gradients": {
            "openvla_branch_image": _gradient_description(openvla_gradient),
            "pi05_branch_image": _gradient_description(pi05_gradient),
            "joint_shared_loss_image": _gradient_description(image_gradient),
            "renderer_texture": _gradient_description(texture_gradient),
            "branch_image_gradient_cosine": gradient_cosine(
                openvla_gradient, pi05_gradient
            ),
        },
        "texture_gradient_norm": texture_gradient_norm,
        "texture_update_norm": update_norm,
        "texture_changed": True,
        "clean_gradients_none": True,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    shared_root = args.shared_feature_root.expanduser().resolve()
    mapping_dir = args.mapping_dir.expanduser().resolve()
    openpi_root = args.openpi_root.expanduser().resolve()
    libero_root = args.libero_root.expanduser().resolve()
    openvla_checkpoint = args.openvla_checkpoint.expanduser().resolve()
    pi05_checkpoint = args.pi05_checkpoint.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be fresh: {output_dir}")
    if _git_head(shared_root) != EXPECTED_SHARED_COMMIT:
        raise RuntimeError("shared-feature authority is not at the frozen Phase 2.2 commit")
    if args.task_id != 0 or args.state_id != 0:
        raise ValueError("Phase 2.3 frozen smoke case is LIBERO-Spatial task 0/state 0")
    if args.seed != 7 or args.optimizer_lr != 0.05:
        raise ValueError("Phase 2.3 smoke reuses seed=7 and optimizer_lr=0.05")

    openvla_device = torch.device(args.openvla_device)
    pi05_device = torch.device(args.pi05_device)
    if (
        openvla_device.type != "cuda"
        or pi05_device.type != "cuda"
        or openvla_device == pi05_device
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        raise RuntimeError("Phase 2.3 authoritative smoke requires two distinct CUDA GPUs")
    torch.cuda.set_device(openvla_device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ["LIBERO_ROOT"] = str(libero_root)
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    _add_source_paths(openpi_root, shared_root)
    output_dir.mkdir(parents=True)

    from openpi_client import image_tools
    from phase2_shared_gradient import (
        DualVLAFeatureAdapter,
        Pi05BaseImageAdapter,
        extract_pi05_p2_autograd,
    )
    from shared_feature.shared_feature_loss import shared_feature_loss
    from shared_feature.torch_cca_mapping import FrozenSharedCCAMapping

    from attack_openvla import (
        DEFAULT_DEPLOYMENT_VIEW,
        DifferentiableRenderer,
        ExactForwardSurrogateBackwardOpenVLAImageProcessor,
        OBJECTS,
        PolicyViewTransform,
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
    from openvla_utils import get_processor, get_vla
    from step1_o2_p2 import extract_openvla_o2
    from libero.libero import benchmark

    openvla_cfg = SimpleNamespace(
        pretrained_checkpoint=openvla_checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
    )
    openvla_model = get_vla(openvla_cfg)
    if torch.device(openvla_model.device) != openvla_device:
        raise RuntimeError(
            f"OpenVLA loaded on {openvla_model.device}, expected {openvla_device}"
        )
    freeze_openvla_for_o2(openvla_model)
    openvla_processor = get_processor(openvla_cfg)
    openvla_image_processor = (
        ExactForwardSurrogateBackwardOpenVLAImageProcessor.from_checkpoint(
            model=openvla_model, processor=openvla_processor
        )
    )
    policy_view = PolicyViewTransform(DEFAULT_DEPLOYMENT_VIEW)

    pi05 = _load_pi05(openpi_root, pi05_checkpoint, pi05_device)
    mapping_o = FrozenSharedCCAMapping.from_artifact(
        mapping_dir,
        dtype=torch.float32,
        device=openvla_device,
        expected_mapping_sha256=EXPECTED_MAPPING_SHA256,
    )
    mapping_p = FrozenSharedCCAMapping.from_artifact(
        mapping_dir,
        dtype=torch.float32,
        device=pi05_device,
        expected_mapping_sha256=EXPECTED_MAPPING_SHA256,
    )

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    if args.state_id >= len(init_states):
        raise IndexError("LIBERO state id is unavailable")
    env, task_description = get_libero_env(task, "openvla", resolution=512)
    try:
        env.reset()
        observation = env.set_init_state(init_states[args.state_id])
        env.env.sim.forward()
        clean_rgb = get_libero_image(observation, 512)
        if clean_rgb.shape != (512, 512, 3) or clean_rgb.dtype != np.uint8:
            raise RuntimeError("canonical LIBERO base image must be uint8 [512,512,3]")
        wrist_rgb = np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        policy_input = {
            "observation/image": _client_image(clean_rgb, image_tools),
            "observation/wrist_image": _client_image(wrist_rgb, image_tools),
            "observation/state": _robot_state(observation),
            "prompt": str(task_description),
        }
        pi05_inputs = Pi05BaseImageAdapter(
            policy=pi05.policy,
            observation_type=pi05.observation_type,
            policy_input=policy_input,
            device=pi05_device,
        )
        clean_image = (
            torch.from_numpy(clean_rgb.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(openvla_device, dtype=torch.float32)
            .div(255.0)
        )

        def openvla_path(image: torch.Tensor) -> torch.Tensor:
            pixels = openvla_image_processor(policy_view(image)).to(torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return extract_openvla_o2(openvla_model, pixels)

        def pi05_path(image: torch.Tensor) -> torch.Tensor:
            model_observation = pi05_inputs.observation_for_base_image(image)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return extract_pi05_p2_autograd(pi05.model, model_observation)

        pipeline = DualVLAFeatureAdapter(
            openvla_path=openvla_path,
            pi05_path=pi05_path,
            openvla_mapping=mapping_o,
            pi05_mapping=mapping_p,
            shared_loss=shared_feature_loss,
            openvla_device=openvla_device,
            pi05_device=pi05_device,
            loss_device=openvla_device,
        )

        gate_a = _gate_a(pipeline, clean_image, args.seed)
        torch.cuda.empty_cache()

        object_config = OBJECTS["akita_black_bowl"]
        mesh_path = Path(object_config["mesh"])
        texture_path = Path(object_config["texture"])
        xml_path = Path(object_config["xml"])
        texture_binding = resolve_runtime_texture_binding(
            xml_path, texture_path, object_name="akita_black_bowl"
        )
        renderer = DifferentiableRenderer(
            mesh_path=str(mesh_path),
            orig_texture_path=str(texture_path),
            device=str(openvla_device),
            scale_xyz=parse_mesh_scale(str(xml_path)),
        ).to(openvla_device)
        initialize_o2_texture_parameter(
            renderer.get_texture_param(), scale=args.optimizer_lr, seed=args.seed
        )
        target_poses = find_target_body_poses(
            env,
            object_config["search"],
            device=openvla_device,
            texture_name=texture_binding.texture_name,
        )
        if not target_poses:
            raise RuntimeError("renderer gate found no target object instances")
        mvps = tuple(
            get_render_mvp_from_matrix(
                env, pose.model_matrix, resolution=(512, 512)
            )
            for pose in target_poses
        )
        rotations = tuple(pose.model_matrix[:3, :3] for pose in target_poses)
        visibility = capture_frontmost_instance_masks(
            env,
            body_ids=tuple(pose.body_id for pose in target_poses),
            resolution=512,
        ).to(openvla_device)
        renderer.calibrate_lighting(
            mvps[0], clean_image, model_rot=rotations[0]
        )
        frame_data = {
            "bg_tensor": clean_image,
            "mvps": mvps,
            "model_rotations": rotations,
            "instance_visibility": visibility,
        }
        gate_b = _gate_b(
            pipeline=pipeline,
            clean=gate_a["clean"],
            renderer=renderer,
            frame_data=frame_data,
            optimizer_lr=args.optimizer_lr,
            output_dir=output_dir,
        )
    finally:
        env.close()

    report = {
        "status": "Phase 2.3 Dual-VLA Gradient Closure — PASS",
        "phase2_3_result": "PASS",
        "scientific_scope": "engineering gradient closure only",
        "repository": {
            "path": str(PROJECT_ROOT),
            "commit": _git_head(PROJECT_ROOT),
        },
        "shared_feature_authority": {
            "path": str(shared_root),
            "commit": _git_head(shared_root),
            "mapping_materialization_id": "phase1_o2_p2_pi05_torch_v1",
            "mapping_sha256": EXPECTED_MAPPING_SHA256,
        },
        "runtime": {
            "openvla_checkpoint": str(openvla_checkpoint),
            "pi05_checkpoint": str(pi05_checkpoint),
            "openvla_device": str(openvla_device),
            "pi05_device": str(pi05_device),
            "renderer_device": str(openvla_device),
            "loss_device": str(openvla_device),
            "task_suite": "libero_spatial",
            "task_id": args.task_id,
            "state_id": args.state_id,
            "batch_size": 1,
            "base_image_sha256": _sha256_rgb(clean_rgb),
            "pi05_base_slot": "base_0_rgb",
            "pi05_wrist_attacked": False,
        },
        "gate_a": gate_a["report"],
        "gate_b": gate_b,
        "claims_excluded": [
            "shared-space vulnerability",
            "attack effectiveness",
            "policy degradation",
            "texture transfer",
            "pi0 transfer success",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = _run(args)
    except Exception as error:
        output_dir = args.output_dir.expanduser().resolve()
        failure = {
            "status": "Phase 2.3 Dual-VLA Gradient Closure — BLOCKED",
            "phase2_3_result": "BLOCKED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if output_dir.is_dir():
            (output_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
