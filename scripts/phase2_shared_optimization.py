#!/usr/bin/env python3
"""Run parameterized Phase 2 source-feature texture optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
EXPECTED_SHARED_COMMIT = "176c46c2b13e8a647a7f513ea7583cefe0ee44ba"
EXPECTED_MAPPING_SHA256 = (
    "572d4772432025f130ecf0403562bab20a20d4bec008c778985b2b9aee28caec"
)
MAPPING_ID = "phase1_o2_p2_pi05_torch_v1"
SHARED_CCA_OBJECTIVE = "shared_cca"
NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE = "native_gradient_ensemble"
MULTILEVEL_NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE = "multilevel_native_gradient_ensemble"
OBJECTIVES = (
    SHARED_CCA_OBJECTIVE,
    NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE,
    MULTILEVEL_NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE,
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=OBJECTIVES, default=SHARED_CCA_OBJECTIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-feature-root", type=Path)
    parser.add_argument("--mapping-dir", type=Path)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--openvla-checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--openvla-device", default="cuda:0")
    parser.add_argument("--pi05-device", default="cuda:1")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--pgd-step", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-steps",
        default="",
        help="Comma-separated completed steps to save, e.g. 500,1000,2000,5000",
    )
    parser.add_argument("--num-train-init-states", type=int, default=10)
    parser.add_argument("--train-frames-per-state", type=int, default=1)
    parser.add_argument("--num-frames-to-attack", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_native_source_paths(openpi_root: Path) -> None:
    """Add fixed-tex3d and OpenPI sources without a shared-feature root."""

    paths = (
        ROBOT_ROOT / "libero",
        ROBOT_ROOT,
        OPENVLA_ROOT,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required native source paths are missing: {missing}")
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


def _save_texture_checkpoint(
    *,
    renderer: Any,
    output: Path,
    step: int,
    optimization_parameters: dict[str, int | float],
    row: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir = output / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    parameter_checkpoint = checkpoint_dir / "vertex_noise.pt"
    texture_checkpoint = checkpoint_dir / "attack_texture.png"
    torch.save(renderer.get_texture_param().detach().cpu(), parameter_checkpoint)
    with torch.no_grad():
        baked_checkpoint = renderer.get_baked_adv_texture()[0].cpu().numpy()
    Image.fromarray(
        (baked_checkpoint * 255.0).round().clip(0, 255).astype(np.uint8)
    ).save(texture_checkpoint)
    metadata = {
        "completed_step": step,
        "optimization_parameters": optimization_parameters,
        "diagnostics": row,
        "vertex_noise": str(parameter_checkpoint.resolve()),
        "vertex_noise_sha256": _sha256(parameter_checkpoint),
        "baked_uv_texture": str(texture_checkpoint.resolve()),
        "baked_uv_texture_sha256": _sha256(texture_checkpoint),
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def _validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "openpi": args.openpi_root.expanduser().resolve(strict=True),
        "openvla_checkpoint": args.openvla_checkpoint.expanduser().resolve(strict=True),
        "pi05_checkpoint": args.pi05_checkpoint.expanduser().resolve(strict=True),
        "libero": args.libero_root.expanduser().resolve(strict=True),
    }
    if args.objective == SHARED_CCA_OBJECTIVE:
        if args.shared_feature_root is None or args.mapping_dir is None:
            raise ValueError(
                "shared_cca requires --shared-feature-root and --mapping-dir"
            )
        paths["shared"] = args.shared_feature_root.expanduser().resolve(strict=True)
        paths["mapping"] = args.mapping_dir.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")
    paths["output"] = output
    return paths


def _run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _validate_paths(args)
    os.environ["LIBERO_ROOT"] = str(paths["libero"])
    if str(paths["libero"]) not in sys.path:
        sys.path.insert(0, str(paths["libero"]))

    # Reuse the exact Phase 2.3 model loading and PI0.5 client preprocessing.
    from phase2_shared_gradient_smoke import (
        _add_source_paths,
        _client_image,
        _git_head,
        _load_pi05,
        _robot_state,
    )

    if args.objective == SHARED_CCA_OBJECTIVE:
        _add_source_paths(paths["openpi"], paths["shared"])
    else:
        _add_native_source_paths(paths["openpi"])
    from phase2_shared_gradient import (
        DualVLAFeatureAdapter,
        Pi05BaseImageAdapter,
        extract_pi05_p2_autograd,
    )
    from phase2_native_gradient_ensemble import (
        DualVLANativeFeatureAdapter,
        NativeTrainingFrame,
        train_native_gradient_ensemble,
    )
    from phase2_multilevel_native_gradient_ensemble import (
        DualVLAMultiLevelNativeFeatureAdapter,
        MultiLevelNativeTrainingFrame,
        extract_openvla_o1_o2_autograd,
        extract_pi05_p1_p2_autograd,
        train_multilevel_native_gradient_ensemble,
    )
    from phase2_shared_optimization import (
        FrozenCleanReference,
        SharedOptimizationProtocol,
        SharedTrainingFrame,
        train_shared_texture,
        validate_texture_budget,
    )
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
        initialize_o2_texture_parameter,
        parse_mesh_scale,
        resolve_runtime_texture_binding,
    )
    from openvla_utils import get_processor, get_vla
    from step1_o2_p2 import extract_openvla_o2
    from libero.libero import benchmark
    from openpi_client import image_tools

    if args.objective == SHARED_CCA_OBJECTIVE:
        from shared_feature.shared_feature_loss import shared_feature_loss
        from shared_feature.torch_cca_mapping import FrozenSharedCCAMapping

    protocol = SharedOptimizationProtocol(
        num_train_init_states=args.num_train_init_states,
        train_frames_per_state=args.train_frames_per_state,
        num_frames_to_attack=args.num_frames_to_attack,
        attack_iterations=args.iterations,
        pgd_step=args.pgd_step,
        seed=args.seed,
    )
    run_configuration = protocol.validate_training_configuration()
    checkpoint_steps = _parse_checkpoint_steps(
        args.checkpoint_steps, iterations=protocol.attack_iterations
    )
    if args.objective == SHARED_CCA_OBJECTIVE:
        if _git_head(paths["shared"]) != EXPECTED_SHARED_COMMIT:
            raise RuntimeError("shared-feature authority is not at the frozen commit")
        mapping_file = paths["mapping"] / "mapping.npz"
        if _sha256(mapping_file) != EXPECTED_MAPPING_SHA256:
            raise RuntimeError("mapping.npz SHA-256 does not match Phase 1 authority")

    openvla_device = torch.device(args.openvla_device)
    pi05_device = torch.device(args.pi05_device)
    if (
        openvla_device.type != "cuda"
        or pi05_device.type != "cuda"
        or openvla_device == pi05_device
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
    ):
        raise RuntimeError("Phase 2.4 requires two distinct CUDA devices")
    torch.cuda.set_device(openvla_device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    output = paths["output"]
    output.mkdir(parents=True)
    config_path = output / "training_config.json"
    config = {
        **vars(protocol),
        "objective": args.objective,
        "run_configuration": run_configuration,
        "checkpoint_steps": list(checkpoint_steps),
        "effective_frame_pool": 10,
        "frame_weight": 0.1,
        "optimizer": "sign_pgd",
        "renderer_epsilon": 128.0 / 255.0,
        "texture_parameterization": "tanh(adv_noise) * epsilon",
        "tex3d_commit": _git_head(PROJECT_ROOT),
        "openvla_checkpoint": str(paths["openvla_checkpoint"]),
        "pi05_checkpoint": str(paths["pi05_checkpoint"]),
        "openvla_device": str(openvla_device),
        "pi05_device": str(pi05_device),
        "uses_shared_feature_artifact": args.objective == SHARED_CCA_OBJECTIVE,
    }
    if args.objective == SHARED_CCA_OBJECTIVE:
        config.update(
            {
                "mapping_materialization_id": MAPPING_ID,
                "mapping_sha256": EXPECTED_MAPPING_SHA256,
                "shared_feature_commit": EXPECTED_SHARED_COMMIT,
            }
        )
    elif args.objective == NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE:
        config["native_gradient_ensemble"] = {
            "openvla_loss": "-mean((O2_adv - O2_clean)^2)",
            "pi05_loss": "-mean((P2_adv - P2_clean)^2)",
            "frame_aggregation": "mean within each model before normalization",
            "model_normalization": "g / (mean(abs(g)) + 1e-12)",
            "model_aggregation": "(g_o_normalized + g_p_normalized) / 2",
        }
    else:
        config["multilevel_native_gradient_ensemble"] = {
            "openvla_loss": "-(MSE(O1-S_adv, O1-S_clean) + MSE(O2_adv, O2_clean))",
            "pi05_loss": "-(MSE(P1_adv, P1_clean) + MSE(P2_adv, P2_clean))",
            "level_weights": {"o1": 1.0, "o2": 1.0, "p1": 1.0, "p2": 1.0},
            "level_normalization": None,
            "frame_aggregation": "mean within each model before normalization",
            "model_normalization": "g / (mean(abs(g)) + 1e-12)",
            "model_aggregation": "(g_o_normalized + g_p_normalized) / 2",
        }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

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
            model=openvla_model, processor=processor
        )
    )
    policy_view = PolicyViewTransform(DEFAULT_DEPLOYMENT_VIEW)
    pi05 = _load_pi05(paths["openpi"], paths["pi05_checkpoint"], pi05_device)
    if args.objective == SHARED_CCA_OBJECTIVE:
        mapping_o = FrozenSharedCCAMapping.from_artifact(
            paths["mapping"],
            dtype=torch.float32,
            device=openvla_device,
            expected_mapping_sha256=EXPECTED_MAPPING_SHA256,
        )
        mapping_p = FrozenSharedCCAMapping.from_artifact(
            paths["mapping"],
            dtype=torch.float32,
            device=pi05_device,
            expected_mapping_sha256=EXPECTED_MAPPING_SHA256,
        )
    # The inherited MVP helper uses ``.cuda()`` without a device argument.
    # Restore the renderer/OpenVLA device after constructing the cuda:1 policy.
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
    if float(renderer.epsilon) != 128.0 / 255.0:
        raise RuntimeError("renderer epsilon differs from current Tex3D")
    initialize_o2_texture_parameter(
        renderer.get_texture_param(), scale=protocol.pgd_step, seed=protocol.seed
    )

    suite = benchmark.get_benchmark_dict()[protocol.task_suite]()
    task = suite.get_task(protocol.task_id)
    init_states = suite.get_task_init_states(protocol.task_id)
    if len(init_states) < protocol.num_train_init_states:
        raise RuntimeError("LIBERO does not provide all frozen training states")

    def openvla_path(image: torch.Tensor) -> torch.Tensor:
        pixels = image_processor(policy_view(image)).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return extract_openvla_o2(openvla_model, pixels)

    def openvla_multilevel_path(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = image_processor(policy_view(image)).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return extract_openvla_o1_o2_autograd(openvla_model, pixels)

    frames: list[
        SharedTrainingFrame | NativeTrainingFrame | MultiLevelNativeTrainingFrame
    ] = []
    frame_contract = []
    calibration_count = 0
    for state_id in range(protocol.num_train_init_states):
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

            def pi05_path(image: torch.Tensor, adapter=pi_inputs) -> torch.Tensor:
                model_observation = adapter.observation_for_base_image(image)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return extract_pi05_p2_autograd(pi05.model, model_observation)

            def pi05_multilevel_path(
                image: torch.Tensor, adapter=pi_inputs
            ) -> tuple[torch.Tensor, torch.Tensor]:
                model_observation = adapter.observation_for_base_image(image)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return extract_pi05_p1_p2_autograd(pi05.model, model_observation)

            if args.objective == SHARED_CCA_OBJECTIVE:
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
                clean_live = pipeline.clean_reference(clean_image)
                clean = FrozenCleanReference.from_tensors(
                    clean_live.h_o, clean_live.h_p
                )
                frame_type = SharedTrainingFrame
                clean_reference_space = "pca_cca_canonical"
            elif args.objective == NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE:
                pipeline = DualVLANativeFeatureAdapter(
                    openvla_path=openvla_path,
                    pi05_path=pi05_path,
                    openvla_device=openvla_device,
                    pi05_device=pi05_device,
                )
                clean = pipeline.clean_reference(clean_image)
                frame_type = NativeTrainingFrame
                clean_reference_space = "native_o2_p2"
            else:
                pipeline = DualVLAMultiLevelNativeFeatureAdapter(
                    openvla_path=openvla_multilevel_path,
                    pi05_path=pi05_multilevel_path,
                    openvla_device=openvla_device,
                    pi05_device=pi05_device,
                )
                clean = pipeline.clean_reference(clean_image)
                frame_type = MultiLevelNativeTrainingFrame
                clean_reference_space = "native_o1s_o2_p1_p2"
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
                env, body_ids=tuple(pose.body_id for pose in poses), resolution=512
            ).to(openvla_device)
            visible_pixels = [int(visibility[i].sum()) for i in range(len(poses))]
            if calibration_count < 5:
                renderer.calibrate_lighting(
                    mvps[0],
                    clean_image,
                    ema=(0.8 if calibration_count else 0.0),
                    model_rot=rotations[0],
                )
                calibration_count += 1
            payload = {
                "pipeline": pipeline,
                "renderer_frame": {
                    "bg_tensor": clean_image,
                    "mvps": mvps,
                    "model_rotations": rotations,
                    "instance_visibility": visibility,
                },
            }
            frame_id = f"task00-state{state_id:02d}-frame00"
            frames.append(
                frame_type(
                    frame_id=frame_id,
                    state_id=state_id,
                    pi05_template_id=f"pi05-state-{state_id:02d}",
                    clean=clean,
                    payload=payload,
                )
            )
            frame_contract.append(
                {
                    "frame_id": frame_id,
                    "state_id": state_id,
                    "pi05_template_id": f"pi05-state-{state_id:02d}",
                    "instance_names": [pose.body_name for pose in poses],
                    "visible_pixel_counts": visible_pixels,
                    "clean_reference_detached": True,
                    "clean_reference_space": clean_reference_space,
                    "base_image_sha256": hashlib.sha256(
                        np.ascontiguousarray(clean_rgb).tobytes()
                    ).hexdigest(),
                    "pi05_wrist_attacked": False,
                }
            )
        finally:
            env.close()
    if len(frames) != 10 or calibration_count != 5:
        raise RuntimeError(
            "frozen frame-pool/calibration contract was not materialized"
        )
    (output / "frame_contract.json").write_text(
        json.dumps(frame_contract, indent=2, sort_keys=True) + "\n"
    )

    def render_frame(
        frame: SharedTrainingFrame
        | NativeTrainingFrame
        | MultiLevelNativeTrainingFrame,
    ):
        images = _build_adv_samples(renderer, frame.payload["renderer_frame"], 512)
        if len(images) != 1:
            raise RuntimeError(f"{frame.frame_id}: renderer must return one base image")
        image = images[0]
        if not bool(torch.isfinite(image).all()):
            raise RuntimeError(f"{frame.frame_id}: rendered image is non-finite")
        return image

    def forward_shared_frame(frame: SharedTrainingFrame):
        image = render_frame(frame)
        return frame.payload["pipeline"].loss(image, frame.clean)[0], image

    def forward_native_frame(frame: NativeTrainingFrame):
        image = render_frame(frame)
        return frame.payload["pipeline"].losses(image, frame.clean)[0], image

    def forward_multilevel_native_frame(frame: MultiLevelNativeTrainingFrame):
        image = render_frame(frame)
        return frame.payload["pipeline"].losses(image, frame.clean)[0], image

    checkpoint_artifacts: list[dict[str, Any]] = []

    def progress(row: dict[str, Any]) -> None:
        completed_step = int(row["iteration"]) + 1
        if completed_step in checkpoint_steps:
            checkpoint_artifacts.append(
                _save_texture_checkpoint(
                    renderer=renderer,
                    output=output,
                    step=completed_step,
                    optimization_parameters={
                        "iterations": protocol.attack_iterations,
                        "pgd_step": protocol.pgd_step,
                    },
                    row=row,
                )
            )
        print(json.dumps(row, sort_keys=True), flush=True)

    training_arguments = {
        "renderer": renderer,
        "frames": frames,
        "iterations": protocol.attack_iterations,
        "requested_batch_size": protocol.num_frames_to_attack,
        "pgd_step": protocol.pgd_step,
        "seed": protocol.seed,
        "metrics_path": output / "step_metrics.jsonl",
        "progress": progress,
    }
    if args.objective == SHARED_CCA_OBJECTIVE:
        history = train_shared_texture(
            forward_frame=forward_shared_frame,
            **training_arguments,
        )
        loss_history = np.asarray([row["loss_shared"] for row in history])
        loss_history_fields = ["loss_shared"]
    elif args.objective == NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE:
        history = train_native_gradient_ensemble(
            forward_frame=forward_native_frame,
            **training_arguments,
        )
        loss_history = np.asarray([[row["loss_o"], row["loss_p"]] for row in history])
        loss_history_fields = ["loss_o", "loss_p"]
    else:
        history = train_multilevel_native_gradient_ensemble(
            forward_frame=forward_multilevel_native_frame,
            **training_arguments,
        )
        loss_history = np.asarray([[row["loss_o"], row["loss_p"]] for row in history])
        loss_history_fields = ["loss_o", "loss_p"]
    maximum = validate_texture_budget(renderer)
    parameter_path = output / "final_vertex_noise.pt"
    texture_path = output / "final_attack_texture.png"
    torch.save(renderer.get_texture_param().detach().cpu(), parameter_path)
    with torch.no_grad():
        baked = renderer.get_baked_adv_texture()[0].cpu().numpy()
    Image.fromarray((baked * 255.0).round().clip(0, 255).astype(np.uint8)).save(
        texture_path
    )
    np.save(output / "loss_history.npy", loss_history)
    summary = {
        "status": {
            SHARED_CCA_OBJECTIVE: "Phase 2.4 Shared-Feature Optimization — COMPLETE",
            NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE: (
                "Phase 2 Native Gradient Ensemble Optimization — COMPLETE"
            ),
            MULTILEVEL_NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE: (
                "Phase 2 Multi-Level Native Gradient Ensemble Optimization — COMPLETE"
            ),
        }[args.objective],
        "phase2_4_training_result": "PASS",
        "objective": args.objective,
        "uses_shared_feature_artifact": args.objective == SHARED_CCA_OBJECTIVE,
        "run_configuration": run_configuration,
        "iterations_completed": len(history),
        "texture_updates": len(history),
        "frame_pool_size": len(frames),
        "effective_batch_size": len(frames),
        "training_state_ids": list(range(protocol.num_train_init_states)),
        "clean_reference_forward_count": len(frames),
        "loss_history_fields": loss_history_fields,
        "final_diagnostics": history[-1],
        "maximum_texture_perturbation": maximum,
        "renderer_epsilon": float(renderer.epsilon),
        "texture_budget_respected": maximum <= float(renderer.epsilon) + 1e-6,
        "checkpoints": checkpoint_artifacts,
        "artifacts": {
            "vertex_noise": str(parameter_path.resolve()),
            "vertex_noise_sha256": _sha256(parameter_path),
            "baked_uv_texture": str(texture_path.resolve()),
            "baked_uv_texture_sha256": _sha256(texture_path),
            "step_metrics": str((output / "step_metrics.jsonl").resolve()),
            "loss_history": str((output / "loss_history.npy").resolve()),
            "frame_contract": str((output / "frame_contract.json").resolve()),
            "training_config": str(config_path.resolve()),
        },
        "scope": {
            SHARED_CCA_OBJECTIVE: (
                "source-model shared-feature texture training; no rollout claim"
            ),
            NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE: (
                "source-model native-gradient-ensemble texture training; no rollout claim"
            ),
            MULTILEVEL_NATIVE_GRADIENT_ENSEMBLE_OBJECTIVE: (
                "source-model hierarchical-native-gradient-ensemble texture training; "
                "no rollout claim"
            ),
        }[args.objective],
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


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
                    "status": "Phase 2 Source-Feature Optimization — BLOCKED",
                    "objective": args.objective,
                    "phase2_4_training_result": "BLOCKED",
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
