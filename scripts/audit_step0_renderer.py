#!/usr/bin/env python3
"""Audit Step-0 renderer alignment, visibility, and zero-delta invariants."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = REPOSITORY_ROOT / "openvla"
ROBOT_DIR = OPENVLA_ROOT / "experiments/robot"
LIBERO_DIR = ROBOT_DIR / "libero"
LIBERO_ROOT = LIBERO_DIR / "libero-eval"
for path in (OPENVLA_ROOT, ROBOT_DIR, LIBERO_DIR, LIBERO_ROOT):
    sys.path.insert(0, str(path))

from libero.libero import benchmark  # noqa: E402
from attack_openvla import (  # noqa: E402
    OBJECTS,
    DifferentiableRenderer,
    get_render_mvp_from_matrix,
    parse_mesh_scale,
)
from libero_utils import get_libero_env, get_libero_image  # noqa: E402
from openvla_renderer_contracts import (  # noqa: E402
    capture_frontmost_instance_masks,
    compose_visibility_masked_renderer_delta,
    find_target_body_poses,
)
from openvla_runtime_assets import resolve_runtime_texture_binding  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def _bbox(mask: torch.Tensor) -> list[int] | None:
    points = torch.nonzero(mask, as_tuple=False)
    if points.numel() == 0:
        return None
    y_min, x_min = points.min(dim=0).values.tolist()
    y_max, x_max = points.max(dim=0).values.tolist()
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    reference = reference.bool()
    candidate = candidate.bool()
    intersection = int((reference & candidate).sum().item())
    union = int((reference | candidate).sum().item())
    reference_count = int(reference.sum().item())
    candidate_count = int(candidate.sum().item())
    reference_points = torch.nonzero(reference, as_tuple=False).float()
    candidate_points = torch.nonzero(candidate, as_tuple=False).float()
    center_offset = None
    if reference_points.numel() and candidate_points.numel():
        reference_center = reference_points.mean(dim=0)
        candidate_center = candidate_points.mean(dim=0)
        offset_yx = candidate_center - reference_center
        center_offset = [float(offset_yx[1]), float(offset_yx[0])]
    return {
        "iou": intersection / union if union else 1.0,
        "visible_recall": intersection / reference_count if reference_count else 1.0,
        "reference_visible_pixels": reference_count,
        "renderer_pixels": candidate_count,
        "center_offset_xy": center_offset,
        "reference_bbox_xyxy": _bbox(reference),
        "renderer_bbox_xyxy": _bbox(candidate),
    }


def main() -> int:
    args = _arguments()
    object_configuration = OBJECTS["akita_black_bowl"]
    binding = resolve_runtime_texture_binding(
        object_configuration["xml"],
        object_configuration["texture"],
        object_name="akita_black_bowl",
    )
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    initial_state = suite.get_task_init_states(0)[0]
    env, _ = get_libero_env(task, "openvla", resolution=args.resolution)
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        env.env.sim.forward()
        clean_numpy = get_libero_image(observation, args.resolution)
        clean = (
            torch.from_numpy(clean_numpy.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cuda()
            .float()
            .div(255.0)
        )
        poses = find_target_body_poses(
            env,
            object_configuration["search"],
            device="cuda",
            texture_name=binding.texture_name,
        )
        body_ids = tuple(pose.body_id for pose in poses)
        visibility = capture_frontmost_instance_masks(
            env, body_ids=body_ids, resolution=args.resolution
        ).cuda()
        renderer = DifferentiableRenderer(
            mesh_path=object_configuration["mesh"],
            orig_texture_path=object_configuration["texture"],
            device="cuda",
            scale_xyz=parse_mesh_scale(object_configuration["xml"]),
        ).cuda()
        mvps = tuple(
            get_render_mvp_from_matrix(
                env,
                pose.model_matrix,
                resolution=(args.resolution, args.resolution),
            )
            for pose in poses
        )
        renders = [
            renderer.render(
                mvp,
                resolution=(args.resolution, args.resolution),
                return_clean=True,
                model_rot=pose.model_matrix[:3, :3],
            )
            for mvp, pose in zip(mvps, poses, strict=True)
        ]
        adversarial = torch.cat(
            [item[0].permute(0, 3, 1, 2) for item in renders], dim=0
        )
        renderer_clean = torch.cat(
            [item[1].permute(0, 3, 1, 2) for item in renders], dim=0
        )
        renderer_masks = torch.cat(
            [item[2].permute(0, 3, 1, 2) for item in renders], dim=0
        )
        per_instance = [
            {
                "body_id": pose.body_id,
                "body_name": pose.body_name,
                **_metrics(
                    visibility[index, 0].cpu(),
                    renderer_masks[index, 0].detach().cpu() > 0.5,
                ),
            }
            for index, pose in enumerate(poses)
        ]
        union_metrics = _metrics(
            visibility.sum(dim=0)[0].cpu() > 0.5,
            renderer_masks.sum(dim=0)[0].detach().cpu() > 0.5,
        )
        zero_delta = compose_visibility_masked_renderer_delta(
            clean, visibility, renderer_clean.clone(), renderer_clean, renderer_masks
        )
        zero_delta_linf = float((zero_delta - clean).abs().max().item())

        renderer.adv_noise.grad = None
        composited = compose_visibility_masked_renderer_delta(
            clean, visibility, adversarial, renderer_clean, renderer_masks
        )
        composited.retain_grad()
        loss = composited.square().mean()
        loss.backward()
        image_gradient_norm = float(composited.grad.norm().item())
        texture_gradient = renderer.adv_noise.grad
        texture_gradient_norm = (
            float(texture_gradient.norm().item())
            if texture_gradient is not None
            else 0.0
        )
        evidence = {
            "task_suite": "libero_spatial",
            "task_id": 0,
            "state_id": 0,
            "resolution": args.resolution,
            "position_offset": renderer.pos_offset.detach().cpu().tolist(),
            "instance_count": len(poses),
            "instance_names": [pose.body_name for pose in poses],
            "per_instance": per_instance,
            "union": union_metrics,
            "zero_delta_linf": zero_delta_linf,
            "loss_finite": bool(torch.isfinite(loss)),
            "image_gradient_norm": image_gradient_norm,
            "texture_gradient_norm": texture_gradient_norm,
        }
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    passed = (
        evidence["position_offset"] == [0.0, 0.0, 0.0]
        and evidence["instance_count"] == 2
        and evidence["union"]["iou"] >= 0.95
        and evidence["union"]["visible_recall"] >= 0.95
        and evidence["zero_delta_linf"] <= 1e-6
        and evidence["loss_finite"]
        and evidence["image_gradient_norm"] > 0.0
        and evidence["texture_gradient_norm"] > 0.0
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
