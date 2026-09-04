#!/usr/bin/env python3
"""Audit Step-0 preprocessing against the real checkpoint processor/model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = REPOSITORY_ROOT / "openvla"
ROBOT_DIR = OPENVLA_ROOT / "experiments/robot"
LIBERO_DIR = ROBOT_DIR / "libero"
LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT", OPENVLA_ROOT / "libero-eval"))
for path in (OPENVLA_ROOT, ROBOT_DIR, LIBERO_DIR, LIBERO_ROOT):
    sys.path.insert(0, str(path))

from libero.libero import benchmark  # noqa: E402
from libero_utils import get_libero_env, get_libero_image  # noqa: E402
from openvla_image_transform import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_model_inputs import ensure_trailing_empty_token  # noqa: E402
from openvla_policy_view import (  # noqa: E402
    DEFAULT_DEPLOYMENT_VIEW,
    POLICY_SOURCE_RESOLUTION,
    deployment_center_crop_uint8,
    resize_policy_pre_crop_canvas,
)
from openvla_utils import get_processor, get_vla  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    return parser.parse_args()


def _synthetic_images() -> list[tuple[str, np.ndarray]]:
    y, x = np.mgrid[:512, :512]
    return [
        (
            "gradient",
            np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(
                np.uint8
            ),
        ),
        (
            "checkerboard",
            np.repeat(
                ((((x // 16) + (y // 16)) % 2) * 255)[..., None], 3, axis=-1
            ).astype(np.uint8),
        ),
        (
            "seeded_noise",
            np.random.default_rng(7).integers(
                0, 256, size=(512, 512, 3), dtype=np.uint8
            ),
        ),
    ]


def _real_spatial_state_zero() -> tuple[np.ndarray, str]:
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    state = suite.get_task_init_states(0)[0]
    env, _ = get_libero_env(task, "openvla", resolution=POLICY_SOURCE_RESOLUTION)
    try:
        env.reset()
        observation = env.set_init_state(state)
        env.env.sim.forward()
        source = get_libero_image(observation, POLICY_SOURCE_RESOLUTION)
    finally:
        env.close()
    canvas = resize_policy_pre_crop_canvas(
        source, specification=DEFAULT_DEPLOYMENT_VIEW
    )
    return (
        deployment_center_crop_uint8(
            canvas, specification=DEFAULT_DEPLOYMENT_VIEW
        ),
        task.language,
    )


def _tensor_rgb(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div(255.0)
    )


def _decode(model: object, generated_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    action_dim = model.get_action_dim(unnorm_key)
    token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()
    bins = model.vocab_size - token_ids
    bins = np.clip(bins - 1, 0, model.bin_centers.shape[0] - 1)
    normalized = model.bin_centers[bins]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


def _generate(
    model: object,
    processor: object,
    prompt: str,
    image: np.ndarray,
    pixel_values: torch.Tensor,
    unnorm_key: str,
) -> torch.Tensor:
    inputs = processor(prompt, images=Image.fromarray(image)).to(model.device)
    inputs = ensure_trailing_empty_token(inputs)
    inputs["pixel_values"] = pixel_values.to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.generate(
            **inputs,
            max_new_tokens=model.get_action_dim(unnorm_key),
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )


def main() -> int:
    args = _arguments()
    cfg = SimpleNamespace(
        pretrained_checkpoint=args.checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
    )
    model = get_vla(cfg)
    model_training_before_eval = bool(model.training)
    model.eval()
    processor = get_processor(cfg)
    differentiable = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=model, processor=processor
    )
    real_image, task_description = _real_spatial_state_zero()
    cases = _synthetic_images() + [("libero_spatial_task0_state0", real_image)]
    prompt = (
        "In: What action should the robot take to "
        f"{task_description.lower()}?\nOut:"
    )

    rows = []
    for name, image in cases:
        official = processor.image_processor(
            Image.fromarray(image), return_tensors="pt"
        )["pixel_values"].to(model.device)
        candidate = differentiable(_tensor_rgb(image, model.device))
        delta = (candidate - official).abs()
        official_model_input = official.to(torch.bfloat16)
        candidate_model_input = candidate.to(torch.bfloat16)
        model_input_delta = (
            candidate_model_input - official_model_input
        ).abs().float()
        branch_rows = []
        channel_offset = 0
        for branch in differentiable.branches:
            branch_delta = delta[:, channel_offset : channel_offset + 3]
            branch_rows.append(
                {
                    "model_id": branch.model_id,
                    "mae": float(branch_delta.mean().item()),
                    "linf": float(branch_delta.max().item()),
                }
            )
            channel_offset += 3
        official_ids = _generate(
            model, processor, prompt, image, official, args.unnorm_key
        )
        candidate_ids = _generate(
            model, processor, prompt, image, candidate, args.unnorm_key
        )
        official_repeat_ids = (
            _generate(
                model, processor, prompt, image, official, args.unnorm_key
            )
            if name == "libero_spatial_task0_state0"
            else official_ids
        )
        action_dim = model.get_action_dim(args.unnorm_key)
        official_tokens = official_ids[0, -action_dim:]
        candidate_tokens = candidate_ids[0, -action_dim:]
        official_repeat_tokens = official_repeat_ids[0, -action_dim:]
        official_action = _decode(model, official_ids, args.unnorm_key)
        candidate_action = _decode(model, candidate_ids, args.unnorm_key)
        action_delta = candidate_action - official_action
        rows.append(
            {
                "name": name,
                "shape": list(candidate.shape),
                "global_mae": float(delta.mean().item()),
                "global_linf": float(delta.max().item()),
                "model_input_bfloat16_mae": float(model_input_delta.mean().item()),
                "model_input_bfloat16_linf": float(model_input_delta.max().item()),
                "model_input_bfloat16_unequal_count": int(
                    torch.count_nonzero(
                        candidate_model_input != official_model_input
                    ).item()
                ),
                "branches": branch_rows,
                "official_action_tokens": official_tokens.tolist(),
                "official_repeat_action_tokens": official_repeat_tokens.tolist(),
                "candidate_action_tokens": candidate_tokens.tolist(),
                "token_hamming": int((official_tokens != candidate_tokens).sum().item()),
                "official_repeat_token_hamming": int(
                    (official_tokens != official_repeat_tokens).sum().item()
                ),
                "decoded_action_l2": float(np.linalg.norm(action_delta)),
                "decoded_action_linf": float(np.abs(action_delta).max()),
            }
        )

    evidence = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_training_before_explicit_eval": model_training_before_eval,
        "model_training_during_audit": bool(model.training),
        "resolved_branch_order": list(differentiable.branch_model_ids),
        "resize_mode": [branch.interpolation for branch in differentiable.branches],
        "antialias": [branch.antialias for branch in differentiable.branches],
        "means": [list(branch.mean) for branch in differentiable.branches],
        "stds": [list(branch.std) for branch in differentiable.branches],
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    real_row = rows[-1]
    if real_row["official_repeat_token_hamming"] != 0:
        return 3
    return 0 if real_row["token_hamming"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
