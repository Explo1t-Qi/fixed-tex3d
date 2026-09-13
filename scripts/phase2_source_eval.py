#!/usr/bin/env python3
"""Paired clean/adversarial LIBERO evaluation for one Phase 2.4 source model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
OPENVLA_ROOT = PROJECT_ROOT / "openvla"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("openvla", "pi05"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--openvla-checkpoint", type=Path)
    parser.add_argument("--pi05-checkpoint", type=Path)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--state-start", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pi05-replan-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _validate(args: argparse.Namespace) -> dict[str, Path]:
    if args.task_id != 0:
        raise ValueError("Phase 2.4 source evaluation is frozen to task 0")
    if args.num_trials < 1 or args.state_start < 0:
        raise ValueError("invalid trial range")
    if args.num_steps_wait != 10:
        raise ValueError("Phase 2.4 num_steps_wait is frozen to 10")
    if args.pi05_replan_steps != 1:
        raise ValueError("PI0Pytorch source evaluation replan_steps is frozen to 1")
    paths = {
        "texture": args.texture.expanduser().resolve(strict=True),
        "libero": args.libero_root.expanduser().resolve(strict=True),
    }
    if args.model == "openvla":
        if args.openvla_checkpoint is None:
            raise ValueError("--openvla-checkpoint is required for OpenVLA")
        paths["checkpoint"] = args.openvla_checkpoint.expanduser().resolve(strict=True)
    else:
        if args.pi05_checkpoint is None or args.openpi_root is None:
            raise ValueError(
                "--pi05-checkpoint and --openpi-root are required for PI0.5"
            )
        paths["checkpoint"] = args.pi05_checkpoint.expanduser().resolve(strict=True)
        paths["openpi"] = args.openpi_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")
    paths["output"] = output
    return paths


def _run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _validate(args)
    os.environ["LIBERO_ROOT"] = str(paths["libero"])
    if str(paths["libero"]) not in sys.path:
        sys.path.insert(0, str(paths["libero"]))
    for path in reversed((ROBOT_ROOT / "libero", ROBOT_ROOT, OPENVLA_ROOT)):
        sys.path.insert(0, str(path))

    from phase2_shared_optimization import PairedTrial, paired_source_metrics
    from openvla_runtime_assets import (
        resolve_runtime_texture_binding,
        temporary_runtime_texture,
    )
    from attack_openvla import OBJECTS, get_libero_env, get_libero_image
    from libero.libero import benchmark

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("authoritative source evaluation requires CUDA")
    torch.cuda.set_device(device)
    _seed(args.seed)
    object_config = OBJECTS["akita_black_bowl"]
    clean_texture = Path(object_config["texture"]).resolve(strict=True)
    xml_path = Path(object_config["xml"]).resolve(strict=True)
    resolve_runtime_texture_binding(
        xml_path, clean_texture, object_name="akita_black_bowl"
    )
    initial_xml = xml_path.read_bytes()

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    state_ids = list(range(args.state_start, args.state_start + args.num_trials))
    if not state_ids or state_ids[-1] >= len(init_states):
        raise ValueError("requested paired state IDs exceed available LIBERO states")

    if args.model == "openvla":
        from openvla_utils import get_processor, get_vla
        from robot_utils import (
            get_action,
            invert_gripper_action,
            normalize_gripper_action,
        )
        from libero_utils import get_libero_dummy_action, quat2axisangle
        from openvla_policy_view import (
            DEFAULT_DEPLOYMENT_VIEW,
            resize_policy_pre_crop_canvas,
        )

        if device != torch.device("cuda:0"):
            raise ValueError("the existing OpenVLA loader is fixed to cuda:0")
        cfg = SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=paths["checkpoint"],
            load_in_8bit=False,
            load_in_4bit=False,
            unnorm_key=None,
            center_crop=True,
        )
        policy = get_vla(cfg)
        processor = get_processor(cfg)
        max_steps = args.max_steps or 300

        def rollout(state_id: int) -> bool:
            env, description = get_libero_env(task, "openvla", resolution=512)
            try:
                env.reset()
                obs = env.set_init_state(init_states[state_id])
                env.env.sim.forward()
                for step in range(max_steps + args.num_steps_wait):
                    if step < args.num_steps_wait:
                        obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                    else:
                        source = get_libero_image(obs, 512)
                        model_image = resize_policy_pre_crop_canvas(
                            source, specification=DEFAULT_DEPLOYMENT_VIEW
                        )
                        observation = {
                            "full_image": model_image,
                            "state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                        }
                        action = get_action(
                            cfg, policy, observation, description, processor=processor
                        )
                        action = normalize_gripper_action(action, binarize=True)
                        action = invert_gripper_action(action)
                        obs, _, done, _ = env.step(action.tolist())
                    if done or env.env._check_success():
                        return True
                return False
            finally:
                env.close()

    else:
        from phase2_shared_gradient_smoke import _add_source_paths, _load_pi05

        _add_source_paths(paths["openpi"], PROJECT_ROOT)  # shared path is unused here.
        from openpi_client import image_tools
        from libero_utils import get_libero_dummy_action, quat2axisangle

        pi05 = _load_pi05(paths["openpi"], paths["checkpoint"], device)
        policy = pi05.policy
        max_steps = args.max_steps or 400

        def client_image(image: np.ndarray) -> np.ndarray:
            return image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, 224, 224)
            )

        def rollout(state_id: int) -> bool:
            env, description = get_libero_env(task, "openvla", resolution=256)
            action_plan: deque[np.ndarray] = deque()
            try:
                env.reset()
                obs = env.set_init_state(init_states[state_id])
                env.env.sim.forward()
                for step in range(max_steps + args.num_steps_wait):
                    if step < args.num_steps_wait:
                        obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                    else:
                        if not action_plan:
                            base = np.ascontiguousarray(
                                obs["agentview_image"][::-1, ::-1]
                            )
                            wrist = np.ascontiguousarray(
                                obs["robot0_eye_in_hand_image"][::-1, ::-1]
                            )
                            raw = {
                                "observation/image": client_image(base),
                                "observation/wrist_image": client_image(wrist),
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(description),
                            }
                            with torch.no_grad():
                                result = policy.infer(raw)
                            actions = (
                                result["actions"]
                                if isinstance(result, dict)
                                else result
                            )
                            if isinstance(actions, torch.Tensor):
                                actions = actions.detach().cpu().numpy()
                            actions = np.asarray(actions)
                            if actions.ndim == 3:
                                actions = actions[0]
                            if actions.ndim == 1:
                                actions = actions[None, :]
                            if not len(actions):
                                raise RuntimeError(
                                    "PI0Pytorch returned an empty action chunk"
                                )
                            action_plan.extend(actions[: args.pi05_replan_steps])
                        action = np.asarray(action_plan.popleft(), dtype=np.float32)
                        obs, _, done, _ = env.step(action.tolist())
                    if done or env.env._check_success():
                        return True
                return False
            finally:
                env.close()

    output = paths["output"]
    output.mkdir(parents=True)
    result_path = output / "paired_trials.jsonl"
    result_path.write_text("", encoding="utf-8")
    trials: list[PairedTrial] = []
    for state_id in state_ids:
        trial_seed = args.seed + state_id
        _seed(trial_seed)
        clean_success = rollout(state_id)
        _seed(trial_seed)
        context = temporary_runtime_texture(
            xml_path,
            clean_texture,
            paths["texture"],
            object_name="akita_black_bowl",
        )
        with context:
            adversarial_success = rollout(state_id)
        if xml_path.read_bytes() != initial_xml:
            raise RuntimeError("asset XML was not restored after adversarial rollout")
        trial = PairedTrial(state_id, clean_success, adversarial_success)
        trials.append(trial)
        with result_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "state_id": state_id,
                        "clean_success": clean_success,
                        "adversarial_success": adversarial_success,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        print(result_path.read_text().splitlines()[-1], flush=True)

    metrics = paired_source_metrics(trials)
    summary = {
        "status": "Phase 2.4 Source-Model Paired Evaluation — COMPLETE",
        "model": args.model,
        "task_suite": "libero_spatial",
        "task_id": args.task_id,
        "checkpoint": str(paths["checkpoint"]),
        "adversarial_texture": str(paths["texture"]),
        "adversarial_texture_sha256": _sha256(paths["texture"]),
        "tex3d_commit": _git_head(),
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "pi05_replan_steps": args.pi05_replan_steps if args.model == "pi05" else None,
        "paired_seed_rule": "seed + state_id, reset before clean and adversarial",
        "asset_xml_restored": xml_path.read_bytes() == initial_xml,
        "metrics": metrics,
    }
    (output / "evaluation_summary.json").write_text(
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
                    "status": "Phase 2.4 Source-Model Evaluation — BLOCKED",
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
