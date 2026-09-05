"""Serialization and completion gates for the single frozen Step 1B trajectory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


CHECKPOINT_SCHEDULE = (10, 100, 500, 1000, 2000, 5000)
TOTAL_ITERATIONS = 5000
# Independent of the historical Step 1 gate. Paths and log labels may vary.
FROZEN_CONFIG = {
    "step1b_mature_trajectory": True, "step1_formal": False,
    "attack_objective": "o2_displacement", "attack_iters": TOTAL_ITERATIONS,
    "task_suite_name": "libero_spatial", "task_id": 0,
    "object_name": "akita_black_bowl", "model_family": "openvla",
    "enable_attack": True, "attack_lr": 0.05, "seed": 7,
    "num_train_init_states": 10, "train_frames_per_state": 1,
    "num_frames_to_attack": 20, "photometric_calib_frames": 5,
    "frame_collect_with_policy": False, "collect_grasp_frames": False,
    "num_trials_per_task": 0, "live_test_enabled": False,
    "save_attack_artifacts": True, "load_texture_path": None,
    "center_crop": True, "load_in_8bit": False, "load_in_4bit": False,
    "override_mesh_path": None, "override_texture_path": None,
    "override_xml_path": None, "use_wandb": False,
    "unnorm_key": "libero_spatial_no_noops", "num_steps_wait": 10,
    "alpha_action": 1.0, "alpha_feature": 10.0,
    "grasp_pre_frames": 40, "grasp_post_frames": 0,
    "grasp_max_steps": 400, "grasp_qpos_threshold": 0.02,
    "live_test_every_n_iters": 20, "live_test_resolution": 256,
    "live_test_max_steps": 300, "replay_resolution": 512,
}


def validate_step1b_config(config: dict) -> None:
    if not config.get("step1b_mature_trajectory", False):
        return
    differences = {k: (config.get(k), v) for k, v in FROZEN_CONFIG.items()
                   if config.get(k) != v}
    if differences or not config.get("step1_output_dir"):
        raise ValueError(f"Step 1B frozen configuration mismatch: {differences}")


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def checkpoint_iteration(zero_based_iteration: int) -> int | None:
    completed = zero_based_iteration + 1
    return completed if completed in CHECKPOINT_SCHEDULE else None


def initialize_manifest(training_dir: Path, config: dict) -> None:
    validate_step1b_config(config)
    directory = training_dir / "checkpoints"
    directory.mkdir()  # Fresh directory; never resume or overwrite a trajectory.
    write_json(directory / "checkpoint_manifest.json", {
        "schema_version": "step1b_checkpoints_v1",
        "checkpoint_schedule": list(CHECKPOINT_SCHEDULE),
        "completed_checkpoints": [], "checkpoints": [],
        "training_complete": False, "total_iterations": TOTAL_ITERATIONS,
        "source_git_commit": config["source_git_commit"],
        "attack_objective": config["attack_objective"],
        "seed": config["seed"], "attack_lr": config["attack_lr"],
        "train_state_ids": list(range(10)), "heldout_state_ids": list(range(10, 20)),
    })


@torch.no_grad()
def save_checkpoint(renderer, training_dir: Path, zero_based_iteration: int,
                    step_metrics: dict) -> None:
    iteration = checkpoint_iteration(zero_based_iteration)
    if iteration is None:
        return
    directory = training_dir / "checkpoints"
    manifest_path = directory / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    expected_previous = list(CHECKPOINT_SCHEDULE[:CHECKPOINT_SCHEDULE.index(iteration)])
    if manifest["training_complete"] or manifest["completed_checkpoints"] != expected_previous:
        raise ValueError("checkpoint schedule/order mismatch")
    target = directory / f"iter_{iteration:06d}"
    target.mkdir()
    parameter = renderer.get_texture_param().detach().cpu()
    baked = renderer.get_baked_adv_texture()
    if not bool(torch.isfinite(parameter).all()) or not bool(torch.isfinite(baked).all()):
        raise ValueError("non-finite checkpoint parameter/texture")
    perturbation = float((torch.tanh(parameter) * renderer.epsilon).abs().max())
    if perturbation > float(renderer.epsilon) + 1e-6:
        raise ValueError("checkpoint texture budget exceeded")
    torch.save(parameter, target / "parameter.pt")
    # Exactly the existing final-artifact PNG conversion; no extra forward.
    Image.fromarray((baked.squeeze(0).cpu().numpy() * 255).astype(np.uint8)).save(
        target / "attack_texture.png")
    metadata = {
        "iteration": iteration, "parameter_state": "after_parameter_update",
        "attack_objective": manifest["attack_objective"],
        "parameter_sha256": sha256_file(target / "parameter.pt"),
        "texture_sha256": sha256_file(target / "attack_texture.png"),
        "parameter_linf": float(parameter.abs().max()),
        "maximum_texture_perturbation": perturbation,
        "renderer_epsilon": float(renderer.epsilon),
        "seed": manifest["seed"], "attack_lr": manifest["attack_lr"],
        "source_git_commit": manifest["source_git_commit"],
        "pre_update_total_loss": step_metrics["total_loss"],
        "pre_update_o2_displacement": step_metrics["o2_displacement"],
    }
    write_json(target / "metadata.json", metadata)
    manifest["checkpoints"].append({
        **metadata,
        "parameter_path": f"{target.name}/parameter.pt",
        "texture_path": f"{target.name}/attack_texture.png",
    })
    manifest["completed_checkpoints"].append(iteration)
    write_json(manifest_path, manifest)


def audit_checkpoints(run_root: Path, *, require_complete: bool = True) -> dict:
    """Read-only gate: all updates, hashes, final equivalence, and restoration."""
    training = run_root / "training"
    config = read_json(run_root / "config.json")
    validate_step1b_config(config)
    if config.get("step1b_mature_trajectory") is not True:
        raise ValueError("not a Step 1B run")
    manifest = read_json(training / "checkpoints/checkpoint_manifest.json")
    if (manifest["schema_version"] != "step1b_checkpoints_v1"
            or manifest["checkpoint_schedule"] != list(CHECKPOINT_SCHEDULE)
            or manifest["completed_checkpoints"] != list(CHECKPOINT_SCHEDULE)
            or manifest["total_iterations"] != TOTAL_ITERATIONS
            or (require_complete and any(manifest.get(k) is not True for k in (
                "training_complete", "final_parameter_equal", "final_texture_sha256_equal"
            )))):
        raise ValueError("5000-update trajectory is not complete")
    for key in ("source_git_commit", "attack_objective", "seed", "attack_lr",
                "train_state_ids", "heldout_state_ids"):
        if manifest[key] != config[key]:
            raise ValueError(f"manifest provenance mismatch: {key}")
    if (config["train_state_ids"] != list(range(10))
            or config["heldout_state_ids"] != list(range(10, 20))
            or config["renderer_epsilon"] != 128 / 255
            or config["renderer_position_offset"] != [0.0, 0.0, 0.0]):
        raise ValueError("frozen split/renderer mismatch")
    if [r["iteration"] for r in manifest["checkpoints"]] != list(CHECKPOINT_SCHEDULE):
        raise ValueError("checkpoint records incomplete")
    for record in manifest["checkpoints"]:
        target = training / "checkpoints" / f"iter_{record['iteration']:06d}"
        metadata = read_json(target / "metadata.json")
        if record != {**metadata, "parameter_path": f"{target.name}/parameter.pt",
                      "texture_path": f"{target.name}/attack_texture.png"}:
            raise ValueError("checkpoint metadata mismatch")
        if any(record[k] != manifest[k] for k in (
            "source_git_commit", "seed", "attack_lr", "attack_objective"
        )) or record["parameter_state"] != "after_parameter_update":
            raise ValueError("checkpoint trajectory provenance mismatch")
        for filename, key in (("parameter.pt", "parameter_sha256"),
                              ("attack_texture.png", "texture_sha256")):
            if sha256_file(target / filename) != record[key]:
                raise ValueError(f"checkpoint SHA256 mismatch: {target / filename}")
        parameter = torch.load(target / "parameter.pt", map_location="cpu", weights_only=True)
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("non-finite saved parameter")
        if (record["parameter_linf"] != float(parameter.abs().max())
                or record["renderer_epsilon"] != config["renderer_epsilon"]
                or record["maximum_texture_perturbation"] != float(
                    (torch.tanh(parameter) * config["renderer_epsilon"]).abs().max())
                or record["maximum_texture_perturbation"] > config["renderer_epsilon"] + 1e-6):
            raise ValueError("checkpoint parameter/budget metadata mismatch")
    final = training / "checkpoints/iter_005000"
    a = torch.load(final / "parameter.pt", map_location="cpu", weights_only=True)
    b = torch.load(training / "parameter.pt", map_location="cpu", weights_only=True)
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
        raise ValueError("5000 checkpoint and final parameter differ")
    final_hash = sha256_file(training / "final_attack_texture.png")
    summary = read_json(training / "training_summary.json")
    if not (final_hash == sha256_file(final / "attack_texture.png")
            == (training / "texture_sha256.txt").read_text().strip()
            == summary["texture_sha256"]):
        raise ValueError("5000 checkpoint and final texture SHA256 differ")
    if (summary["num_iterations"] != TOTAL_ITERATIONS
            or summary["num_training_frames"] != 10
            or summary["pi05_loaded_during_training"] is not False
            or summary["texture_budget_respected"] is not True):
        raise ValueError("training summary invalid")
    metrics = [json.loads(line) for line in
               (training / "Ep0_step_metrics.jsonl").read_text().splitlines()]
    if [m["iteration"] for m in metrics] != list(range(TOTAL_ITERATIONS)):
        raise ValueError("not exactly 5000 update records")
    for m in metrics:
        if (not all(np.isfinite(v) for v in m.values() if isinstance(v, (int, float)))
                or m["texture_gradient_norm"] <= 0 or m["image_gradient_norm_max"] <= 0
                or m["parameter_change_linf"] <= 0
                or m["maximum_texture_perturbation"] > config["renderer_epsilon"] + 1e-6
                or m["attack_objective"] != "o2_displacement"
                or not np.isclose(m["total_loss"], -m["o2_displacement"], rtol=1e-6, atol=1e-12)
                or m["action_loss"] != 0 or m["feature_loss"] != 0):
            raise ValueError("training update/gradient contract failed")
    for record in manifest["checkpoints"]:
        m = metrics[record["iteration"] - 1]
        if (record["pre_update_total_loss"] != m["total_loss"]
                or record["pre_update_o2_displacement"] != m["o2_displacement"]):
            raise ValueError("checkpoint pre-update metric mismatch")
    losses = np.load(training / "loss_history.npy", allow_pickle=False)
    if not np.array_equal(losses, [m["total_loss"] for m in metrics]):
        raise ValueError("loss history mismatch")
    restoration = read_json(run_root / "asset_restoration.json")
    if any(restoration.get(k) is not True for k in (
        "pass", "xml_restored", "clean_texture_restored", "xml_backup_removed", "texture_backup_removed"
    )):
        raise ValueError("asset restoration failed")
    return manifest


def finalize_trajectory(run_root: Path) -> None:
    manifest = audit_checkpoints(run_root, require_complete=False)
    manifest["training_complete"] = True
    manifest["final_parameter_equal"] = True
    manifest["final_texture_sha256_equal"] = True
    write_json(run_root / "training/checkpoints/checkpoint_manifest.json", manifest)
