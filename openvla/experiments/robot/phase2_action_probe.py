"""Dependency-light fitting and artifact logic for action linear probes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ACTION_NAMES = ("x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper")
MODELS = ("openvla", "pi05")
NODES = {"openvla": ("o2", "deep"), "pi05": ("p2", "deep")}
NODE_WIDTHS = {
    ("openvla", "o2"): 4096,
    ("openvla", "deep"): 4096,
    ("pi05", "p2"): 2048,
    ("pi05", "deep"): 2048,
}
SPLIT_RULE_ID = "pilot-v0.2-c5-split-v1"
SCHEMA_VERSION = "phase2_action_representation_v1"
PROBE_SCHEMA_VERSION = "phase2_action_probe_v1"


class ActionProbeError(RuntimeError):
    """Raised when the Phase 2 action-probe contract is invalid."""


@dataclass(frozen=True)
class SampleIdentity:
    sample_id: str
    task_id: int
    initial_state_id: int
    target_progress: float


@dataclass(frozen=True)
class ProbeConfig:
    seed: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 2000
    action_std_epsilon: float = 1e-6
    probe_reg: float = 1e-4

    def validate(self) -> None:
        if self.steps <= 0 or self.seed < 0:
            raise ActionProbeError("probe steps must be positive and seed non-negative")
        for name in ("learning_rate", "action_std_epsilon", "probe_reg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ActionProbeError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ActionProbeError("weight_decay must be finite and non-negative")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def split_digest(task_id: int, initial_state_id: int) -> str:
    canonical = f"{SPLIT_RULE_ID}|task_id={task_id}|initial_state_id={initial_state_id}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_group_split(
    samples: Sequence[SampleIdentity],
) -> dict[str, Any]:
    groups = sorted({(sample.task_id, sample.initial_state_id) for sample in samples})
    tasks = sorted({task for task, _ in groups})
    heldout: set[tuple[int, int]] = set()
    for task in tasks:
        candidates = [(task_id, state) for task_id, state in groups if task_id == task]
        heldout.add(min(candidates, key=lambda group: (split_digest(*group), group[1])))
    train = [group for group in groups if group not in heldout]
    held = [group for group in groups if group in heldout]
    train_ids = [
        s.sample_id for s in samples if (s.task_id, s.initial_state_id) in train
    ]
    held_ids = [
        s.sample_id for s in samples if (s.task_id, s.initial_state_id) in heldout
    ]
    if set(train_ids) & set(held_ids):
        raise ActionProbeError("TRAIN and HELD-OUT sample identities overlap")
    if len(tasks) == 10 and (
        len(train) != 40
        or len(held) != 10
        or len(train_ids) != 160
        or len(held_ids) != 40
    ):
        raise ActionProbeError(
            "Pilot v0.2 split must be 40/10 groups and 160/40 observations"
        )
    return {
        "rule_id": SPLIT_RULE_ID,
        "train_groups": [_group_json(group) for group in train],
        "heldout_groups": [_group_json(group) for group in held],
        "train_sample_ids": train_ids,
        "heldout_sample_ids": held_ids,
    }


def _group_json(group: tuple[int, int]) -> dict[str, Any]:
    return {
        "task_id": group[0],
        "initial_state_id": group[1],
        "digest": split_digest(*group),
    }


def action_distribution_audit(
    actions: np.ndarray,
    samples: Sequence[SampleIdentity],
    *,
    near_zero_threshold: float = 1e-6,
) -> dict[str, Any]:
    actions = _actions(actions)
    if len(actions) != len(samples):
        raise ActionProbeError("action/sample counts differ")

    def summarize(indices: np.ndarray) -> dict[str, Any]:
        values = actions[indices]
        result: dict[str, Any] = {"count": int(len(values)), "dimensions": {}}
        for index, name in enumerate(ACTION_NAMES):
            column = values[:, index]
            std = float(column.std())
            percentiles = np.percentile(column, [5, 25, 50, 75, 95])
            result["dimensions"][name] = {
                "mean": float(column.mean()),
                "std": std,
                "min": float(column.min()),
                "max": float(column.max()),
                "p05": float(percentiles[0]),
                "p25": float(percentiles[1]),
                "median": float(percentiles[2]),
                "p75": float(percentiles[3]),
                "p95": float(percentiles[4]),
                "near_zero_fraction": float(
                    np.mean(np.abs(column) <= near_zero_threshold)
                ),
                "near_constant": bool(std <= near_zero_threshold),
            }
        rounded, counts = np.unique(np.round(values[:, 6], 6), return_counts=True)
        result["gripper"] = {
            "rounding_decimals": 6,
            "effective_states": int(len(rounded)),
            "state_frequency": {
                str(float(k)): int(v) for k, v in zip(rounded, counts, strict=True)
            },
            "state_fraction": {
                str(float(k)): float(v / len(values))
                for k, v in zip(rounded, counts, strict=True)
            },
        }
        return result

    all_indices = np.arange(len(samples))
    by_task = {
        str(task): summarize(
            np.asarray([i for i, s in enumerate(samples) if s.task_id == task])
        )
        for task in sorted({s.task_id for s in samples})
    }
    targets = (0.10, 0.40, 0.70, 0.90)
    by_progress = {
        f"{target:.2f}": summarize(
            np.asarray(
                [
                    i
                    for i, s in enumerate(samples)
                    if abs(s.target_progress - target) < 1e-8
                ]
            )
        )
        for target in targets
    }
    return {
        "near_zero_threshold": near_zero_threshold,
        "overall": summarize(all_indices),
        "by_task_id": by_task,
        "by_target_progress": by_progress,
    }


def _actions(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 7 or not np.all(np.isfinite(result)):
        raise ActionProbeError("actions must be finite [N,7]")
    return result


def training_action_statistics(
    actions: np.ndarray, *, epsilon: float
) -> dict[str, np.ndarray]:
    values = _actions(actions)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    near_constant = std <= epsilon
    safe_std = np.maximum(std, np.float32(epsilon))
    return {
        "mean": mean,
        "std": std,
        "safe_std": safe_std,
        "near_constant": near_constant,
    }


def mean_pool_features(features: np.ndarray, *, width: int) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (256, width):
        raise ActionProbeError(f"features must have shape [N,256,{width}]")
    if not np.all(np.isfinite(value)):
        raise ActionProbeError("features contain non-finite values")
    return value.mean(axis=1, dtype=np.float32)


def fit_linear_probe(
    train_features: np.ndarray,
    train_actions: np.ndarray,
    *,
    config: ProbeConfig,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Linear, dict[str, np.ndarray], list[float]]:
    config.validate()
    target_device = torch.device(device)
    x = torch.from_numpy(np.asarray(train_features, dtype=np.float32)).to(target_device)
    if x.ndim != 2 or not bool(torch.isfinite(x).all()):
        raise ActionProbeError("pooled features must be finite [N,D]")
    stats = training_action_statistics(train_actions, epsilon=config.action_std_epsilon)
    y = torch.from_numpy(
        (_actions(train_actions) - stats["mean"]) / stats["safe_std"]
    ).to(target_device)
    torch.manual_seed(config.seed)
    probe = torch.nn.Linear(x.shape[1], 7, bias=False, device=target_device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[float] = []
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(probe(x), y)
        if not bool(torch.isfinite(loss)):
            raise ActionProbeError("probe optimization produced non-finite loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return probe.eval(), stats, history


def prediction_metrics(
    predicted_normalized: np.ndarray,
    raw_actions: np.ndarray,
    stats: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    target_raw = _actions(raw_actions)
    pred_norm = np.asarray(predicted_normalized, dtype=np.float32)
    if pred_norm.shape != target_raw.shape or not np.all(np.isfinite(pred_norm)):
        raise ActionProbeError("predictions must be finite [N,7]")
    target_norm = (target_raw - stats["mean"]) / stats["safe_std"]
    pred_raw = pred_norm * stats["safe_std"] + stats["mean"]
    return {
        "normalized": _metric_block(pred_norm, target_norm),
        "raw_deployed_action": _metric_block(pred_raw, target_raw),
    }


def _metric_block(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    residual = prediction - target
    per_dimension: dict[str, Any] = {}
    for index, name in enumerate(ACTION_NAMES):
        y = target[:, index].astype(np.float64)
        p = prediction[:, index].astype(np.float64)
        ss_res = float(np.square(p - y).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        y_std, p_std = float(y.std()), float(p.std())
        pearson = None if y_std == 0 or p_std == 0 else float(np.corrcoef(y, p)[0, 1])
        per_dimension[name] = {
            "mse": float(np.mean(np.square(p - y))),
            "mae": float(np.mean(np.abs(p - y))),
            "pearson": pearson,
            "r2": None if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
        }
    return {
        "mse": float(np.mean(np.square(residual, dtype=np.float64))),
        "mae": float(np.mean(np.abs(residual), dtype=np.float64)),
        "per_dimension": per_dimension,
    }


def materialize_action_projection(
    weight: torch.Tensor, *, probe_reg: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    w = weight.detach().to(dtype=torch.float32, device="cpu")
    if w.ndim != 2 or w.shape[0] != 7 or not bool(torch.isfinite(w).all()):
        raise ActionProbeError("probe W must be finite [7,D]")
    gram = w @ w.T
    solve = torch.linalg.solve(gram + probe_reg * torch.eye(7, dtype=torch.float32), w)
    projection = w.T @ solve
    if not bool(torch.isfinite(projection).all()):
        raise ActionProbeError("P_action is non-finite")
    symmetry = torch.linalg.vector_norm(projection - projection.T).item()
    # Exploit rank(W) <= 7; a dense D x D square would be cubic and wasteful.
    projection_squared = w.T @ (solve @ w.T) @ solve
    idempotence = torch.linalg.vector_norm(projection_squared - projection).item()
    singular = torch.linalg.svdvals(w)
    if symmetry > 1e-3:
        raise ActionProbeError(
            f"P_action symmetry residual exceeds tolerance: {symmetry}"
        )
    return projection, {
        "probe_reg": probe_reg,
        "rank_W": int(torch.linalg.matrix_rank(w).item()),
        "singular_values_W": [float(v) for v in singular],
        "symmetry_residual_fro": float(symmetry),
        "idempotence_residual_fro": float(idempotence),
    }


def atomic_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def config_dict(config: ProbeConfig) -> dict[str, Any]:
    return asdict(config)
