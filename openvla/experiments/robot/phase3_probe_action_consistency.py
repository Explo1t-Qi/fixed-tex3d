"""Dependency-light metrics for probe-vs-policy action consistency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


ACTION_NAMES = ("x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper")
EPS = 1e-12


class ProbeActionConsistencyError(RuntimeError):
    """Raised when consistency inputs or metrics violate the contract."""


@dataclass(frozen=True)
class ConsistencyInputs:
    z_clean: np.ndarray
    z_adv: np.ndarray
    action_clean: np.ndarray
    action_adv: np.ndarray
    action_mean: np.ndarray
    action_safe_std: np.ndarray


def _array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ProbeActionConsistencyError(f"{name} must be finite shape {shape}")
    return result


def _cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= EPS:
        return None
    value = float(np.dot(first, second) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = float(
        np.linalg.norm(centered_first) * np.linalg.norm(centered_second)
    )
    if denominator <= EPS:
        return None
    value = float(np.dot(centered_first, centered_second) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def consistency_metrics(inputs: ConsistencyInputs) -> dict[str, Any]:
    """Compare frozen-probe coordinates to normalized deployed actions."""

    z_clean = np.asarray(inputs.z_clean, dtype=np.float64)
    if z_clean.ndim != 2 or z_clean.shape[1] != 7 or not np.all(np.isfinite(z_clean)):
        raise ProbeActionConsistencyError("z_clean must be finite [N,7]")
    count = z_clean.shape[0]
    if count < 2:
        raise ProbeActionConsistencyError("at least two paired frames are required")
    z_adv = _array(inputs.z_adv, name="z_adv", shape=(count, 7))
    action_clean = _array(inputs.action_clean, name="action_clean", shape=(count, 7))
    action_adv = _array(inputs.action_adv, name="action_adv", shape=(count, 7))
    mean = _array(inputs.action_mean, name="action_mean", shape=(7,))
    safe_std = _array(inputs.action_safe_std, name="action_safe_std", shape=(7,))
    if np.any(safe_std <= 0):
        raise ProbeActionConsistencyError("action_safe_std must be positive")

    normalized_clean = (action_clean - mean) / safe_std
    normalized_adv = (action_adv - mean) / safe_std
    error_clean = z_clean - normalized_clean
    error_adv = z_adv - normalized_adv
    delta_z = z_adv - z_clean
    delta_action = normalized_adv - normalized_clean
    raw_delta_action = action_adv - action_clean
    clean_sample_mse = np.mean(np.square(error_clean), axis=1)
    adv_sample_mse = np.mean(np.square(error_adv), axis=1)
    delta_z_l2 = np.linalg.norm(delta_z, axis=1)
    delta_action_l2 = np.linalg.norm(delta_action, axis=1)
    raw_delta_action_l2 = np.linalg.norm(raw_delta_action, axis=1)
    per_sample_cosines = [_cosine(delta_z[i], delta_action[i]) for i in range(count)]
    defined_cosines = np.asarray(
        [value for value in per_sample_cosines if value is not None], dtype=np.float64
    )

    per_dimension: dict[str, Any] = {}
    for index, name in enumerate(ACTION_NAMES):
        per_dimension[name] = {
            "delta_z_mean": float(delta_z[:, index].mean()),
            "delta_z_mean_abs": float(np.abs(delta_z[:, index]).mean()),
            "delta_action_mean": float(delta_action[:, index].mean()),
            "delta_action_mean_abs": float(np.abs(delta_action[:, index]).mean()),
            "raw_delta_action_mean": float(raw_delta_action[:, index].mean()),
            "raw_delta_action_mean_abs": float(
                np.abs(raw_delta_action[:, index]).mean()
            ),
            "delta_pearson": _pearson(delta_z[:, index], delta_action[:, index]),
            "clean_probe_mse": float(np.mean(np.square(error_clean[:, index]))),
            "adv_probe_mse": float(np.mean(np.square(error_adv[:, index]))),
        }

    clean_mse = float(np.mean(np.square(error_clean)))
    adv_mse = float(np.mean(np.square(error_adv)))
    metrics = {
        "sample_count": count,
        "probe_prediction_error": {
            "clean_mse": clean_mse,
            "adv_mse": adv_mse,
            "adv_to_clean_mse_ratio": adv_mse / (clean_mse + EPS),
            "clean_sample_mse": _summary(clean_sample_mse),
            "adv_sample_mse": _summary(adv_sample_mse),
        },
        "displacement": {
            "delta_z_mse": float(np.mean(np.square(delta_z))),
            "normalized_action_delta_mse": float(np.mean(np.square(delta_action))),
            "delta_z_l2": _summary(delta_z_l2),
            "delta_action_l2": _summary(delta_action_l2),
            "raw_action_delta_l2": _summary(raw_delta_action_l2),
            "flattened_cosine": _cosine(delta_z.reshape(-1), delta_action.reshape(-1)),
            "flattened_pearson": _pearson(
                delta_z.reshape(-1), delta_action.reshape(-1)
            ),
            "per_sample_cosine": per_sample_cosines,
            "per_sample_cosine_defined_count": int(defined_cosines.size),
            "per_sample_cosine_mean": (
                float(defined_cosines.mean()) if defined_cosines.size else None
            ),
        },
        "per_dimension": per_dimension,
        "arrays": {
            "z_clean": z_clean.tolist(),
            "z_adv": z_adv.tolist(),
            "normalized_action_clean": normalized_clean.tolist(),
            "normalized_action_adv": normalized_adv.tolist(),
            "delta_z": delta_z.tolist(),
            "delta_action": delta_action.tolist(),
            "raw_delta_action": raw_delta_action.tolist(),
            "raw_action_clean": action_clean.tolist(),
            "raw_action_adv": action_adv.tolist(),
        },
    }
    scalars = (
        clean_mse,
        adv_mse,
        *clean_sample_mse,
        *adv_sample_mse,
        *delta_z_l2,
        *delta_action_l2,
        *raw_delta_action_l2,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ProbeActionConsistencyError("consistency metrics are non-finite")
    return metrics
