"""Low-rank diagnostics for action-probe initialization stability."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from phase2_action_probe import ACTION_NAMES, ActionProbeError


STABILITY_SCHEMA_VERSION = "phase2_action_probe_stability_v1"
DEFAULT_SEEDS = (1, 2, 3, 4, 5, 7)
REFERENCE_SEED = 7
PREDICTION_CV_THRESHOLD = 0.05
PRINCIPAL_COSINE_THRESHOLD = 0.90
PROJECTION_DISTANCE_THRESHOLD = 0.25


def validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if len(set(values)) != len(values):
        raise ActionProbeError("stability diagnostic requires distinct seeds")
    if any(type(value) is not int or value < 0 for value in values):
        raise ActionProbeError("stability seeds must be non-negative integers")
    missing = sorted(set(DEFAULT_SEEDS) - set(values))
    if missing:
        raise ActionProbeError(
            f"stability seeds must include 1,2,3,4,5,7; missing {missing}"
        )
    return values


def signed_action_row_cosines(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Return signed cosine for corresponding action rows of two probe weights."""

    left, right = _weight_pair(first, second)
    denominator = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(
        right, dim=1
    )
    if bool(torch.any(denominator == 0)):
        raise ActionProbeError("probe action row has zero norm")
    return torch.sum(left * right, dim=1) / denominator


def principal_angle_cosines(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the seven principal-angle cosines between probe row spaces."""

    left, right = _weight_pair(first, second)
    if (
        int(torch.linalg.matrix_rank(left).item()) != 7
        or int(torch.linalg.matrix_rank(right).item()) != 7
    ):
        raise ActionProbeError("principal angles require rank-7 probe weights")
    q_left = torch.linalg.qr(left.T, mode="reduced").Q
    q_right = torch.linalg.qr(right.T, mode="reduced").Q
    values = torch.linalg.svdvals(q_left.T @ q_right)
    return values.clamp(0.0, 1.0)


def projection_relative_frobenius_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    probe_reg: float,
) -> float:
    """Compare ridge projectors using only 7x7 Gram matrices."""

    if not math.isfinite(probe_reg) or probe_reg <= 0:
        raise ActionProbeError("probe_reg must be finite and positive")
    left, right = _weight_pair(first, second)
    identity = torch.eye(7, dtype=torch.float64)
    gram_left = left @ left.T
    gram_right = right @ right.T
    inverse_left = torch.linalg.solve(gram_left + probe_reg * identity, identity)
    inverse_right = torch.linalg.solve(gram_right + probe_reg * identity, identity)
    cross = left @ right.T

    left_product = inverse_left @ gram_left
    right_product = inverse_right @ gram_right
    left_norm_sq = torch.trace(left_product @ left_product)
    right_norm_sq = torch.trace(right_product @ right_product)
    inner = torch.trace(inverse_left @ cross @ inverse_right @ cross.T)
    distance_sq = torch.clamp(left_norm_sq + right_norm_sq - 2.0 * inner, min=0.0)
    denominator = torch.maximum(torch.sqrt(left_norm_sq), torch.sqrt(right_norm_sq))
    if float(denominator) == 0:
        raise ActionProbeError("ridge projection has zero Frobenius norm")
    return float(torch.sqrt(distance_sq) / denominator)


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ActionProbeError("summary values must be a finite non-empty vector")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def pairwise_stability(
    weights: Mapping[int, torch.Tensor],
    heldout_mse: Mapping[int, float],
    heldout_r2: Mapping[int, Mapping[str, float | None]],
    *,
    probe_reg: float,
) -> dict[str, Any]:
    seeds = validate_seeds(sorted(weights))
    if set(seeds) != set(heldout_mse) or set(seeds) != set(heldout_r2):
        raise ActionProbeError("stability inputs use different seed sets")
    pair_rows: list[dict[str, Any]] = []
    row_values: dict[str, list[float]] = {name: [] for name in ACTION_NAMES}
    principal_means: list[float] = []
    principal_minima: list[float] = []
    projection_distances: list[float] = []
    for index, seed_i in enumerate(seeds):
        for seed_j in seeds[index + 1 :]:
            row = signed_action_row_cosines(weights[seed_i], weights[seed_j])
            principal = principal_angle_cosines(weights[seed_i], weights[seed_j])
            projection = projection_relative_frobenius_distance(
                weights[seed_i], weights[seed_j], probe_reg=probe_reg
            )
            row_dict = {
                name: float(value)
                for name, value in zip(ACTION_NAMES, row, strict=True)
            }
            for name, value in row_dict.items():
                row_values[name].append(value)
            principal_list = [float(value) for value in principal]
            principal_means.append(float(principal.mean()))
            principal_minima.append(float(principal.min()))
            projection_distances.append(projection)
            pair_rows.append(
                {
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "action_row_signed_cosines": row_dict,
                    "principal_angle_cosines": principal_list,
                    "mean_principal_cosine": float(principal.mean()),
                    "minimum_principal_cosine": float(principal.min()),
                    "projection_relative_frobenius_distance": projection,
                }
            )

    mse_summary = summarize([heldout_mse[seed] for seed in seeds])
    prediction_cv = mse_summary["std"] / max(abs(mse_summary["mean"]), 1e-12)
    per_action_r2: dict[str, Any] = {}
    for name in ACTION_NAMES:
        values = [heldout_r2[seed][name] for seed in seeds]
        if any(value is None for value in values):
            per_action_r2[name] = None
        else:
            per_action_r2[name] = {
                key: value
                for key, value in summarize([float(value) for value in values]).items()
                if key in ("mean", "std")
            }

    reference: dict[str, Any] = {}
    for seed in seeds:
        if seed == REFERENCE_SEED:
            continue
        row = next(
            value
            for value in pair_rows
            if {value["seed_i"], value["seed_j"]} == {seed, REFERENCE_SEED}
        )
        reference[str(seed)] = {
            "heldout_mse_difference_vs_seed7": float(
                heldout_mse[seed] - heldout_mse[REFERENCE_SEED]
            ),
            "action_row_signed_cosines": row["action_row_signed_cosines"],
            "principal_angle_cosines": row["principal_angle_cosines"],
            "projection_relative_frobenius_distance": row[
                "projection_relative_frobenius_distance"
            ],
        }

    return {
        "heldout_mse": mse_summary,
        "heldout_mse_coefficient_of_variation": float(prediction_cv),
        "per_action_r2": per_action_r2,
        "action_row_signed_cosine": {
            name: summarize(values) for name, values in row_values.items()
        },
        "principal_angle_cosine": {
            "pairwise_mean": summarize(principal_means),
            "pairwise_minimum": summarize(principal_minima),
        },
        "projection_relative_frobenius_distance": summarize(projection_distances),
        "pairs": pair_rows,
        "reference_to_seed7": reference,
    }


def classify_stability(nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    stable = True
    for name, value in nodes.items():
        prediction_cv = float(value["heldout_mse_coefficient_of_variation"])
        minimum_principal = float(
            value["principal_angle_cosine"]["pairwise_minimum"]["min"]
        )
        maximum_projection = float(
            value["projection_relative_frobenius_distance"]["max"]
        )
        checks = {
            "prediction_cv_at_most_0.05": prediction_cv <= PREDICTION_CV_THRESHOLD,
            "minimum_principal_cosine_at_least_0.90": minimum_principal
            >= PRINCIPAL_COSINE_THRESHOLD,
            "maximum_projection_distance_at_most_0.25": maximum_projection
            <= PROJECTION_DISTANCE_THRESHOLD,
        }
        decisions[name] = {
            "stable": all(checks.values()),
            "checks": checks,
        }
        stable &= all(checks.values())
    return {
        "status": "STABLE" if stable else "NEEDS_REVIEW",
        "thresholds": {
            "maximum_heldout_mse_coefficient_of_variation": PREDICTION_CV_THRESHOLD,
            "minimum_principal_cosine": PRINCIPAL_COSINE_THRESHOLD,
            "maximum_projection_relative_frobenius_distance": PROJECTION_DISTANCE_THRESHOLD,
        },
        "nodes": decisions,
    }


def tree_hashes(root: Path) -> dict[str, str]:
    directory = root.expanduser().resolve(strict=True)
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(directory))] = digest.hexdigest()
    return result


def assert_tree_unchanged(root: Path, before: Mapping[str, str]) -> None:
    if tree_hashes(root) != dict(before):
        raise ActionProbeError(
            "Phase 2A source artifact tree changed during diagnostic"
        )


def _weight_pair(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    left = first.detach().to(dtype=torch.float64, device="cpu")
    right = second.detach().to(dtype=torch.float64, device="cpu")
    if (
        left.ndim != 2
        or left.shape[0] != 7
        or left.shape != right.shape
        or not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
    ):
        raise ActionProbeError("probe weights must be finite matching [7,D] tensors")
    return left, right
