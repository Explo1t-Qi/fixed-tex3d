from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import phase3_probe_action_consistency as entrypoint


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

from phase3_probe_action_consistency import (  # noqa: E402
    ConsistencyInputs,
    ProbeActionConsistencyError,
    consistency_metrics,
)


def _inputs(
    *,
    z_clean: np.ndarray | None = None,
    z_adv: np.ndarray | None = None,
    action_clean: np.ndarray | None = None,
    action_adv: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    safe_std: np.ndarray | None = None,
) -> ConsistencyInputs:
    zeros = np.zeros((2, 7), dtype=np.float64)
    return ConsistencyInputs(
        z_clean=zeros.copy() if z_clean is None else z_clean,
        z_adv=zeros.copy() if z_adv is None else z_adv,
        action_clean=zeros.copy() if action_clean is None else action_clean,
        action_adv=zeros.copy() if action_adv is None else action_adv,
        action_mean=np.zeros(7) if mean is None else mean,
        action_safe_std=np.ones(7) if safe_std is None else safe_std,
    )


def test_perfect_probe_predictions_and_aligned_displacements() -> None:
    clean = np.stack((np.arange(7), np.arange(7) + 1)).astype(np.float64)
    delta = np.stack((np.ones(7), np.full(7, 2.0)))
    result = consistency_metrics(
        _inputs(
            z_clean=clean,
            z_adv=clean + delta,
            action_clean=clean,
            action_adv=clean + delta,
        )
    )

    error = result["probe_prediction_error"]
    displacement = result["displacement"]
    assert error["clean_mse"] == 0
    assert error["adv_mse"] == 0
    assert displacement["flattened_cosine"] == pytest.approx(1.0)
    assert displacement["flattened_pearson"] == pytest.approx(1.0)
    assert displacement["delta_z_mse"] == pytest.approx(2.5)
    assert displacement["normalized_action_delta_mse"] == pytest.approx(2.5)
    assert displacement["raw_action_delta_l2"]["mean"] > 0
    assert displacement["continuous_6d"]["flattened_cosine"] == pytest.approx(1.0)
    assert displacement["gripper"]["flip_count"] == 0


def test_train_statistics_normalize_actual_actions() -> None:
    mean = np.arange(7, dtype=np.float64)
    safe_std = np.arange(1, 8, dtype=np.float64)
    normalized_clean = np.full((2, 7), 2.0)
    normalized_adv = np.full((2, 7), 3.0)
    result = consistency_metrics(
        _inputs(
            z_clean=normalized_clean,
            z_adv=normalized_adv,
            action_clean=np.tile(mean + 2 * safe_std, (2, 1)),
            action_adv=np.tile(mean + 3 * safe_std, (2, 1)),
            mean=mean,
            safe_std=safe_std,
        )
    )

    assert result["probe_prediction_error"]["clean_mse"] == pytest.approx(0)
    assert result["probe_prediction_error"]["adv_mse"] == pytest.approx(0)
    assert result["arrays"]["normalized_action_adv"] == normalized_adv.tolist()


def test_probe_extrapolation_is_exposed_by_adv_error() -> None:
    huge_probe_delta = np.full((2, 7), 23.5)
    result = consistency_metrics(_inputs(z_adv=huge_probe_delta))

    assert result["probe_prediction_error"]["clean_mse"] == 0
    assert result["probe_prediction_error"]["adv_mse"] == pytest.approx(23.5**2)
    assert result["displacement"]["delta_z_l2"]["mean"] > 60
    assert result["displacement"]["raw_action_delta_l2"]["mean"] == 0
    assert result["displacement"]["flattened_cosine"] is None


def test_zero_displacements_have_explicit_undefined_alignment() -> None:
    result = consistency_metrics(_inputs())

    assert result["displacement"]["flattened_cosine"] is None
    assert result["displacement"]["flattened_pearson"] is None
    assert result["displacement"]["per_sample_cosine"] == [None, None]
    assert result["displacement"]["per_sample_cosine_defined_count"] == 0


def test_pi05_noise_pair_is_exact_and_independently_owned() -> None:
    noise = np.arange(320, dtype=np.float32).reshape(10, 32)
    clean, adversarial, digest = entrypoint._paired_pi05_noise(noise)

    assert np.array_equal(clean, adversarial)
    assert clean is not adversarial
    assert len(digest) == 64
    clean[0, 0] = -1
    assert adversarial[0, 0] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"z_adv": np.full((2, 7), np.nan)},
        {"action_adv": np.full((2, 7), np.inf)},
        {"safe_std": np.zeros(7)},
        {"z_clean": np.zeros((1, 7))},
    ],
)
def test_invalid_inputs_fail_explicitly(changes: dict[str, np.ndarray]) -> None:
    with pytest.raises(ProbeActionConsistencyError):
        consistency_metrics(_inputs(**changes))
