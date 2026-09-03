"""Regression tests for the shared OpenVLA deployment-view contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from openvla_policy_view import (  # noqa: E402
    DeploymentViewSpecification,
    PolicyViewTransform,
    build_policy_and_replay_views,
    deployment_center_crop_uint8,
    resize_policy_pre_crop_canvas,
)


def _source(resolution: int) -> np.ndarray:
    y, x = np.mgrid[:resolution, :resolution]
    return np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(
        np.uint8
    )


def test_policy_input_is_independent_from_replay_resolution() -> None:
    specification = DeploymentViewSpecification(
        source_resolution=8, pre_crop_resolution=6, crop_area=0.9
    )
    source = _source(8)

    policy_a, replay_a = build_policy_and_replay_views(
        source, replay_resolution=4, specification=specification
    )
    policy_b, replay_b = build_policy_and_replay_views(
        source, replay_resolution=7, specification=specification
    )

    np.testing.assert_array_equal(policy_a, policy_b)
    assert replay_a.shape == (4, 4, 3)
    assert replay_b.shape == (7, 7, 3)


def test_policy_pre_crop_canvas_uses_explicit_bicubic_resize() -> None:
    specification = DeploymentViewSpecification(
        source_resolution=8, pre_crop_resolution=6, crop_area=0.9
    )
    source = _source(8)

    canvas = resize_policy_pre_crop_canvas(source, specification=specification)

    assert canvas.shape == (6, 6, 3)
    assert canvas.dtype == np.uint8


def test_rollout_and_training_center_crop_share_one_specification() -> None:
    specification = DeploymentViewSpecification(
        source_resolution=8, pre_crop_resolution=6, crop_area=0.9
    )
    source = _source(8)
    canvas = resize_policy_pre_crop_canvas(source, specification=specification)
    rollout_effective = deployment_center_crop_uint8(
        canvas, specification=specification
    )
    source_tensor = (
        torch.from_numpy(source.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .requires_grad_(True)
    )

    training_effective = PolicyViewTransform(specification)(source_tensor)

    assert rollout_effective.shape == (6, 6, 3)
    assert training_effective.shape == (1, 3, 6, 6)
    training_effective.square().mean().backward()
    assert source_tensor.grad is not None
    assert bool(torch.isfinite(source_tensor.grad).all())
    assert float(source_tensor.grad.abs().sum()) > 0.0
