"""CPU tests for multi-instance renderer and visibility composition contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from openvla_renderer_contracts import (  # noqa: E402
    DEFAULT_RENDERER_POSITION_OFFSET,
    build_frontmost_instance_masks,
    capture_frontmost_instance_masks,
    compose_visibility_masked_renderer_delta,
    find_target_body_poses,
)


class _Model:
    nbody = 4
    ngeom = 4
    body_parentid = np.array([0, 0, 0, 0])
    geom_bodyid = np.array([0, 1, 2, 3])

    @staticmethod
    def body_id2name(body_id: int) -> str:
        return ("world", "akita_black_bowl_1", "akita_black_bowl_2", "plate")[
            body_id
        ]


def _environment() -> SimpleNamespace:
    data = SimpleNamespace(
        body_xpos=np.zeros((4, 3), dtype=np.float32),
        body_xquat=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)
        ),
    )
    return SimpleNamespace(sim=SimpleNamespace(model=_Model(), data=data))


def test_renderer_default_position_offset_is_exact_zero() -> None:
    assert DEFAULT_RENDERER_POSITION_OFFSET == (0.0, 0.0, 0.0)


def test_all_shared_texture_instances_are_discovered() -> None:
    poses = find_target_body_poses(
        _environment(), (("akita_black_bowl",), ("bowl",)), device="cpu"
    )

    assert [pose.body_id for pose in poses] == [1, 2]
    assert [pose.body_name for pose in poses] == [
        "akita_black_bowl_1",
        "akita_black_bowl_2",
    ]


def test_frontmost_segmentation_is_split_into_mutually_exclusive_masks() -> None:
    # Channel 0 is object type; channel 1 is object/geom id.
    segmentation = np.array(
        [[[5, 1], [5, 2]], [[5, 3], [-1, -1]]], dtype=np.int32
    )

    masks = build_frontmost_instance_masks(
        segmentation,
        model=_Model(),
        body_ids=(1, 2),
        geom_object_type=5,
    )

    assert masks.shape == (2, 1, 2, 2)
    assert masks[:, 0, 0, 0].tolist() == [1.0, 0.0]
    assert masks[:, 0, 0, 1].tolist() == [0.0, 1.0]
    assert bool((masks.sum(dim=0) <= 1.0).all())


def test_segmentation_capture_uses_the_policy_camera_orientation() -> None:
    environment = _environment()
    raw = np.array(
        [[[5, 1], [5, 2]], [[5, 3], [-1, -1]]], dtype=np.int32
    )
    environment.sim.render = lambda **_: raw

    masks = capture_frontmost_instance_masks(
        environment,
        body_ids=(1, 2),
        resolution=2,
        geom_object_type=5,
    )

    # Double-axis orientation puts raw [0,0] at output [1,1].
    assert masks[:, 0, 1, 1].tolist() == [1.0, 0.0]


def test_zero_delta_returns_exact_clean_mujoco_image() -> None:
    clean = torch.rand((1, 3, 2, 2), dtype=torch.float32)
    alpha = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 1.0], [0.0, 0.0]]]]
    )
    renderer_clean = torch.rand((2, 3, 2, 2), dtype=torch.float32)
    renderer_mask = torch.ones((2, 1, 2, 2), dtype=torch.float32)

    composited = compose_visibility_masked_renderer_delta(
        clean,
        alpha,
        renderer_clean.clone(),
        renderer_clean,
        renderer_mask,
    )

    torch.testing.assert_close(composited, clean, rtol=0.0, atol=0.0)


def test_visibility_gates_delta_and_shared_parameter_gradients_accumulate() -> None:
    parameter = torch.tensor(0.25, requires_grad=True)
    clean = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    alpha = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 1.0], [0.0, 0.0]]]]
    )
    renderer_clean = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    renderer_adv = renderer_clean + parameter
    renderer_mask = torch.ones((2, 1, 2, 2), dtype=torch.float32)

    composited = compose_visibility_masked_renderer_delta(
        clean, alpha, renderer_adv, renderer_clean, renderer_mask
    )
    reversed_composited = compose_visibility_masked_renderer_delta(
        clean,
        alpha.flip(0),
        renderer_adv.flip(0),
        renderer_clean.flip(0),
        renderer_mask.flip(0),
    )

    torch.testing.assert_close(composited, reversed_composited)
    assert torch.count_nonzero(composited) == 6
    composited.sum().backward()
    assert parameter.grad is not None
    assert float(parameter.grad) == 6.0
