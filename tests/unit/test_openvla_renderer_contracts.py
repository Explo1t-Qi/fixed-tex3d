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
    resolve_mujoco_object_id,
)
from openvla_image_transform import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)
from openvla_policy_view import (  # noqa: E402
    DeploymentViewSpecification,
    PolicyViewTransform,
)


class _Model:
    nbody = 4
    ngeom = 4
    nmat = 2
    body_parentid = np.array([0, 0, 0, 0])
    geom_bodyid = np.array([0, 1, 2, 3])
    geom_matid = np.array([1, 0, 0, 1])
    mat_texid = np.array([0, 1])

    @staticmethod
    def body_id2name(body_id: int) -> str:
        return ("world", "akita_black_bowl_1", "akita_black_bowl_2", "plate")[
            body_id
        ]

    @staticmethod
    def tex_name2id(name: str) -> int:
        return {"shared-akita": 0, "other": 1}[name]


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
        _environment(),
        (("akita_black_bowl",), ("bowl",)),
        device="cpu",
        texture_name="shared-akita",
    )

    assert [pose.body_id for pose in poses] == [1, 2]
    assert [pose.body_name for pose in poses] == [
        "akita_black_bowl_1",
        "akita_black_bowl_2",
    ]


def test_native_mujoco_name_api_discovers_shared_texture_instances(
    monkeypatch,
) -> None:
    """Native mujoco.MjModel exposes module functions, not name methods."""
    model = SimpleNamespace(
        nbody=4,
        ngeom=4,
        body_parentid=np.array([0, 0, 0, 0]),
        geom_bodyid=np.array([0, 1, 2, 3]),
        geom_matid=np.array([1, 0, 0, 1]),
        mat_texid=np.array([0, 1]),
    )
    body_names = ("world", "akita_black_bowl_1", "akita_black_bowl_2", "plate")
    object_types = SimpleNamespace(
        mjOBJ_TEXTURE=7, mjOBJ_BODY=1, mjOBJ_CAMERA=3
    )

    def name_to_id(candidate_model, object_type, name):
        assert candidate_model is model
        values = {
            (object_types.mjOBJ_TEXTURE, "shared-akita"): 0,
            (object_types.mjOBJ_CAMERA, "agentview"): 2,
        }
        return values.get((object_type, name), -1)

    def id_to_name(candidate_model, object_type, object_id):
        assert candidate_model is model
        assert object_type == object_types.mjOBJ_BODY
        return body_names[object_id]

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(
            mjtObj=object_types,
            mj_name2id=name_to_id,
            mj_id2name=id_to_name,
        ),
    )
    data = SimpleNamespace(
        xpos=np.zeros((4, 3), dtype=np.float32),
        xquat=np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)
        ),
    )
    environment = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))

    assert resolve_mujoco_object_id(model, "agentview", "camera") == 2

    poses = find_target_body_poses(
        environment,
        (("akita_black_bowl",),),
        device="cpu",
        texture_name="shared-akita",
    )

    assert [pose.body_name for pose in poses] == [
        "akita_black_bowl_1",
        "akita_black_bowl_2",
    ]


def test_robosuite_model_wrapper_is_unwrapped_for_native_name_lookup(
    monkeypatch,
) -> None:
    raw_model = object()
    wrapper = SimpleNamespace(_model=raw_model)
    object_types = SimpleNamespace(mjOBJ_TEXTURE=7)

    def name_to_id(candidate_model, object_type, name):
        assert candidate_model is raw_model
        assert object_type == object_types.mjOBJ_TEXTURE
        assert name == "shared-akita"
        return 4

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(mjtObj=object_types, mj_name2id=name_to_id),
    )

    assert resolve_mujoco_object_id(
        wrapper, "shared-akita", "texture"
    ) == 4


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


def test_compositor_deployment_and_preprocessing_chain_has_finite_texture_gradient(
) -> None:
    parameter = torch.tensor(0.1, requires_grad=True)
    clean = torch.full((1, 3, 8, 8), 0.4, dtype=torch.float32)
    alpha = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    renderer_clean = torch.full((1, 3, 8, 8), 0.3, dtype=torch.float32)
    renderer_adv = renderer_clean + parameter
    renderer_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    composited = compose_visibility_masked_renderer_delta(
        clean, alpha, renderer_adv, renderer_clean, renderer_mask
    )
    view_specification = DeploymentViewSpecification(
        source_resolution=8, pre_crop_resolution=6, crop_area=0.9
    )
    effective_view = PolicyViewTransform(view_specification)(composited)
    processor_config = SimpleNamespace(
        image_resize_strategy="resize-naive",
        input_sizes=((3, 6, 6),),
        tvf_resize_params=(
            {"size": (6, 6), "interpolation": 3, "antialias": True},
        ),
        tvf_crop_params=({"output_size": (6, 6)},),
        tvf_normalize_params=(
            {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        ),
    )
    preprocessor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=SimpleNamespace(
            config=SimpleNamespace(timm_model_ids=("vision-branch",))
        ),
        processor=SimpleNamespace(image_processor=processor_config),
    )

    loss = preprocessor(effective_view).square().mean()
    loss.backward()

    assert bool(torch.isfinite(loss))
    assert parameter.grad is not None
    assert bool(torch.isfinite(parameter.grad))
    assert float(parameter.grad.abs()) > 0.0
