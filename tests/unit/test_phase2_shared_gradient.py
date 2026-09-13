"""CPU contracts for the Phase 2.3 dual-VLA gradient wiring."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from phase2_shared_gradient import (  # noqa: E402
    DualVLAFeatureAdapter,
    PI05_IMAGE_KEYS,
    Phase2GradientClosureError,
    Pi05BaseImageAdapter,
    checked_gradient,
    extract_pi05_p2_autograd,
    gradient_cosine,
)


class _Observation:
    @classmethod
    def from_dict(cls, values):
        return SimpleNamespace(
            images=values["image"],
            image_masks=values["image_mask"],
            state=values["state"],
            tokenized_prompt=values["tokenized_prompt"],
            tokenized_prompt_mask=values["tokenized_prompt_mask"],
        )


class _Policy:
    _is_pytorch_model = True

    @staticmethod
    def _input_transform(values):
        base = np.asarray(values["observation/image"])
        wrist = np.asarray(values["observation/wrist_image"])

        def normalized_chw(image):
            return np.transpose(image.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))

        return {
            "state": np.pad(np.asarray(values["observation/state"]), (0, 24)),
            "image": {
                "base_0_rgb": normalized_chw(base),
                "left_wrist_0_rgb": normalized_chw(wrist),
                "right_wrist_0_rgb": np.zeros((3, 224, 224), dtype=np.float32),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "tokenized_prompt": np.zeros(200, dtype=np.int64),
            "tokenized_prompt_mask": np.ones(200, dtype=bool),
        }


class _PiEncoder:
    @staticmethod
    def embed_image(image: torch.Tensor) -> torch.Tensor:
        value = image.mean(dim=(1, 2, 3))
        return value[:, None, None].expand(-1, 256, 2048)


class _PiModel:
    paligemma_with_expert = _PiEncoder()

    @staticmethod
    def _preprocess_observation(observation, *, train):
        assert train is False
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )


def _policy_input():
    return {
        "observation/image": np.full((224, 224, 3), 64, dtype=np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 96, dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "put the bowl on the plate",
    }


def _pi_adapter() -> Pi05BaseImageAdapter:
    return Pi05BaseImageAdapter(
        policy=_Policy(),
        observation_type=_Observation,
        policy_input=_policy_input(),
        device="cpu",
    )


def test_pi05_adapter_replaces_only_base_slot_and_preserves_autograd() -> None:
    adapter = _pi_adapter()
    clean = adapter.clean_observation
    source = torch.full((1, 3, 512, 512), 0.75, requires_grad=True)

    adversarial = adapter.observation_for_base_image(source)
    p2 = extract_pi05_p2_autograd(_PiModel(), adversarial)

    assert tuple(adversarial.images) == PI05_IMAGE_KEYS
    assert p2.shape == (1, 256, 2048)
    assert torch.equal(
        adversarial.images["left_wrist_0_rgb"],
        clean.images["left_wrist_0_rgb"],
    )
    assert torch.equal(
        adversarial.images["right_wrist_0_rgb"],
        clean.images["right_wrist_0_rgb"],
    )
    assert torch.equal(
        adversarial.image_masks["base_0_rgb"],
        clean.image_masks["base_0_rgb"],
    )
    p2.square().mean().backward()
    assert source.grad is not None
    assert bool(torch.isfinite(source.grad).all())
    assert bool(torch.any(source.grad != 0))


def test_pi05_adapter_rejects_changed_slot_order_and_bad_source() -> None:
    class BadPolicy(_Policy):
        @staticmethod
        def _input_transform(values):
            result = _Policy._input_transform(values)
            result["image"] = dict(reversed(tuple(result["image"].items())))
            return result

    with pytest.raises(Phase2GradientClosureError, match="image-slot ordering"):
        Pi05BaseImageAdapter(
            policy=BadPolicy(),
            observation_type=_Observation,
            policy_input=_policy_input(),
            device="cpu",
        )
    with pytest.raises(Phase2GradientClosureError, match="source image"):
        _pi_adapter().observation_for_base_image(torch.zeros(3, 512, 512))


class _Mapping(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("projection", torch.ones(width, 262) / width)

    def map_o2(self, value):
        return value @ self.projection

    def map_p2(self, value):
        return value @ self.projection


@dataclass(frozen=True)
class _LossResult:
    loss: torch.Tensor
    shared_mse: torch.Tensor
    o2_mse: torch.Tensor
    p2_mse: torch.Tensor
    displacement_cosine_mean: torch.Tensor
    o2_to_p2_mse_ratio: torch.Tensor


def _shared_loss(h_o_adv, h_p_adv, h_o_clean, h_p_clean):
    delta_o = h_o_adv - h_o_clean.detach()
    delta_p = h_p_adv - h_p_clean.detach()
    shared = ((delta_o + delta_p) / 2).square().mean()
    o2_mse = delta_o.square().mean()
    p2_mse = delta_p.square().mean()
    cosine = torch.nn.functional.cosine_similarity(
        delta_o.flatten(1), delta_p.flatten(1), dim=1, eps=1e-12
    ).mean()
    return _LossResult(
        loss=-shared,
        shared_mse=shared,
        o2_mse=o2_mse,
        p2_mse=p2_mse,
        displacement_cosine_mean=cosine,
        o2_to_p2_mse_ratio=o2_mse / (p2_mse + 1e-12),
    )


def _feature_path(width: int, scale: float):
    def path(image):
        values = image.mean(dim=(2, 3), keepdim=False).mean(dim=1) * scale
        return values[:, None, None].expand(-1, 256, width)

    return path


def _dual_adapter() -> DualVLAFeatureAdapter:
    return DualVLAFeatureAdapter(
        openvla_path=_feature_path(4096, 1.0),
        pi05_path=_feature_path(2048, 2.0),
        openvla_mapping=_Mapping(4096),
        pi05_mapping=_Mapping(2048),
        shared_loss=_shared_loss,
        openvla_device="cpu",
        pi05_device="cpu",
        loss_device="cpu",
    )


def test_joint_adapter_preserves_shapes_and_clean_detach_semantics() -> None:
    adapter = _dual_adapter()
    clean_image = torch.full((1, 3, 512, 512), 0.25, requires_grad=True)
    clean = adapter.clean_reference(clean_image)
    adversarial = torch.full(
        (1, 3, 512, 512), 0.5, requires_grad=True
    )

    result, features = adapter.loss(adversarial, clean)

    assert features.o2.shape == (1, 256, 4096)
    assert features.p2.shape == (1, 256, 2048)
    assert features.h_o.shape == (1, 256, 262)
    assert features.h_p.shape == (1, 256, 262)
    assert result.loss.shape == ()
    assert clean.h_o.requires_grad is False
    assert clean.h_p.requires_grad is False
    assert clean_image.grad is None

    openvla_gradient = checked_gradient(
        result.o2_mse,
        adversarial,
        label="OpenVLA branch image",
        retain_graph=True,
    )
    pi05_gradient = checked_gradient(
        result.p2_mse,
        adversarial,
        label="PI0Pytorch branch image",
        retain_graph=True,
    )
    joint_gradient = checked_gradient(
        result.loss,
        adversarial,
        label="joint shared-loss image",
        retain_graph=False,
    )
    assert gradient_cosine(openvla_gradient, pi05_gradient) > 0.99
    for gradient in (openvla_gradient, pi05_gradient, joint_gradient):
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.norm()) > 0.0


def test_joint_adapter_fails_at_feature_shape_and_finite_boundaries() -> None:
    malformed = _dual_adapter()
    malformed.openvla_path = lambda image: torch.zeros(1, 255, 4096)
    with pytest.raises(Phase2GradientClosureError, match="OpenVLA O2 must have"):
        malformed.features(torch.zeros(1, 3, 512, 512))

    nonfinite = _dual_adapter()
    nonfinite.pi05_path = lambda image: torch.full(
        (1, 256, 2048), float("nan")
    )
    with pytest.raises(Phase2GradientClosureError, match="P2 must be finite"):
        nonfinite.features(torch.zeros(1, 3, 512, 512))


def test_authoritative_entrypoint_has_one_update_and_no_witness_bridge() -> None:
    entrypoint = (
        Path(__file__).resolve().parents[2]
        / "scripts/phase2_shared_gradient_smoke.py"
    ).read_text(encoding="utf-8")

    assert entrypoint.count("optimizer.step()") == 1
    assert "extract_pi05_p2_no_grad" not in entrypoint
    assert "extract_pi05_p2_autograd" in entrypoint
    assert "shared_feature_loss" in entrypoint


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cross_device_copy_preserves_joint_autograd() -> None:
    adapter = DualVLAFeatureAdapter(
        openvla_path=_feature_path(4096, 1.0),
        pi05_path=_feature_path(2048, 2.0),
        openvla_mapping=_Mapping(4096).to("cuda:0"),
        pi05_mapping=_Mapping(2048).to("cuda:1"),
        shared_loss=_shared_loss,
        openvla_device="cuda:0",
        pi05_device="cuda:1",
        loss_device="cuda:0",
    )
    clean = adapter.clean_reference(torch.zeros(1, 3, 512, 512, device="cuda:0"))
    adversarial = torch.ones(
        1, 3, 512, 512, device="cuda:0", requires_grad=True
    )

    result, _ = adapter.loss(adversarial, clean)
    result.loss.backward()

    assert adversarial.grad is not None
    assert bool(torch.isfinite(adversarial.grad).all())
    assert bool(torch.any(adversarial.grad != 0))
