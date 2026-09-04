"""Regression tests for the frozen Step 1 O2/P2 contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from step1_o2_p2 import (  # noqa: E402
    LEGACY_OBJECTIVE,
    assert_pi05_not_loaded,
    detached_clean_o2,
    extract_openvla_o2,
    extract_pi05_p2_no_grad,
    initialize_o2_texture_parameter,
    legacy_attack_objective,
    o2_displacement_loss,
    sha256_rgb,
    token_rms,
    validate_attack_objective,
    verify_pair_identity,
)


class _Branch(nn.Module):
    def __init__(self, width: int, offset: float) -> None:
        super().__init__()
        self.width = width
        self.offset = offset

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        marker = value.mean(dim=(1, 2, 3)).reshape(-1, 1, 1) + self.offset
        return marker.expand(-1, 256, self.width)


class _Projector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        marker = value.mean(dim=-1, keepdim=True)
        return marker.expand(-1, 256, 4096)


class _OpenVLAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_backbone = nn.Module()
        self.vision_backbone.featurizer = _Branch(1024, 1.0)
        self.vision_backbone.fused_featurizer = _Branch(1152, 2.0)
        self.projector = _Projector()


def test_openvla_o2_has_frozen_shape_and_projector_node() -> None:
    model = _OpenVLAModel()
    pixels = torch.cat(
        (torch.ones((2, 3, 4, 4)), torch.full((2, 3, 4, 4), 3.0)), dim=1
    )

    o2 = extract_openvla_o2(model, pixels)

    assert o2.shape == (2, 256, 4096)
    assert bool(torch.isfinite(o2).all())
    assert model.projector.last_input is not None
    assert model.projector.last_input.shape == (2, 256, 2176)
    torch.testing.assert_close(
        model.projector.last_input[:, :, :1024],
        torch.full((2, 256, 1024), 2.0),
    )
    torch.testing.assert_close(
        model.projector.last_input[:, :, 1024:],
        torch.full((2, 256, 1152), 5.0),
    )


def test_negative_mse_objective_sign_is_correct() -> None:
    clean = torch.zeros((1, 256, 4096))
    near = torch.ones_like(clean)
    far = torch.full_like(clean, 2.0)

    near_loss, near_displacement = o2_displacement_loss(near, clean)
    far_loss, far_displacement = o2_displacement_loss(far, clean)

    assert far_displacement > near_displacement
    assert far_loss < near_loss
    assert near_loss.item() == -1.0
    assert far_loss.item() == -4.0


def test_clean_o2_is_detached() -> None:
    model = _OpenVLAModel()
    pixels = torch.ones((1, 6, 2, 2), requires_grad=True)

    clean = detached_clean_o2(model, pixels)

    assert clean.shape == (1, 256, 4096)
    assert clean.requires_grad is False
    assert clean.grad_fn is None


def test_adv_o2_propagates_finite_nonzero_image_and_texture_gradient() -> None:
    model = _OpenVLAModel()
    clean_pixels = torch.zeros((1, 6, 2, 2))
    clean = detached_clean_o2(model, clean_pixels)
    texture = torch.tensor(0.25, requires_grad=True)
    image = texture.expand(1, 6, 2, 2)
    image.retain_grad()

    adv = extract_openvla_o2(model, image)
    loss, _ = o2_displacement_loss(adv, clean)
    loss.backward()

    assert image.grad is not None
    assert bool(torch.isfinite(image.grad).all())
    assert float(image.grad.abs().sum()) > 0.0
    assert texture.grad is not None
    assert bool(torch.isfinite(texture.grad))
    assert float(texture.grad.abs()) > 0.0


def test_o2_initialization_is_seeded_nonzero_and_bounded() -> None:
    first = torch.zeros((32, 3))
    second = torch.zeros_like(first)

    initialize_o2_texture_parameter(first, scale=0.05, seed=7)
    initialize_o2_texture_parameter(second, scale=0.05, seed=7)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert bool(torch.isfinite(first).all())
    assert float(first.abs().max()) <= 0.05
    assert float(first.abs().sum()) > 0.0


def test_legacy_remains_the_default_objective_contract() -> None:
    assert validate_attack_objective(LEGACY_OBJECTIVE) == "legacy"
    action = torch.tensor(3.0, requires_grad=True)
    feature = torch.tensor(-2.0, requires_grad=True)
    combined = legacy_attack_objective(
        action, feature, alpha_action=1.5, alpha_feature=4.0
    )
    assert combined.item() == pytest.approx(-3.5)
    combined.backward()
    assert action.grad.item() == pytest.approx(1.5)
    assert feature.grad.item() == pytest.approx(4.0)
    with pytest.raises(ValueError, match="attack_objective"):
        validate_attack_objective("combined_o2_p2")

    entrypoint = (
        Path(__file__).resolve().parents[2]
        / "openvla/experiments/robot/libero/attack_openvla.py"
    ).read_text(encoding="utf-8")
    assert "attack_objective:       str            = LEGACY_OBJECTIVE" in entrypoint
    assert "if attack_objective == LEGACY_OBJECTIVE:" in entrypoint
    assert "Preserve the historical legacy logging reduction." in entrypoint


@pytest.mark.parametrize("width", [4096, 2048])
def test_token_rms_reduction_is_correct(width: int) -> None:
    residual = np.zeros((2, 256, width), dtype=np.float32)
    residual[0].fill(3.0)
    residual[1].fill(4.0)

    result = token_rms(residual)

    assert result.shape == (2, 256)
    np.testing.assert_allclose(result[0], 3.0)
    np.testing.assert_allclose(result[1], 4.0)


class _Pi05Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()

    def encode(self, image: torch.Tensor, *, train: bool):
        assert train is False
        p2 = self.weight * torch.ones(
            (image.shape[0], 256, 2048), device=image.device
        )
        return p2, {"encoded": torch.ones((image.shape[0], 256, 1152))}


def test_pi05_p2_witness_is_eval_no_grad_and_has_frozen_shape() -> None:
    model = _Pi05Model()
    image = torch.zeros((2, 224, 224, 3), requires_grad=True)

    p2 = extract_pi05_p2_no_grad(
        model=model,
        base_image=image,
        image_encoder=model.encode,
    )

    assert model.eval_called is True
    assert model.training is False
    assert p2.shape == (2, 256, 2048)
    assert p2.requires_grad is False
    assert p2.grad_fn is None


def test_sample_and_hash_identity_check_fails_closed(tmp_path) -> None:
    clean = np.zeros((512, 512, 3), dtype=np.uint8)
    adv = np.ones((512, 512, 3), dtype=np.uint8)
    metadata = {
        "sample_id": "state-10",
        "state_id": 10,
        "clean_rgb_sha256": sha256_rgb(clean),
        "adv_rgb_sha256": sha256_rgb(adv),
        "texture_sha256": "sha256:" + "0" * 64,
    }

    verify_pair_identity(
        metadata,
        sample_id="state-10",
        clean_rgb=clean,
        adv_rgb=adv,
    )
    changed = adv.copy()
    changed[0, 0, 0] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_pair_identity(
            metadata,
            sample_id="state-10",
            clean_rgb=clean,
            adv_rgb=changed,
        )

    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert json.loads(path.read_text())["sample_id"] == "state-10"


def test_training_guard_rejects_loaded_openpi(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openpi.fake", object())
    with pytest.raises(RuntimeError, match="must not be loaded"):
        assert_pi05_not_loaded()
