"""Tests for checkpoint-derived differentiable OpenVLA preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from openvla_image_transform import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
)


def _model(*model_ids: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(timm_model_ids=model_ids)
    )


def _processor(
    *,
    interpolations: tuple[object, object] = (3, 3),
    antialias: tuple[bool, bool] = (True, True),
) -> SimpleNamespace:
    return SimpleNamespace(
        image_processor=SimpleNamespace(
            image_resize_strategy="resize-naive",
            input_sizes=((3, 2, 2), (3, 2, 2)),
            tvf_resize_params=(
                {
                    "size": (2, 2),
                    "interpolation": interpolations[0],
                    "antialias": antialias[0],
                    "max_size": None,
                },
                {
                    "size": (2, 2),
                    "interpolation": interpolations[1],
                    "antialias": antialias[1],
                    "max_size": None,
                },
            ),
            tvf_crop_params=(
                {"output_size": (2, 2)},
                {"output_size": (2, 2)},
            ),
            tvf_normalize_params=(
                {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            ),
        )
    )


def test_fused_pixels_follow_checkpoint_branch_order_and_normalization() -> None:
    processor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(),
    )
    image = torch.full(
        (1, 3, 4, 4), 0.75, dtype=torch.float32, requires_grad=True
    )

    fused = processor(image)

    assert processor.branch_model_ids == ("dinov2", "siglip")
    assert fused.shape == (1, 6, 2, 2)
    torch.testing.assert_close(fused[:, :3], torch.full((1, 3, 2, 2), 0.75))
    torch.testing.assert_close(fused[:, 3:], torch.full((1, 3, 2, 2), 0.5))
    fused.sum().backward()
    assert image.grad is not None
    assert bool(torch.isfinite(image.grad).all())
    assert float(image.grad.abs().sum()) > 0.0


def test_branch_order_is_not_inferred_from_branch_names() -> None:
    processor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("siglip", "dinov2"),
        processor=_processor(),
    )
    image = torch.full((1, 3, 2, 2), 0.75)

    fused = processor(image)

    assert processor.branch_model_ids == ("siglip", "dinov2")
    # Processor arrays and model IDs are positional; no semantic-name reordering occurs.
    torch.testing.assert_close(fused[:, :3], torch.full((1, 3, 2, 2), 0.75))
    torch.testing.assert_close(fused[:, 3:], torch.full((1, 3, 2, 2), 0.5))


def test_resize_mode_and_antialias_follow_processor_configuration() -> None:
    processor = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(interpolations=(3, 2), antialias=(True, False)),
    )
    image = torch.tensor(
        [[[[0.0, 1.0, 0.0, 1.0]] * 4] * 3], dtype=torch.float32
    )

    fused = processor(image)

    expected_dino = F.interpolate(
        image, size=(2, 2), mode="bicubic", align_corners=False, antialias=True
    )
    expected_siglip = F.interpolate(
        image, size=(2, 2), mode="bilinear", align_corners=False, antialias=False
    )
    torch.testing.assert_close(fused[:, :3], expected_dino)
    torch.testing.assert_close(fused[:, 3:], (expected_siglip - 0.5) / 0.5)


def test_invalid_processor_branch_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="branch count"):
        DifferentiableOpenVLAImageProcessor.from_checkpoint(
            model=_model("dinov2"),
            processor=_processor(),
        )


def test_clean_and_adversarial_images_share_one_preprocessing_contract() -> None:
    contract = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(),
    )
    clean = torch.zeros((1, 3, 4, 4))
    adversarial = clean.clone().requires_grad_(True)

    clean_pixels = contract(clean)
    adversarial_pixels = contract(adversarial)

    torch.testing.assert_close(clean_pixels, adversarial_pixels)
    adversarial_pixels.square().sum().backward()
    assert adversarial.grad is not None
    assert bool(torch.isfinite(adversarial.grad).all())
