"""Tests for checkpoint-derived differentiable OpenVLA preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from openvla_image_transform import (  # noqa: E402
    DifferentiableOpenVLAImageProcessor,
    ExactForwardSurrogateBackwardOpenVLAImageProcessor,
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


class _ExactImageProcessor:
    def __init__(self) -> None:
        self.last_output: torch.Tensor | None = None
        self.last_images = None

    def __call__(self, images, *, return_tensors):
        assert return_tensors == "pt"
        self.last_images = images
        self.last_output = torch.arange(
            len(images) * 6 * 2 * 2, dtype=torch.float32
        ).reshape(len(images), 6, 2, 2)
        return {"pixel_values": self.last_output}


class _GradientBearingExactImageProcessor:
    def __init__(self) -> None:
        self.parameter = torch.tensor(0.25, requires_grad=True)

    def __call__(self, images, *, return_tensors):
        assert return_tensors == "pt"
        return {
            "pixel_values": self.parameter
            * torch.ones((len(images), 6, 2, 2), dtype=torch.float32)
        }


def test_exact_forward_surrogate_backward_is_bit_exact_to_official() -> None:
    surrogate = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(),
    )
    exact = _ExactImageProcessor()
    processor = ExactForwardSurrogateBackwardOpenVLAImageProcessor(
        official_image_processor=exact,
        surrogate=surrogate,
    )
    image = torch.full((1, 3, 4, 4), 0.75, dtype=torch.float32)

    fused = processor(image)

    assert exact.last_output is not None
    assert torch.equal(fused.cpu(), exact.last_output)


def test_exact_forward_surrogate_backward_has_finite_input_gradient() -> None:
    surrogate = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(),
    )
    processor = ExactForwardSurrogateBackwardOpenVLAImageProcessor(
        official_image_processor=_ExactImageProcessor(),
        surrogate=surrogate,
    )
    image = torch.full(
        (1, 3, 4, 4), 0.75, dtype=torch.float32, requires_grad=True
    )

    processor(image).square().mean().backward()

    assert image.grad is not None
    assert bool(torch.isfinite(image.grad).all())
    assert float(image.grad.abs().sum()) > 0.0


def test_exact_forward_branch_receives_no_gradient() -> None:
    surrogate = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(),
    )
    exact = _GradientBearingExactImageProcessor()
    processor = ExactForwardSurrogateBackwardOpenVLAImageProcessor(
        official_image_processor=exact,
        surrogate=surrogate,
    )
    image = torch.full(
        (1, 3, 4, 4), 0.75, dtype=torch.float32, requires_grad=True
    )

    processor(image).sum().backward()

    assert exact.parameter.grad is None
    assert image.grad is not None


def test_exact_forward_preserves_official_shape_dtype_and_branch_order() -> None:
    surrogate = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(),
    )
    processor = ExactForwardSurrogateBackwardOpenVLAImageProcessor(
        official_image_processor=_ExactImageProcessor(),
        surrogate=surrogate,
    )
    image = torch.full((2, 3, 4, 4), 0.75, dtype=torch.float64)

    fused = processor(image)

    assert fused.shape == (2, 6, 2, 2)
    assert fused.dtype == torch.float32
    assert processor.branch_model_ids == ("dinov2", "siglip")
    assert processor.output_size == (2, 2)


def test_exact_forward_restores_deployment_uint8_before_official_processor() -> None:
    surrogate = DifferentiableOpenVLAImageProcessor.from_checkpoint(
        model=_model("dinov2", "siglip"),
        processor=_processor(),
    )
    exact = _ExactImageProcessor()
    processor = ExactForwardSurrogateBackwardOpenVLAImageProcessor(
        official_image_processor=exact,
        surrogate=surrogate,
    )
    deployment_uint8 = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    image = (
        torch.from_numpy(deployment_uint8.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )

    processor(image)

    assert exact.last_images is not None
    assert np.array_equal(np.asarray(exact.last_images[0]), deployment_uint8)


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
