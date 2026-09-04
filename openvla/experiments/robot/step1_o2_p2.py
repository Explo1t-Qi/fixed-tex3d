"""Frozen Step 1 O2/P2 contracts and dependency-light analysis helpers.

The OpenVLA extractor follows the validated implementation in
``shared-feature-tex3d`` at commit
``f0b7b644862fc002b1992c5c4618bf8f2c7e5177``.  In particular, O2 is the
multimodal-projector output over checkpoint-ordered DINOv2 and SigLIP visual
tokens.  P2 extraction remains in the separate OpenPI witness process.
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


LEGACY_OBJECTIVE = "legacy"
O2_DISPLACEMENT_OBJECTIVE = "o2_displacement"
ATTACK_OBJECTIVES = (LEGACY_OBJECTIVE, O2_DISPLACEMENT_OBJECTIVE)
O2_FEATURE_SHAPE = (256, 4096)
P2_FEATURE_SHAPE = (256, 2048)
RAW_RGB_SHAPE = (512, 512, 3)


def validate_attack_objective(value: str) -> str:
    if value not in ATTACK_OBJECTIVES:
        raise ValueError(
            f"attack_objective must be one of {ATTACK_OBJECTIVES}, got {value!r}"
        )
    return value


def legacy_attack_objective(
    action_loss: torch.Tensor,
    feature_loss: torch.Tensor,
    *,
    alpha_action: float,
    alpha_feature: float,
) -> torch.Tensor:
    """Preserve the pre-Step-1 Tex3D objective exactly."""
    return alpha_action * action_loss + alpha_feature * feature_loss


def assert_pi05_not_loaded() -> None:
    """Fail closed if OpenPI enters the single-surrogate training process."""
    forbidden = sorted(
        name
        for name in sys.modules
        if name == "openpi"
        or name.startswith("openpi.")
        or name == "openpi_client"
        or name.startswith("openpi_client.")
    )
    if forbidden:
        raise RuntimeError(
            "pi0.5/OpenPI must not be loaded during OpenVLA training: "
            f"{forbidden[:5]}"
        )


def freeze_openvla_for_o2(model: Any) -> None:
    """Freeze model weights while retaining gradients to the visual input."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def initialize_o2_texture_parameter(
    parameter: torch.Tensor, *, scale: float, seed: int
) -> None:
    """Escape the mathematically zero gradient at exact clean initialization."""
    if not torch.is_floating_point(parameter) or parameter.numel() == 0:
        raise ValueError("texture parameter must be a non-empty floating tensor")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("O2 initialization scale must be finite and positive")
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-scale, scale, size=tuple(parameter.shape)).astype(
        np.float32
    )
    with torch.no_grad():
        parameter.copy_(
            torch.from_numpy(initial).to(device=parameter.device, dtype=parameter.dtype)
        )
    if not bool(torch.isfinite(parameter).all()) or not bool(torch.any(parameter != 0)):
        raise RuntimeError("O2 texture initialization is zero or non-finite")


def extract_openvla_o2(model: Any, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return native OpenVLA O2, matching the frozen shared-feature node."""
    if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
        raise ValueError("OpenVLA pixel_values must be a rank-4 tensor")
    if pixel_values.shape[1] != 6:
        raise ValueError("OpenVLA fused pixel_values must contain six channels")

    vision_backbone = getattr(model, "vision_backbone", None)
    projector = getattr(model, "projector", None)
    featurizer = getattr(vision_backbone, "featurizer", None)
    fused_featurizer = getattr(vision_backbone, "fused_featurizer", None)
    if not callable(featurizer) or not callable(fused_featurizer):
        raise ValueError(
            "OpenVLA model must expose vision_backbone featurizer branches"
        )
    if not callable(projector):
        raise ValueError("OpenVLA model must expose a callable projector")

    dino_pixels, siglip_pixels = torch.split(pixel_values, [3, 3], dim=1)
    dino_feature = featurizer(dino_pixels)
    siglip_feature = fused_featurizer(siglip_pixels)
    expected_batch = pixel_values.shape[0]
    if tuple(dino_feature.shape) != (expected_batch, 256, 1024):
        raise ValueError(
            "OpenVLA DINOv2 feature must have shape "
            f"({expected_batch}, 256, 1024), got {tuple(dino_feature.shape)}"
        )
    if tuple(siglip_feature.shape) != (expected_batch, 256, 1152):
        raise ValueError(
            "OpenVLA SigLIP feature must have shape "
            f"({expected_batch}, 256, 1152), got {tuple(siglip_feature.shape)}"
        )
    o2 = projector(torch.cat((dino_feature, siglip_feature), dim=-1))
    expected_o2 = (expected_batch, *O2_FEATURE_SHAPE)
    if tuple(o2.shape) != expected_o2:
        raise ValueError(
            f"OpenVLA O2 must have shape {expected_o2}, got {tuple(o2.shape)}"
        )
    if not bool(torch.isfinite(o2).all()):
        raise ValueError("OpenVLA O2 contains non-finite values")
    return o2


def detached_clean_o2(model: Any, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        clean_o2 = extract_openvla_o2(model, pixel_values)
    clean_o2 = clean_o2.detach()
    if clean_o2.requires_grad:
        raise RuntimeError("clean OpenVLA O2 must be detached")
    return clean_o2


def o2_displacement_loss(
    adv_o2: torch.Tensor, clean_o2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(-mean squared displacement, positive displacement)``."""
    if tuple(adv_o2.shape) != tuple(clean_o2.shape):
        raise ValueError("clean and adversarial O2 shapes must match")
    if adv_o2.ndim != 3 or tuple(adv_o2.shape[1:]) != O2_FEATURE_SHAPE:
        raise ValueError(
            f"O2 tensors must have shape [B, {O2_FEATURE_SHAPE[0]}, "
            f"{O2_FEATURE_SHAPE[1]}]"
        )
    if clean_o2.requires_grad:
        raise ValueError("clean O2 must be detached")
    displacement = ((adv_o2.float() - clean_o2.float()) ** 2).mean()
    return -displacement, displacement


def token_rms(residual: np.ndarray) -> np.ndarray:
    """Reduce ``[..., token, feature]`` residuals to token-wise RMS."""
    value = np.asarray(residual)
    if value.ndim < 2 or value.shape[-2] != 256 or value.shape[-1] <= 0:
        raise ValueError("residual must have shape [..., 256, feature_dim]")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError("residual must use a floating dtype")
    result = np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=-1))
    if not np.all(np.isfinite(result)):
        raise ValueError("token RMS contains non-finite values")
    return result


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_rgb(image: np.ndarray) -> str:
    value = np.asarray(image)
    if value.shape != RAW_RGB_SHAPE or value.dtype != np.uint8:
        raise ValueError(f"raw RGB must have shape {RAW_RGB_SHAPE} and dtype uint8")
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def load_pair_metadata(path: str | Path) -> dict[str, Any]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "sample_id",
        "state_id",
        "clean_rgb_sha256",
        "adv_rgb_sha256",
        "texture_sha256",
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("pair metadata is missing required identity fields")
    return metadata


def verify_pair_identity(
    metadata: Mapping[str, Any],
    *,
    sample_id: str,
    clean_rgb: np.ndarray,
    adv_rgb: np.ndarray,
) -> None:
    expected = {
        "sample_id": sample_id,
        "clean_rgb_sha256": sha256_rgb(clean_rgb),
        "adv_rgb_sha256": sha256_rgb(adv_rgb),
    }
    actual = {name: metadata.get(name) for name in expected}
    if actual != expected:
        raise ValueError(
            f"raw pair identity mismatch: expected={expected}, actual={actual}"
        )


def extract_pi05_p2_no_grad(
    *,
    model: Any,
    base_image: Any,
    image_encoder: Callable[..., Any] | None = None,
) -> Any:
    """Run the frozen P2 node without an autograd graph.

    The released formal checkpoint is JAX/NNX, where ordinary forward calls do
    not record gradients.  The optional ``eval`` call and PyTorch no-grad guard
    also make this contract safe for the validated PyTorch-compatible backend.
    """
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    encoder = image_encoder or getattr(getattr(model, "PaliGemma", None), "img", None)
    if not callable(encoder):
        raise ValueError("pi0.5 model must expose callable PaliGemma.img")
    context = torch.no_grad() if hasattr(torch, "no_grad") else nullcontext()
    with context:
        output = encoder(base_image, train=False)
    p2 = output[0] if isinstance(output, tuple) else output
    expected = (int(base_image.shape[0]), *P2_FEATURE_SHAPE)
    if tuple(getattr(p2, "shape", ())) != expected:
        raise ValueError(f"pi0.5 P2 must have shape {expected}")
    if bool(getattr(p2, "requires_grad", False)):
        raise RuntimeError("pi0.5 P2 witness must not require gradients")
    return p2
