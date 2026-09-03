"""Shared spatial contract for OpenVLA policy and attack-training views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image


POLICY_SOURCE_RESOLUTION: Final[int] = 512
POLICY_PRE_CROP_RESOLUTION: Final[int] = 224
DEPLOYMENT_CROP_AREA: Final[float] = 0.9


@dataclass(frozen=True)
class DeploymentViewSpecification:
    """Policy-source, pre-crop, and center-crop geometry."""

    source_resolution: int = POLICY_SOURCE_RESOLUTION
    pre_crop_resolution: int = POLICY_PRE_CROP_RESOLUTION
    crop_area: float = DEPLOYMENT_CROP_AREA

    def __post_init__(self) -> None:
        if self.source_resolution <= 0 or self.pre_crop_resolution <= 0:
            raise ValueError("deployment view resolutions must be positive")
        if not math.isfinite(self.crop_area) or not 0.0 < self.crop_area <= 1.0:
            raise ValueError("deployment crop area must be in (0, 1]")


DEFAULT_DEPLOYMENT_VIEW = DeploymentViewSpecification()


def _validate_uint8_rgb(image: np.ndarray, resolution: int) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("deployment image must be a uint8 numpy array")
    if image.shape != (resolution, resolution, 3):
        raise ValueError(
            "deployment image shape does not match specification: "
            f"{image.shape} != {(resolution, resolution, 3)}"
        )


def _resize_uint8_rgb(image: np.ndarray, resolution: int) -> np.ndarray:
    resized = Image.fromarray(np.ascontiguousarray(image)).resize(
        (resolution, resolution), resample=Image.Resampling.BICUBIC
    )
    return np.array(resized, dtype=np.uint8, copy=True, order="C")


def resize_policy_pre_crop_canvas(
    source_rgb: np.ndarray,
    *,
    specification: DeploymentViewSpecification = DEFAULT_DEPLOYMENT_VIEW,
) -> np.ndarray:
    """Create the fixed policy pre-crop canvas with explicit PIL bicubic."""
    _validate_uint8_rgb(source_rgb, specification.source_resolution)
    return _resize_uint8_rgb(source_rgb, specification.pre_crop_resolution)


def deployment_center_crop_uint8(
    pre_crop_rgb: np.ndarray,
    *,
    specification: DeploymentViewSpecification = DEFAULT_DEPLOYMENT_VIEW,
) -> np.ndarray:
    """Apply the exact TensorFlow rollout center crop to a uint8 canvas."""
    _validate_uint8_rgb(pre_crop_rgb, specification.pre_crop_resolution)
    image = tf.image.convert_image_dtype(
        tf.convert_to_tensor(np.ascontiguousarray(pre_crop_rgb)), tf.float32
    )
    side = tf.sqrt(tf.constant(specification.crop_area, dtype=tf.float32))
    offset = (tf.constant(1.0, tf.float32) - side) / 2.0
    boxes = tf.reshape(
        tf.stack((offset, offset, offset + side, offset + side)), (1, 4)
    )
    cropped = tf.image.crop_and_resize(
        image[None, ...],
        boxes,
        tf.constant([0], dtype=tf.int32),
        (specification.pre_crop_resolution, specification.pre_crop_resolution),
        method="bilinear",
    )[0]
    result = tf.image.convert_image_dtype(
        tf.clip_by_value(cropped, 0.0, 1.0), tf.uint8, saturate=True
    )
    return np.array(result.numpy(), dtype=np.uint8, copy=True, order="C")


def build_policy_and_replay_views(
    source_rgb: np.ndarray,
    *,
    replay_resolution: int,
    specification: DeploymentViewSpecification = DEFAULT_DEPLOYMENT_VIEW,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive policy pre-crop and replay frames from one fixed source frame."""
    if replay_resolution <= 0:
        raise ValueError("replay resolution must be positive")
    policy_pre_crop = resize_policy_pre_crop_canvas(
        source_rgb, specification=specification
    )
    replay = _resize_uint8_rgb(source_rgb, replay_resolution)
    return policy_pre_crop, replay


class PolicyViewTransform:
    """Differentiable source→pre-crop→deployment-center-crop transform."""

    def __init__(
        self,
        specification: DeploymentViewSpecification = DEFAULT_DEPLOYMENT_VIEW,
    ) -> None:
        self.specification = specification

    def __call__(self, source_nchw: torch.Tensor) -> torch.Tensor:
        spec = self.specification
        expected = (3, spec.source_resolution, spec.source_resolution)
        if (
            source_nchw.ndim != 4
            or tuple(source_nchw.shape[1:]) != expected
            or not torch.is_floating_point(source_nchw)
        ):
            raise ValueError(
                "policy source must be floating NCHW with tail " f"{expected}"
            )
        pre_crop = F.interpolate(
            source_nchw,
            size=(spec.pre_crop_resolution, spec.pre_crop_resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        side = math.sqrt(spec.crop_area)
        theta = pre_crop.new_zeros((pre_crop.shape[0], 2, 3))
        theta[:, 0, 0] = side
        theta[:, 1, 1] = side
        grid = F.affine_grid(theta, pre_crop.shape, align_corners=True)
        return F.grid_sample(
            pre_crop,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
