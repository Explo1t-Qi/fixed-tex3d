"""Checkpoint-derived differentiable preprocessing for OpenVLA RGB images.

The fused six-channel tensor is a positional contract between the checkpoint's
``timm_model_ids`` and the corresponding image-processor arrays.  This module
reads that contract once and applies it with differentiable PyTorch operators;
it deliberately contains no OpenVLA-specific hard-coded means, standard
deviations, branch ordering, image size, interpolation, or antialias setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


ImageSize = tuple[int, int]
RGBStatistics = tuple[float, float, float]


def _as_sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _as_image_size(value: Any, name: str) -> ImageSize:
    if isinstance(value, int):
        result = (value, value)
    else:
        values = _as_sequence(value, name)
        if len(values) != 2:
            raise ValueError(f"{name} must contain height and width")
        result = (int(values[0]), int(values[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _as_rgb(value: Any, name: str) -> RGBStatistics:
    values = _as_sequence(value, name)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three RGB values")
    result = (float(values[0]), float(values[1]), float(values[2]))
    if not bool(torch.isfinite(torch.tensor(result)).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _interpolation_mode(value: Any) -> str:
    raw = getattr(value, "value", value)
    mapping = {
        2: "bilinear",
        3: "bicubic",
        "bilinear": "bilinear",
        "bicubic": "bicubic",
    }
    key = raw.lower() if isinstance(raw, str) else raw
    if key not in mapping:
        raise ValueError(f"unsupported checkpoint interpolation: {value!r}")
    return mapping[key]


@dataclass(frozen=True)
class VisionBranchPreprocessing:
    """All processor settings for one positional vision branch."""

    model_id: str
    output_size: ImageSize
    interpolation: str
    antialias: bool
    mean: RGBStatistics
    std: RGBStatistics


@dataclass(frozen=True)
class DifferentiableOpenVLAImageProcessor:
    """Map float NCHW RGB to checkpoint-ordered fused pixel values."""

    branches: tuple[VisionBranchPreprocessing, ...]

    @classmethod
    def from_checkpoint(
        cls, *, model: Any, processor: Any
    ) -> "DifferentiableOpenVLAImageProcessor":
        model_ids = tuple(
            str(item)
            for item in _as_sequence(
                getattr(model.config, "timm_model_ids", None),
                "model.config.timm_model_ids",
            )
        )
        image_processor = processor.image_processor
        if getattr(image_processor, "image_resize_strategy", None) != "resize-naive":
            raise ValueError(
                "only checkpoint image_resize_strategy='resize-naive' is supported"
            )

        input_sizes = _as_sequence(image_processor.input_sizes, "input_sizes")
        resize_params = _as_sequence(
            image_processor.tvf_resize_params, "tvf_resize_params"
        )
        crop_params = _as_sequence(
            image_processor.tvf_crop_params, "tvf_crop_params"
        )
        normalize_params = _as_sequence(
            image_processor.tvf_normalize_params, "tvf_normalize_params"
        )
        branch_count = len(model_ids)
        if branch_count == 0 or not all(
            len(values) == branch_count
            for values in (input_sizes, resize_params, crop_params, normalize_params)
        ):
            raise ValueError("checkpoint and processor branch count must match")

        branches: list[VisionBranchPreprocessing] = []
        for index, model_id in enumerate(model_ids):
            input_size = _as_sequence(input_sizes[index], f"input_sizes[{index}]")
            if len(input_size) != 3 or int(input_size[0]) != 3:
                raise ValueError("vision input size must have shape [3, height, width]")
            output_size = (int(input_size[1]), int(input_size[2]))
            resize = resize_params[index]
            crop = crop_params[index]
            normalization = normalize_params[index]
            if not all(isinstance(item, dict) for item in (resize, crop, normalization)):
                raise ValueError("processor transform parameters must be dictionaries")
            resize_size = _as_image_size(
                resize.get("size"), f"tvf_resize_params[{index}].size"
            )
            crop_size = _as_image_size(
                crop.get("output_size"), f"tvf_crop_params[{index}].output_size"
            )
            if resize_size != output_size or crop_size != output_size:
                raise ValueError(
                    "resize-naive resize, crop, and input sizes must match"
                )
            mean = _as_rgb(
                normalization.get("mean"),
                f"tvf_normalize_params[{index}].mean",
            )
            std = _as_rgb(
                normalization.get("std"),
                f"tvf_normalize_params[{index}].std",
            )
            if any(value <= 0.0 for value in std):
                raise ValueError("normalization standard deviations must be positive")
            antialias = resize.get("antialias")
            if not isinstance(antialias, bool):
                raise ValueError("checkpoint antialias setting must be boolean")
            branches.append(
                VisionBranchPreprocessing(
                    model_id=model_id,
                    output_size=output_size,
                    interpolation=_interpolation_mode(resize.get("interpolation")),
                    antialias=antialias,
                    mean=mean,
                    std=std,
                )
            )

        output_sizes = {branch.output_size for branch in branches}
        if len(output_sizes) != 1:
            raise ValueError("fused OpenVLA branches must have one output size")
        return cls(branches=tuple(branches))

    @property
    def branch_model_ids(self) -> tuple[str, ...]:
        return tuple(branch.model_id for branch in self.branches)

    @property
    def output_size(self) -> ImageSize:
        return self.branches[0].output_size

    def _transform_branch(
        self, rgb_images: torch.Tensor, branch: VisionBranchPreprocessing
    ) -> torch.Tensor:
        resized = rgb_images
        if tuple(rgb_images.shape[-2:]) != branch.output_size:
            resized = F.interpolate(
                rgb_images,
                size=branch.output_size,
                mode=branch.interpolation,
                align_corners=False,
                antialias=branch.antialias,
            )
        mean = resized.new_tensor(branch.mean).view(1, 3, 1, 1)
        std = resized.new_tensor(branch.std).view(1, 3, 1, 1)
        return (resized - mean) / std

    def __call__(self, rgb_images: torch.Tensor) -> torch.Tensor:
        if rgb_images.ndim != 4 or rgb_images.shape[1] != 3:
            raise ValueError("OpenVLA RGB input must have shape [batch, 3, H, W]")
        if not torch.is_floating_point(rgb_images):
            raise ValueError("OpenVLA RGB input must be a floating-point tensor")
        return torch.cat(
            tuple(
                self._transform_branch(rgb_images, branch)
                for branch in self.branches
            ),
            dim=1,
        )
