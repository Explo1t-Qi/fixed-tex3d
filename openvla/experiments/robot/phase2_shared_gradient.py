"""Phase 2.3 contracts for differentiable dual-VLA shared features.

This module contains only the live wiring.  The frozen PCA+CCA mapping and the
shared-feature objective remain owned by ``shared-feature-tex3d`` and are
injected as callables/modules at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PI05_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
O2_SHAPE = (256, 4096)
P2_SHAPE = (256, 2048)
SHARED_SHAPE = (256, 262)


class Phase2GradientClosureError(RuntimeError):
    """Raised at the earliest invalid stage of the Phase 2.3 graph."""


def _tree_to_batched_torch(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _tree_to_batched_torch(item, device=device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_tree_to_batched_torch(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_batched_torch(item, device=device) for item in value)
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise Phase2GradientClosureError("PI0Pytorch transformed input contains objects")
    return torch.from_numpy(np.array(array, copy=True)).to(device).unsqueeze(0)


def _validate_finite_feature(
    value: torch.Tensor,
    *,
    name: str,
    expected_tail: tuple[int, int],
) -> None:
    if not isinstance(value, torch.Tensor):
        raise Phase2GradientClosureError(f"{name} is not a torch.Tensor")
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_tail:
        raise Phase2GradientClosureError(
            f"{name} must have shape [B,{expected_tail[0]},{expected_tail[1]}], "
            f"got {tuple(value.shape)}"
        )
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise Phase2GradientClosureError(f"{name} must be finite floating point")


class Pi05BaseImageAdapter:
    """Inject one differentiable base image into frozen PI0Pytorch inputs.

    ``policy_input`` must already contain the official client-preprocessed
    (oriented, resize-with-pad, uint8) base and wrist images.  The policy's
    frozen input transform is evaluated once for those fixed inputs.  Live
    source images are then adapted with the same differentiable replacement
    used by the existing ``pi/attack_pi0.py`` image-feature attack and replace
    only ``base_0_rgb``.  Wrist, padded image, masks, state, and language remain
    byte/tensor-identical to the frozen template.
    """

    def __init__(
        self,
        *,
        policy: Any,
        observation_type: Any,
        policy_input: Mapping[str, Any],
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        transform = getattr(policy, "_input_transform", None)
        if getattr(policy, "_is_pytorch_model", None) is not True or not callable(
            transform
        ):
            raise Phase2GradientClosureError(
                "policy must expose the current PI0Pytorch input transform"
            )
        try:
            transformed = transform(dict(policy_input))
            batched = _tree_to_batched_torch(
                transformed, device=self.device
            )
        except Phase2GradientClosureError:
            raise
        except Exception as error:
            raise Phase2GradientClosureError(
                "PI0Pytorch fixed-input preprocessing failed"
            ) from error
        self._observation_type = observation_type
        template_observation = self._from_dict(batched)
        self._template = self._observation_values(template_observation)
        self._validate_template()

    @staticmethod
    def _observation_values(observation: Any) -> dict[str, Any]:
        values = {
            "image": dict(observation.images),
            "image_mask": dict(observation.image_masks),
            "state": observation.state,
        }
        optional = (
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        )
        for name in optional:
            value = getattr(observation, name, None)
            if value is not None:
                values[name] = value
        return values

    def _validate_template(self) -> None:
        images = self._template.get("image")
        masks = self._template.get("image_mask")
        if not isinstance(images, Mapping) or tuple(images) != PI05_IMAGE_KEYS:
            raise Phase2GradientClosureError(
                "PI0Pytorch image-slot ordering must be base/left-wrist/right-wrist"
            )
        if not isinstance(masks, Mapping) or tuple(masks) != PI05_IMAGE_KEYS:
            raise Phase2GradientClosureError(
                "PI0Pytorch image-mask ordering does not match image slots"
            )
        expected_masks = (True, True, False)
        for key, expected_mask in zip(PI05_IMAGE_KEYS, expected_masks, strict=True):
            image = images[key]
            mask = masks[key]
            if not isinstance(image, torch.Tensor) or tuple(image.shape) != (
                1,
                3,
                224,
                224,
            ):
                raise Phase2GradientClosureError(
                    f"PI0Pytorch {key} template must be [1,3,224,224]"
                )
            if image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
                raise Phase2GradientClosureError(
                    f"PI0Pytorch {key} template must be finite float32"
                )
            if tuple(mask.shape) != (1,) or mask.dtype != torch.bool:
                raise Phase2GradientClosureError(
                    f"PI0Pytorch {key} mask must be bool [1]"
                )
            if bool(mask.item()) is not expected_mask:
                raise Phase2GradientClosureError(
                    f"PI0Pytorch {key} mask violates pi05_libero semantics"
                )
        state = self._template.get("state")
        tokens = self._template.get("tokenized_prompt")
        token_masks = self._template.get("tokenized_prompt_mask")
        if not isinstance(state, torch.Tensor) or tuple(state.shape) != (1, 32):
            raise Phase2GradientClosureError(
                "PI0Pytorch state template must be [1,32]"
            )
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (1, 200):
            raise Phase2GradientClosureError(
                "PI0Pytorch token template must be [1,200]"
            )
        if not isinstance(token_masks, torch.Tensor) or tuple(
            token_masks.shape
        ) != (1, 200):
            raise Phase2GradientClosureError(
                "PI0Pytorch token-mask template must be [1,200]"
            )

    @property
    def clean_observation(self) -> Any:
        """Return a fresh model observation containing the frozen clean inputs."""

        return self._from_dict(self._copy_template())

    def observation_for_base_image(self, source_rgb: torch.Tensor) -> Any:
        """Replace only ``base_0_rgb`` while preserving source-image autograd."""

        if (
            not isinstance(source_rgb, torch.Tensor)
            or source_rgb.ndim != 4
            or source_rgb.shape[0] != 1
            or source_rgb.shape[1] != 3
            or not source_rgb.is_floating_point()
        ):
            raise Phase2GradientClosureError(
                "PI0Pytorch source image must be floating [1,3,H,W]"
            )
        if not bool(torch.isfinite(source_rgb).all()):
            raise Phase2GradientClosureError("PI0Pytorch source image is non-finite")

        template_base = self._template["image"]["base_0_rgb"]
        live = source_rgb.to(self.device)
        if tuple(live.shape[-2:]) != tuple(template_base.shape[-2:]):
            live = F.interpolate(
                live,
                size=tuple(template_base.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        with torch.no_grad():
            minimum = float(template_base.min().item())
            maximum = float(template_base.max().item())
        if maximum > 2.0:
            live = live * 255.0
        elif minimum < -0.5:
            live = live * 2.0 - 1.0
        live = live.to(dtype=template_base.dtype)

        inputs = self._copy_template()
        images = dict(self._template["image"])
        images["base_0_rgb"] = live
        inputs["image"] = images
        return self._from_dict(inputs)

    def _copy_template(self) -> dict[str, Any]:
        values = dict(self._template)
        values["image"] = dict(self._template["image"])
        values["image_mask"] = dict(self._template["image_mask"])
        return values

    def _from_dict(self, values: Mapping[str, Any]) -> Any:
        try:
            observation = self._observation_type.from_dict(values)
        except Exception as error:
            raise Phase2GradientClosureError(
                "PI0Pytorch Observation construction failed"
            ) from error
        images = getattr(observation, "images", None)
        masks = getattr(observation, "image_masks", None)
        if not isinstance(images, Mapping) or tuple(images) != PI05_IMAGE_KEYS:
            raise Phase2GradientClosureError(
                "PI0Pytorch Observation changed image-slot ordering"
            )
        if not isinstance(masks, Mapping) or tuple(masks) != PI05_IMAGE_KEYS:
            raise Phase2GradientClosureError(
                "PI0Pytorch Observation changed image-mask ordering"
            )
        return observation


def extract_pi05_p2_from_preprocessed_base_image(
    model: Any, base_image: torch.Tensor
) -> torch.Tensor:
    """Run the live P2 implementation on an already-preprocessed base image."""

    try:
        p2 = model.paligemma_with_expert.embed_image(base_image)
    except Exception as error:
        raise Phase2GradientClosureError(
            "PI0Pytorch base_0_rgb image-to-P2 forward failed"
        ) from error
    _validate_finite_feature(p2, name="PI0Pytorch P2", expected_tail=P2_SHAPE)
    return p2


def extract_pi05_p2_autograd(model: Any, observation: Any) -> torch.Tensor:
    """Return live base-camera P2 without no-grad, detach, pooling, or reordering."""

    images_by_key = getattr(observation, "images", None)
    if not isinstance(images_by_key, Mapping) or tuple(images_by_key) != PI05_IMAGE_KEYS:
        raise Phase2GradientClosureError(
            "PI0Pytorch input image slots violate the frozen ordering"
        )
    try:
        prepared = model._preprocess_observation(observation, train=False)
    except Exception as error:
        raise Phase2GradientClosureError(
            "PI0Pytorch observation preprocessing failed"
        ) from error
    if not isinstance(prepared, tuple) or len(prepared) != 5:
        raise Phase2GradientClosureError(
            "PI0Pytorch preprocessing must return five values"
        )
    images, image_masks, _, _, _ = prepared
    if len(images) != len(PI05_IMAGE_KEYS) or len(image_masks) != len(
        PI05_IMAGE_KEYS
    ):
        raise Phase2GradientClosureError(
            "PI0Pytorch preprocessing changed the three image slots"
        )
    expected_masks = (True, True, False)
    for key, image, mask, expected_mask in zip(
        PI05_IMAGE_KEYS, images, image_masks, expected_masks, strict=True
    ):
        if tuple(image.shape[1:]) != (3, 224, 224):
            raise Phase2GradientClosureError(
                f"PI0Pytorch preprocessing changed {key} image layout"
            )
        expected = torch.full_like(mask, expected_mask, dtype=torch.bool)
        if mask.dtype != torch.bool or not torch.equal(mask, expected):
            raise Phase2GradientClosureError(
                f"PI0Pytorch preprocessing changed {key} mask semantics"
            )
    return extract_pi05_p2_from_preprocessed_base_image(model, images[0])


@dataclass(frozen=True)
class DualSharedFeatures:
    o2: torch.Tensor
    p2: torch.Tensor
    h_o: torch.Tensor
    h_p: torch.Tensor


@dataclass(frozen=True)
class CleanSharedReference:
    h_o: torch.Tensor
    h_p: torch.Tensor


class DualVLAFeatureAdapter:
    """Run one base RGB tensor through both frozen live feature branches."""

    def __init__(
        self,
        *,
        openvla_path: Callable[[torch.Tensor], torch.Tensor],
        pi05_path: Callable[[torch.Tensor], torch.Tensor],
        openvla_mapping: Any,
        pi05_mapping: Any,
        shared_loss: Callable[..., Any],
        openvla_device: torch.device | str,
        pi05_device: torch.device | str,
        loss_device: torch.device | str,
    ) -> None:
        self.openvla_path = openvla_path
        self.pi05_path = pi05_path
        self.openvla_mapping = openvla_mapping
        self.pi05_mapping = pi05_mapping
        self.shared_loss = shared_loss
        self.openvla_device = torch.device(openvla_device)
        self.pi05_device = torch.device(pi05_device)
        self.loss_device = torch.device(loss_device)

    def features(self, source_rgb: torch.Tensor) -> DualSharedFeatures:
        self._validate_source(source_rgb)
        try:
            o2 = self.openvla_path(source_rgb.to(self.openvla_device))
        except Phase2GradientClosureError:
            raise
        except Exception as error:
            raise Phase2GradientClosureError("OpenVLA image-to-O2 failed") from error
        _validate_finite_feature(o2, name="OpenVLA O2", expected_tail=O2_SHAPE)
        try:
            p2 = self.pi05_path(source_rgb.to(self.pi05_device))
        except Phase2GradientClosureError:
            raise
        except Exception as error:
            raise Phase2GradientClosureError("PI0Pytorch image-to-P2 failed") from error
        _validate_finite_feature(p2, name="PI0Pytorch P2", expected_tail=P2_SHAPE)

        try:
            h_o = self.openvla_mapping.map_o2(o2)
        except Exception as error:
            raise Phase2GradientClosureError("OpenVLA CCA mapping failed") from error
        try:
            h_p_native = self.pi05_mapping.map_p2(p2)
        except Exception as error:
            raise Phase2GradientClosureError("PI0Pytorch CCA mapping failed") from error
        _validate_finite_feature(h_o, name="OpenVLA H_O", expected_tail=SHARED_SHAPE)
        _validate_finite_feature(
            h_p_native, name="PI0Pytorch H_P", expected_tail=SHARED_SHAPE
        )
        h_o = h_o.to(self.loss_device)
        h_p = h_p_native.to(self.loss_device)
        if h_o.dtype != h_p.dtype:
            raise Phase2GradientClosureError(
                "canonical branches must use one dtype on the loss device"
            )
        return DualSharedFeatures(o2=o2, p2=p2, h_o=h_o, h_p=h_p)

    def clean_reference(self, source_rgb: torch.Tensor) -> CleanSharedReference:
        with torch.no_grad():
            features = self.features(source_rgb)
        return CleanSharedReference(
            h_o=features.h_o.detach(), h_p=features.h_p.detach()
        )

    def loss(self, source_rgb: torch.Tensor, clean: CleanSharedReference) -> tuple[Any, DualSharedFeatures]:
        features = self.features(source_rgb)
        try:
            result = self.shared_loss(
                features.h_o,
                features.h_p,
                clean.h_o,
                clean.h_p,
            )
        except Exception as error:
            raise Phase2GradientClosureError("shared-feature loss failed") from error
        if result.loss.ndim != 0 or not bool(torch.isfinite(result.loss)):
            raise Phase2GradientClosureError("shared-feature loss is not finite scalar")
        return result, features

    @staticmethod
    def _validate_source(source_rgb: torch.Tensor) -> None:
        if (
            not isinstance(source_rgb, torch.Tensor)
            or source_rgb.ndim != 4
            or tuple(source_rgb.shape[1:]) != (3, 512, 512)
            or not source_rgb.is_floating_point()
        ):
            raise Phase2GradientClosureError(
                "canonical source image must be floating [B,3,512,512]"
            )
        if source_rgb.shape[0] != 1:
            raise Phase2GradientClosureError("Phase 2.3 smoke requires batch size 1")
        if not bool(torch.isfinite(source_rgb).all()):
            raise Phase2GradientClosureError("canonical source image is non-finite")


def checked_gradient(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    label: str,
    retain_graph: bool,
) -> torch.Tensor:
    """Return a finite, nonzero diagnostic gradient or fail at its named gate."""

    gradient = torch.autograd.grad(
        output,
        input_tensor,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if gradient is None:
        raise Phase2GradientClosureError(f"{label} gradient is missing")
    if not bool(torch.isfinite(gradient).all()):
        raise Phase2GradientClosureError(f"{label} gradient is non-finite")
    if not bool(torch.any(gradient != 0)):
        raise Phase2GradientClosureError(f"{label} gradient is zero")
    return gradient


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return a stable diagnostic cosine between two image gradients."""

    return float(
        F.cosine_similarity(first.flatten(), second.flatten(), dim=0, eps=1e-12)
        .detach()
        .item()
    )
