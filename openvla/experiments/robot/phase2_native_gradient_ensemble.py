"""Native O2/P2 model-level gradient ensemble for Phase 2 experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from phase2_shared_gradient import (
    O2_SHAPE,
    P2_SHAPE,
    Phase2GradientClosureError,
    _validate_finite_feature,
)
from phase2_shared_optimization import sign_pgd_update, validate_texture_budget


NORMALIZATION_EPS = 1e-12


class NativeGradientEnsembleError(RuntimeError):
    """Raised at the earliest invalid native-gradient ensemble stage."""


@dataclass(frozen=True)
class NativeFeatures:
    o2: torch.Tensor
    p2: torch.Tensor


@dataclass(frozen=True)
class FrozenNativeReference:
    o2: torch.Tensor
    p2: torch.Tensor

    @classmethod
    def from_tensors(
        cls, o2: torch.Tensor, p2: torch.Tensor
    ) -> "FrozenNativeReference":
        _validate_finite_feature(o2, name="clean OpenVLA O2", expected_tail=O2_SHAPE)
        _validate_finite_feature(p2, name="clean PI0Pytorch P2", expected_tail=P2_SHAPE)
        if o2.shape[0] != p2.shape[0]:
            raise NativeGradientEnsembleError(
                "clean O2/P2 references must use the same batch size"
            )
        return cls(o2=o2.detach().clone(), p2=p2.detach().clone())


@dataclass(frozen=True)
class NativeDisplacementLosses:
    loss_o: torch.Tensor
    loss_p: torch.Tensor
    o2_mse: torch.Tensor
    p2_mse: torch.Tensor


class DualVLANativeFeatureAdapter:
    """Run one source image through native O2 and P2 without PCA or CCA."""

    def __init__(
        self,
        *,
        openvla_path: Callable[[torch.Tensor], torch.Tensor],
        pi05_path: Callable[[torch.Tensor], torch.Tensor],
        openvla_device: torch.device | str,
        pi05_device: torch.device | str,
    ) -> None:
        self.openvla_path = openvla_path
        self.pi05_path = pi05_path
        self.openvla_device = torch.device(openvla_device)
        self.pi05_device = torch.device(pi05_device)

    def features(self, source_rgb: torch.Tensor) -> NativeFeatures:
        self._validate_source(source_rgb)
        try:
            o2 = self.openvla_path(source_rgb.to(self.openvla_device))
        except Phase2GradientClosureError:
            raise
        except Exception as error:
            raise NativeGradientEnsembleError("OpenVLA image-to-O2 failed") from error
        _validate_finite_feature(o2, name="OpenVLA O2", expected_tail=O2_SHAPE)
        try:
            p2 = self.pi05_path(source_rgb.to(self.pi05_device))
        except Phase2GradientClosureError:
            raise
        except Exception as error:
            raise NativeGradientEnsembleError(
                "PI0Pytorch image-to-P2 failed"
            ) from error
        _validate_finite_feature(p2, name="PI0Pytorch P2", expected_tail=P2_SHAPE)
        if o2.shape[0] != p2.shape[0]:
            raise NativeGradientEnsembleError(
                "live O2/P2 features must use the same batch size"
            )
        return NativeFeatures(o2=o2, p2=p2)

    def clean_reference(self, source_rgb: torch.Tensor) -> FrozenNativeReference:
        with torch.no_grad():
            features = self.features(source_rgb)
        return FrozenNativeReference.from_tensors(features.o2, features.p2)

    def losses(
        self, source_rgb: torch.Tensor, clean: FrozenNativeReference
    ) -> tuple[NativeDisplacementLosses, NativeFeatures]:
        features = self.features(source_rgb)
        losses = native_displacement_losses(features, clean)
        return losses, features

    @staticmethod
    def _validate_source(source_rgb: torch.Tensor) -> None:
        if (
            not isinstance(source_rgb, torch.Tensor)
            or source_rgb.ndim != 4
            or tuple(source_rgb.shape[1:]) != (3, 512, 512)
            or source_rgb.shape[0] != 1
            or not source_rgb.is_floating_point()
        ):
            raise NativeGradientEnsembleError(
                "canonical source image must be floating [1,3,512,512]"
            )
        if not bool(torch.isfinite(source_rgb).all()):
            raise NativeGradientEnsembleError("canonical source image is non-finite")


def native_displacement_losses(
    adversarial: NativeFeatures,
    clean: FrozenNativeReference,
) -> NativeDisplacementLosses:
    """Return independent negative native-feature MSE objectives."""

    for name, adv, reference, expected_tail in (
        ("O2", adversarial.o2, clean.o2, O2_SHAPE),
        ("P2", adversarial.p2, clean.p2, P2_SHAPE),
    ):
        _validate_finite_feature(
            adv, name=f"adversarial {name}", expected_tail=expected_tail
        )
        _validate_finite_feature(
            reference, name=f"clean {name}", expected_tail=expected_tail
        )
        if reference.requires_grad:
            raise NativeGradientEnsembleError(f"clean {name} must be detached")
        if adv.shape != reference.shape or adv.device != reference.device:
            raise NativeGradientEnsembleError(
                f"clean/adversarial {name} shape and device must match"
            )
    o2_mse = (adversarial.o2.float() - clean.o2.float()).square().mean()
    p2_mse = (adversarial.p2.float() - clean.p2.float()).square().mean()
    return NativeDisplacementLosses(
        loss_o=-o2_mse,
        loss_p=-p2_mse,
        o2_mse=o2_mse,
        p2_mse=p2_mse,
    )


@dataclass(frozen=True)
class ModelGradientEnsemble:
    g_o: torch.Tensor
    g_p: torch.Tensor
    g_o_normalized: torch.Tensor
    g_p_normalized: torch.Tensor
    gradient: torch.Tensor
    diagnostics: dict[str, float]


def _validate_gradient(gradient: torch.Tensor | None, *, label: str) -> torch.Tensor:
    if gradient is None:
        raise NativeGradientEnsembleError(f"{label} gradient is missing")
    if not isinstance(gradient, torch.Tensor) or not gradient.is_floating_point():
        raise NativeGradientEnsembleError(f"{label} gradient must be floating point")
    if not bool(torch.isfinite(gradient).all()):
        raise NativeGradientEnsembleError(f"{label} gradient is non-finite")
    return gradient


def _gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    cosine = F.cosine_similarity(
        first.reshape(1, -1), second.reshape(1, -1), dim=1, eps=NORMALIZATION_EPS
    )
    value = float(cosine.detach().item())
    if not np.isfinite(value):
        raise NativeGradientEnsembleError("model gradient cosine is non-finite")
    return value


class ModelGradientAccumulator:
    """Accumulate raw frame gradients separately, then normalize by model."""

    def __init__(self) -> None:
        self._g_o_sum: torch.Tensor | None = None
        self._g_p_sum: torch.Tensor | None = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def add(self, g_o_frame: torch.Tensor, g_p_frame: torch.Tensor) -> None:
        g_o_frame = _validate_gradient(g_o_frame, label="OpenVLA frame")
        g_p_frame = _validate_gradient(g_p_frame, label="PI0Pytorch frame")
        if (
            g_o_frame.shape != g_p_frame.shape
            or g_o_frame.device != g_p_frame.device
            or g_o_frame.dtype != g_p_frame.dtype
        ):
            raise NativeGradientEnsembleError(
                "O2/P2 frame gradients must match the texture parameter layout"
            )
        if self._g_o_sum is None:
            self._g_o_sum = g_o_frame.detach().clone()
            self._g_p_sum = g_p_frame.detach().clone()
        else:
            if (
                g_o_frame.shape != self._g_o_sum.shape
                or g_o_frame.device != self._g_o_sum.device
                or g_o_frame.dtype != self._g_o_sum.dtype
            ):
                raise NativeGradientEnsembleError(
                    "frame gradients changed layout during accumulation"
                )
            self._g_o_sum.add_(g_o_frame.detach())
            assert self._g_p_sum is not None
            self._g_p_sum.add_(g_p_frame.detach())
        self._count += 1

    def finalize(self) -> ModelGradientEnsemble:
        if self._count == 0 or self._g_o_sum is None or self._g_p_sum is None:
            raise NativeGradientEnsembleError("model gradient batch is empty")
        g_o = self._g_o_sum / self._count
        g_p = self._g_p_sum / self._count
        for label, gradient in (("OpenVLA batch", g_o), ("PI0Pytorch batch", g_p)):
            _validate_gradient(gradient, label=label)
            if not bool(torch.any(gradient != 0)):
                raise NativeGradientEnsembleError(f"{label} gradient is zero")

        g_o_mean_abs = g_o.abs().mean()
        g_p_mean_abs = g_p.abs().mean()
        g_o_normalized = g_o / (g_o_mean_abs + NORMALIZATION_EPS)
        g_p_normalized = g_p / (g_p_mean_abs + NORMALIZATION_EPS)
        gradient = (g_o_normalized + g_p_normalized) / 2.0
        _validate_gradient(g_o_normalized, label="normalized OpenVLA batch")
        _validate_gradient(g_p_normalized, label="normalized PI0Pytorch batch")
        _validate_gradient(gradient, label="gradient ensemble")
        if not bool(torch.any(gradient != 0)):
            raise NativeGradientEnsembleError("gradient ensemble is zero")

        diagnostics = {
            "g_o_l2_norm": float(g_o.norm().item()),
            "g_p_l2_norm": float(g_p.norm().item()),
            "g_o_mean_abs": float(g_o_mean_abs.item()),
            "g_p_mean_abs": float(g_p_mean_abs.item()),
            "raw_gradient_cosine": _gradient_cosine(g_o, g_p),
            "g_o_normalized_l2_norm": float(g_o_normalized.norm().item()),
            "g_p_normalized_l2_norm": float(g_p_normalized.norm().item()),
            "normalized_gradient_cosine": _gradient_cosine(
                g_o_normalized, g_p_normalized
            ),
            "gradient_ensemble_l2_norm": float(gradient.norm().item()),
        }
        if not all(np.isfinite(value) for value in diagnostics.values()):
            raise NativeGradientEnsembleError(
                "gradient ensemble diagnostics are non-finite"
            )
        return ModelGradientEnsemble(
            g_o=g_o,
            g_p=g_p,
            g_o_normalized=g_o_normalized,
            g_p_normalized=g_p_normalized,
            gradient=gradient,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class NativeTrainingFrame:
    frame_id: str
    state_id: int
    pi05_template_id: str
    clean: FrozenNativeReference
    payload: Any


def _scalar(value: torch.Tensor, *, label: str) -> float:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise NativeGradientEnsembleError(f"{label} must be a scalar tensor")
    scalar = float(value.detach().item())
    if not np.isfinite(scalar):
        raise NativeGradientEnsembleError(f"{label} is non-finite")
    return scalar


def _model_gradient(
    loss: torch.Tensor,
    parameter: torch.Tensor,
    *,
    label: str,
    retain_graph: bool,
) -> torch.Tensor:
    _scalar(loss, label=f"{label} loss")
    gradient = torch.autograd.grad(
        loss,
        parameter,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )[0]
    return _validate_gradient(gradient, label=label)


def train_native_gradient_ensemble(
    *,
    renderer: Any,
    frames: Sequence[NativeTrainingFrame],
    forward_frame: Callable[
        [NativeTrainingFrame], tuple[NativeDisplacementLosses, torch.Tensor]
    ],
    iterations: int,
    requested_batch_size: int,
    pgd_step: float,
    seed: int,
    metrics_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate by frame within each model, then make one ensemble update."""

    if iterations < 1 or requested_batch_size < 1 or not frames:
        raise NativeGradientEnsembleError("invalid training loop dimensions")
    frame_ids = [frame.frame_id for frame in frames]
    template_ids = [frame.pi05_template_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise NativeGradientEnsembleError("frame IDs must be unique")
    if len(template_ids) != len(set(template_ids)):
        raise NativeGradientEnsembleError(
            "each training frame requires its own PI0Pytorch template"
        )
    parameter = renderer.adv_noise
    if not isinstance(parameter, torch.Tensor) or not parameter.requires_grad:
        raise NativeGradientEnsembleError("renderer adv_noise must be trainable")
    batch_size = min(requested_batch_size, len(frames))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")

    for iteration in range(iterations):
        parameter.grad = None
        chosen = rng.choice(len(frames), size=batch_size, replace=False)
        accumulator = ModelGradientAccumulator()
        o2_mse = 0.0
        p2_mse = 0.0
        for index in chosen:
            frame = frames[int(index)]
            losses, adversarial_image = forward_frame(frame)
            if (
                not isinstance(adversarial_image, torch.Tensor)
                or not adversarial_image.is_floating_point()
                or not bool(torch.isfinite(adversarial_image).all())
            ):
                raise NativeGradientEnsembleError(
                    "adversarial image must be finite floating point"
                )
            g_o_frame = _model_gradient(
                losses.loss_o,
                parameter,
                label="OpenVLA frame",
                retain_graph=True,
            )
            g_p_frame = _model_gradient(
                losses.loss_p,
                parameter,
                label="PI0Pytorch frame",
                retain_graph=False,
            )
            accumulator.add(g_o_frame, g_p_frame)
            o2_mse += _scalar(losses.o2_mse, label="O2 MSE") / batch_size
            p2_mse += _scalar(losses.p2_mse, label="P2 MSE") / batch_size
            del losses, adversarial_image, g_o_frame, g_p_frame

        ensemble = accumulator.finalize()
        parameter.grad = ensemble.gradient.detach().clone()
        _, change_linf = sign_pgd_update(parameter, step_size=pgd_step)
        maximum_perturbation = validate_texture_budget(renderer)
        row = {
            "iteration": iteration,
            "selected_frame_ids": [frames[int(i)].frame_id for i in chosen],
            "loss_o": -o2_mse,
            "loss_p": -p2_mse,
            "o2_mse": o2_mse,
            "p2_mse": p2_mse,
            **ensemble.diagnostics,
            "texture_gradient_norm": ensemble.diagnostics["gradient_ensemble_l2_norm"],
            "parameter_change_linf": change_linf,
            "maximum_texture_perturbation": maximum_perturbation,
            "texture_budget_respected": maximum_perturbation
            <= float(renderer.epsilon) + 1e-6,
        }
        history.append(row)
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
    return history
