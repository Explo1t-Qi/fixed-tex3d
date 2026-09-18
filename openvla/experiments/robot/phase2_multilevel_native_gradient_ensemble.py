"""O1-S/O2 and P1/P2 hierarchical native gradient ensemble."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase2_native_gradient_ensemble import (
    ModelGradientAccumulator,
    NativeGradientEnsembleError,
)
from phase2_shared_gradient import (
    O2_SHAPE,
    P2_SHAPE,
    Phase2GradientClosureError,
    _validate_finite_feature,
    extract_pi05_p2_autograd,
)
from phase2_shared_optimization import sign_pgd_update, validate_texture_budget
from step1_o2_p2 import extract_openvla_o2


O1_SHAPE = (256, 1152)
P1_SHAPE = (256, 1152)
RATIO_EPS = 1e-12


def extract_openvla_o1_o2_autograd(
    model: Any, pixel_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return O1-S and O2 from one differentiable OpenVLA vision forward."""

    vision_backbone = getattr(model, "vision_backbone", None)
    fused_featurizer = getattr(vision_backbone, "fused_featurizer", None)
    register_hook = getattr(fused_featurizer, "register_forward_hook", None)
    if not callable(register_hook):
        raise NativeGradientEnsembleError(
            "OpenVLA fused SigLIP featurizer must support forward hooks"
        )
    captures: list[torch.Tensor] = []

    def capture_o1(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        captures.append(output)

    handle = register_hook(capture_o1)
    try:
        o2 = extract_openvla_o2(model, pixel_values)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise NativeGradientEnsembleError(
            "OpenVLA O1-S hook must capture exactly one feature tensor"
        )
    o1 = captures[0]
    _validate_finite_feature(o1, name="OpenVLA O1-S", expected_tail=O1_SHAPE)
    _validate_finite_feature(o2, name="OpenVLA O2", expected_tail=O2_SHAPE)
    if o1.shape[0] != o2.shape[0]:
        raise NativeGradientEnsembleError("OpenVLA O1-S/O2 batch sizes differ")
    return o1, o2


def extract_pi05_p1_p2_autograd(
    model: Any, observation: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return projector-input P1 and P2 from one differentiable embed call."""

    try:
        projector = model.paligemma_with_expert.paligemma.model.multi_modal_projector
    except AttributeError as error:
        raise NativeGradientEnsembleError(
            "PI0Pytorch model does not expose the PaliGemma multimodal projector"
        ) from error
    register_hook = getattr(projector, "register_forward_pre_hook", None)
    if not callable(register_hook):
        raise NativeGradientEnsembleError(
            "PI0Pytorch multimodal projector must support forward pre-hooks"
        )
    captures: list[torch.Tensor] = []

    def capture_p1(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1:
            raise NativeGradientEnsembleError(
                "PI0Pytorch multimodal projector received unexpected arguments"
            )
        captures.append(inputs[0])

    handle = register_hook(capture_p1)
    try:
        p2 = extract_pi05_p2_autograd(model, observation)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise NativeGradientEnsembleError(
            "PI0Pytorch P1 hook must capture exactly one feature tensor"
        )
    p1 = captures[0]
    _validate_finite_feature(p1, name="PI0Pytorch P1", expected_tail=P1_SHAPE)
    _validate_finite_feature(p2, name="PI0Pytorch P2", expected_tail=P2_SHAPE)
    if p1.shape[0] != p2.shape[0]:
        raise NativeGradientEnsembleError("PI0Pytorch P1/P2 batch sizes differ")
    return p1, p2


@dataclass(frozen=True)
class MultiLevelNativeFeatures:
    o1: torch.Tensor
    o2: torch.Tensor
    p1: torch.Tensor
    p2: torch.Tensor


@dataclass(frozen=True)
class FrozenMultiLevelNativeReference:
    o1: torch.Tensor
    o2: torch.Tensor
    p1: torch.Tensor
    p2: torch.Tensor

    @classmethod
    def from_tensors(
        cls,
        o1: torch.Tensor,
        o2: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
    ) -> "FrozenMultiLevelNativeReference":
        values = (
            ("O1-S", o1, O1_SHAPE),
            ("O2", o2, O2_SHAPE),
            ("P1", p1, P1_SHAPE),
            ("P2", p2, P2_SHAPE),
        )
        for name, value, expected_tail in values:
            _validate_finite_feature(
                value, name=f"clean {name}", expected_tail=expected_tail
            )
        if len({value.shape[0] for _, value, _ in values}) != 1:
            raise NativeGradientEnsembleError(
                "clean multi-level references must use the same batch size"
            )
        return cls(*(value.detach().clone() for _, value, _ in values))


@dataclass(frozen=True)
class MultiLevelNativeDisplacementLosses:
    loss_o: torch.Tensor
    loss_p: torch.Tensor
    o1_mse: torch.Tensor
    o2_mse: torch.Tensor
    p1_mse: torch.Tensor
    p2_mse: torch.Tensor
    o1_to_o2_mse_ratio: torch.Tensor
    p1_to_p2_mse_ratio: torch.Tensor


class DualVLAMultiLevelNativeFeatureAdapter:
    """Run one source image through hierarchical O1-S/O2 and P1/P2 paths."""

    def __init__(
        self,
        *,
        openvla_path: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        pi05_path: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        openvla_device: torch.device | str,
        pi05_device: torch.device | str,
    ) -> None:
        self.openvla_path = openvla_path
        self.pi05_path = pi05_path
        self.openvla_device = torch.device(openvla_device)
        self.pi05_device = torch.device(pi05_device)

    def features(self, source_rgb: torch.Tensor) -> MultiLevelNativeFeatures:
        self._validate_source(source_rgb)
        try:
            o1, o2 = self.openvla_path(source_rgb.to(self.openvla_device))
        except (Phase2GradientClosureError, NativeGradientEnsembleError):
            raise
        except Exception as error:
            raise NativeGradientEnsembleError(
                "OpenVLA image-to-O1-S/O2 failed"
            ) from error
        try:
            p1, p2 = self.pi05_path(source_rgb.to(self.pi05_device))
        except (Phase2GradientClosureError, NativeGradientEnsembleError):
            raise
        except Exception as error:
            raise NativeGradientEnsembleError(
                "PI0Pytorch image-to-P1/P2 failed"
            ) from error
        values = (
            ("O1-S", o1, O1_SHAPE),
            ("O2", o2, O2_SHAPE),
            ("P1", p1, P1_SHAPE),
            ("P2", p2, P2_SHAPE),
        )
        for name, value, expected_tail in values:
            _validate_finite_feature(value, name=name, expected_tail=expected_tail)
        if len({value.shape[0] for _, value, _ in values}) != 1:
            raise NativeGradientEnsembleError(
                "live multi-level features must use the same batch size"
            )
        return MultiLevelNativeFeatures(o1=o1, o2=o2, p1=p1, p2=p2)

    def clean_reference(
        self, source_rgb: torch.Tensor
    ) -> FrozenMultiLevelNativeReference:
        with torch.no_grad():
            features = self.features(source_rgb)
        return FrozenMultiLevelNativeReference.from_tensors(
            features.o1, features.o2, features.p1, features.p2
        )

    def losses(
        self, source_rgb: torch.Tensor, clean: FrozenMultiLevelNativeReference
    ) -> tuple[MultiLevelNativeDisplacementLosses, MultiLevelNativeFeatures]:
        features = self.features(source_rgb)
        return multilevel_native_displacement_losses(features, clean), features

    @staticmethod
    def _validate_source(source_rgb: torch.Tensor) -> None:
        if (
            not isinstance(source_rgb, torch.Tensor)
            or source_rgb.ndim != 4
            or tuple(source_rgb.shape) != (1, 3, 512, 512)
            or not source_rgb.is_floating_point()
        ):
            raise NativeGradientEnsembleError(
                "canonical source image must be floating [1,3,512,512]"
            )
        if not bool(torch.isfinite(source_rgb).all()):
            raise NativeGradientEnsembleError("canonical source image is non-finite")


def multilevel_native_displacement_losses(
    adversarial: MultiLevelNativeFeatures,
    clean: FrozenMultiLevelNativeReference,
) -> MultiLevelNativeDisplacementLosses:
    """Return equal-weight hierarchical feature displacement objectives."""

    values = (
        ("O1-S", adversarial.o1, clean.o1, O1_SHAPE),
        ("O2", adversarial.o2, clean.o2, O2_SHAPE),
        ("P1", adversarial.p1, clean.p1, P1_SHAPE),
        ("P2", adversarial.p2, clean.p2, P2_SHAPE),
    )
    for name, adv, reference, expected_tail in values:
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
    o1_mse = (adversarial.o1.float() - clean.o1.float()).square().mean()
    o2_mse = (adversarial.o2.float() - clean.o2.float()).square().mean()
    p1_mse = (adversarial.p1.float() - clean.p1.float()).square().mean()
    p2_mse = (adversarial.p2.float() - clean.p2.float()).square().mean()
    return MultiLevelNativeDisplacementLosses(
        loss_o=-(o1_mse + o2_mse),
        loss_p=-(p1_mse + p2_mse),
        o1_mse=o1_mse,
        o2_mse=o2_mse,
        p1_mse=p1_mse,
        p2_mse=p2_mse,
        o1_to_o2_mse_ratio=o1_mse / (o2_mse + RATIO_EPS),
        p1_to_p2_mse_ratio=p1_mse / (p2_mse + RATIO_EPS),
    )


@dataclass(frozen=True)
class MultiLevelNativeTrainingFrame:
    frame_id: str
    state_id: int
    pi05_template_id: str
    clean: FrozenMultiLevelNativeReference
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
    if gradient is None:
        raise NativeGradientEnsembleError(f"{label} gradient is missing")
    if not gradient.is_floating_point() or not bool(torch.isfinite(gradient).all()):
        raise NativeGradientEnsembleError(f"{label} gradient is non-finite")
    return gradient


def train_multilevel_native_gradient_ensemble(
    *,
    renderer: Any,
    frames: Sequence[MultiLevelNativeTrainingFrame],
    forward_frame: Callable[
        [MultiLevelNativeTrainingFrame],
        tuple[MultiLevelNativeDisplacementLosses, torch.Tensor],
    ],
    iterations: int,
    requested_batch_size: int,
    pgd_step: float,
    seed: int,
    metrics_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Train with frame-first, model-second gradient aggregation."""

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
        metrics = {name: 0.0 for name in ("o1_mse", "o2_mse", "p1_mse", "p2_mse")}
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
            for name in metrics:
                metrics[name] += _scalar(getattr(losses, name), label=name) / batch_size
            del losses, adversarial_image, g_o_frame, g_p_frame

        ensemble = accumulator.finalize()
        parameter.grad = ensemble.gradient.detach().clone()
        _, change_linf = sign_pgd_update(parameter, step_size=pgd_step)
        maximum_perturbation = validate_texture_budget(renderer)
        row = {
            "iteration": iteration,
            "selected_frame_ids": [frames[int(i)].frame_id for i in chosen],
            "loss_o": -(metrics["o1_mse"] + metrics["o2_mse"]),
            "loss_p": -(metrics["p1_mse"] + metrics["p2_mse"]),
            **metrics,
            "o1_to_o2_mse_ratio": metrics["o1_mse"] / (metrics["o2_mse"] + RATIO_EPS),
            "p1_to_p2_mse_ratio": metrics["p1_mse"] / (metrics["p2_mse"] + RATIO_EPS),
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
