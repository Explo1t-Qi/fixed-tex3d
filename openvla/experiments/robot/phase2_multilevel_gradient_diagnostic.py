"""Read-only O1/O2/P1/P2 texture-gradient decomposition diagnostics."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from phase2_multilevel_native_gradient_ensemble import (
    FrozenMultiLevelNativeReference,
    MultiLevelNativeFeatures,
    multilevel_native_displacement_losses,
)


COMPONENTS = ("O1", "O2", "P1", "P2")
DIAGNOSTIC_EPS = 1e-12


class MultiLevelGradientDiagnosticError(RuntimeError):
    """Raised when a gradient-decomposition contract is violated."""


@dataclass(frozen=True)
class ComponentLosses:
    losses: dict[str, torch.Tensor]
    mses: dict[str, torch.Tensor]


@dataclass(frozen=True)
class GradientDiagnosticFrame:
    frame_id: str
    payload: Any


def component_losses(
    adversarial: MultiLevelNativeFeatures,
    clean: FrozenMultiLevelNativeReference,
) -> ComponentLosses:
    """Return the four negative-MSE attack components without normalization."""

    combined = multilevel_native_displacement_losses(adversarial, clean)
    mses = {
        "O1": combined.o1_mse,
        "O2": combined.o2_mse,
        "P1": combined.p1_mse,
        "P2": combined.p2_mse,
    }
    return ComponentLosses(
        losses={name: -value for name, value in mses.items()},
        mses=mses,
    )


def _validate_gradient(
    gradient: torch.Tensor | None,
    *,
    label: str,
    expected_shape: torch.Size | tuple[int, ...] | None = None,
) -> torch.Tensor:
    if gradient is None:
        raise MultiLevelGradientDiagnosticError(f"{label} gradient is missing")
    if not isinstance(gradient, torch.Tensor) or not gradient.is_floating_point():
        raise MultiLevelGradientDiagnosticError(
            f"{label} gradient must be a floating tensor"
        )
    if expected_shape is not None and tuple(gradient.shape) != tuple(expected_shape):
        raise MultiLevelGradientDiagnosticError(
            f"{label} gradient shape mismatch: {tuple(gradient.shape)} != "
            f"{tuple(expected_shape)}"
        )
    if not bool(torch.isfinite(gradient).all()):
        raise MultiLevelGradientDiagnosticError(f"{label} gradient is non-finite")
    return gradient


class ComponentGradientAccumulator:
    """Mean raw frame gradients independently for all four feature levels."""

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def add(self, gradients: dict[str, torch.Tensor | None]) -> None:
        if set(gradients) != set(COMPONENTS):
            raise MultiLevelGradientDiagnosticError(
                f"component gradients must contain exactly {COMPONENTS}"
            )
        validated: dict[str, torch.Tensor] = {}
        reference: torch.Tensor | None = None
        for name in COMPONENTS:
            gradient = _validate_gradient(gradients[name], label=f"{name} frame")
            if reference is None:
                reference = gradient
            elif (
                gradient.shape != reference.shape
                or gradient.device != reference.device
                or gradient.dtype != reference.dtype
            ):
                raise MultiLevelGradientDiagnosticError(
                    "component gradients must share the texture parameter layout"
                )
            validated[name] = gradient

        if not self._sums:
            self._sums = {
                name: gradient.detach().clone() for name, gradient in validated.items()
            }
        else:
            for name, gradient in validated.items():
                expected = self._sums[name]
                if (
                    gradient.shape != expected.shape
                    or gradient.device != expected.device
                    or gradient.dtype != expected.dtype
                ):
                    raise MultiLevelGradientDiagnosticError(
                        "component gradient layout changed across frames"
                    )
                expected.add_(gradient.detach())
        self._count += 1

    def finalize(self) -> dict[str, torch.Tensor]:
        if self._count == 0 or set(self._sums) != set(COMPONENTS):
            raise MultiLevelGradientDiagnosticError("component gradient batch is empty")
        return {name: self._sums[name] / self._count for name in COMPONENTS}


def gradient_magnitude(gradient: torch.Tensor) -> dict[str, float | bool]:
    """Summarize one finite raw batch gradient without changing its scale."""

    gradient = _validate_gradient(gradient, label="component batch")
    absolute = gradient.detach().double().abs()
    zero = not bool(torch.any(absolute != 0))
    return {
        "l2": float(torch.linalg.vector_norm(absolute).item()),
        "mean_abs": float(absolute.mean().item()),
        "linf": float(absolute.max().item()),
        "finite": True,
        "zero_gradient": zero,
    }


def gradient_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float | None:
    """Return raw-gradient cosine, or None when either gradient is zero."""

    first = _validate_gradient(first, label="first cosine")
    second = _validate_gradient(
        second, label="second cosine", expected_shape=first.shape
    )
    first_flat = first.detach().double().reshape(-1)
    second_flat = second.detach().double().reshape(-1)
    first_norm = torch.linalg.vector_norm(first_flat)
    second_norm = torch.linalg.vector_norm(second_flat)
    if float(first_norm.item()) == 0.0 or float(second_norm.item()) == 0.0:
        return None
    value = float(
        torch.dot(first_flat, second_flat).div(first_norm * second_norm).item()
    )
    return max(-1.0, min(1.0, value))


def pairwise_cosines(
    gradients: dict[str, torch.Tensor],
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    if set(gradients) != set(COMPONENTS):
        raise MultiLevelGradientDiagnosticError(
            f"batch gradients must contain exactly {COMPONENTS}"
        )
    matrix: dict[str, dict[str, float | None]] = {}
    for first in COMPONENTS:
        matrix[first] = {
            second: gradient_cosine(gradients[first], gradients[second])
            for second in COMPONENTS
        }
    pairs = {
        f"{first}-{second}": matrix[first][second]
        for index, first in enumerate(COMPONENTS)
        for second in COMPONENTS[index + 1 :]
    }
    return pairs, matrix


def cancellation_ratio(first: torch.Tensor, second: torch.Tensor) -> float:
    """Measure the retained L2 magnitude after adding two raw gradients."""

    first = _validate_gradient(first, label="first cancellation")
    second = _validate_gradient(
        second, label="second cancellation", expected_shape=first.shape
    )
    first64 = first.detach().double()
    second64 = second.detach().double()
    numerator = torch.linalg.vector_norm(first64 + second64)
    denominator = (
        torch.linalg.vector_norm(first64)
        + torch.linalg.vector_norm(second64)
        + DIAGNOSTIC_EPS
    )
    return float((numerator / denominator).item())


def analyze_batch_gradients(
    gradients: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute magnitudes, all cosines, and reconstructed model gradients."""

    components = {name: gradient_magnitude(gradients[name]) for name in COMPONENTS}
    pairs, matrix = pairwise_cosines(gradients)
    openvla = gradients["O1"] + gradients["O2"]
    pi05 = gradients["P1"] + gradients["P2"]
    return {
        "component_gradients": components,
        "pairwise_cosine": pairs,
        "cosine_matrix": matrix,
        "combined": {
            "openvla_gradient_l2": gradient_magnitude(openvla)["l2"],
            "pi05_gradient_l2": gradient_magnitude(pi05)["l2"],
            "combined_model_cosine": gradient_cosine(openvla, pi05),
            "openvla_internal_cancellation_ratio": cancellation_ratio(
                gradients["O1"], gradients["O2"]
            ),
            "pi05_internal_cancellation_ratio": cancellation_ratio(
                gradients["P1"], gradients["P2"]
            ),
        },
    }


def _scalar(value: torch.Tensor, *, label: str) -> float:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise MultiLevelGradientDiagnosticError(f"{label} must be a scalar tensor")
    detached = value.detach()
    if not bool(torch.isfinite(detached)):
        raise MultiLevelGradientDiagnosticError(f"{label} is non-finite")
    return float(detached.item())


def component_texture_gradients(
    losses: ComponentLosses,
    parameter: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiate four losses from one frame graph and then release it."""

    if set(losses.losses) != set(COMPONENTS) or set(losses.mses) != set(COMPONENTS):
        raise MultiLevelGradientDiagnosticError(
            f"component losses must contain exactly {COMPONENTS}"
        )
    gradients: dict[str, torch.Tensor] = {}
    for index, name in enumerate(COMPONENTS):
        loss = losses.losses[name]
        _scalar(loss, label=f"{name} loss")
        gradient = torch.autograd.grad(
            loss,
            parameter,
            retain_graph=index < len(COMPONENTS) - 1,
            create_graph=False,
            allow_unused=True,
        )[0]
        gradients[name] = _validate_gradient(
            gradient,
            label=name,
            expected_shape=parameter.shape,
        )
    return gradients


def run_gradient_decomposition(
    *,
    parameter: torch.Tensor,
    frames: Sequence[GradientDiagnosticFrame],
    forward_frame: Callable[[GradientDiagnosticFrame], ComponentLosses],
) -> dict[str, Any]:
    """Analyze batch-mean raw gradients while preserving texture bit-for-bit."""

    if not isinstance(parameter, torch.Tensor) or not parameter.requires_grad:
        raise MultiLevelGradientDiagnosticError(
            "texture parameter must be a trainable tensor"
        )
    if not frames:
        raise MultiLevelGradientDiagnosticError("diagnostic frame batch is empty")
    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise MultiLevelGradientDiagnosticError("diagnostic frame IDs must be unique")

    theta_before = parameter.detach().clone()
    accumulator = ComponentGradientAccumulator()
    mse_totals = {name: 0.0 for name in COMPONENTS}
    try:
        for frame in frames:
            losses = forward_frame(frame)
            gradients = component_texture_gradients(losses, parameter)
            accumulator.add(gradients)
            for name in COMPONENTS:
                mse_totals[name] += _scalar(
                    losses.mses[name], label=f"{name} MSE"
                ) / len(frames)
            del losses, gradients
        batch_gradients = accumulator.finalize()
        analysis = analyze_batch_gradients(batch_gradients)
    finally:
        texture_unchanged = torch.equal(theta_before, parameter.detach())
    if not texture_unchanged:
        raise MultiLevelGradientDiagnosticError(
            "diagnostic mutated the texture parameter"
        )

    return {
        "batch_size": len(frames),
        "frame_ids": frame_ids,
        "o1_mse": mse_totals["O1"],
        "o2_mse": mse_totals["O2"],
        "p1_mse": mse_totals["P1"],
        "p2_mse": mse_totals["P2"],
        "o1_to_o2_mse_ratio": mse_totals["O1"] / (mse_totals["O2"] + DIAGNOSTIC_EPS),
        "p1_to_p2_mse_ratio": mse_totals["P1"] / (mse_totals["P2"] + DIAGNOSTIC_EPS),
        **analysis,
        "texture_parameter_unchanged": True,
    }


def load_texture_parameter(
    parameter: torch.Tensor,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Load one tensor artifact into the renderer before the read-only guard."""

    if (
        not isinstance(parameter, torch.Tensor)
        or not parameter.is_floating_point()
        or parameter.numel() == 0
    ):
        raise MultiLevelGradientDiagnosticError(
            "renderer texture parameter must be a non-empty floating tensor"
        )
    path = Path(artifact_path).expanduser().resolve(strict=True)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise MultiLevelGradientDiagnosticError(
            "texture artifact must contain one floating tensor"
        )
    if tuple(value.shape) != tuple(parameter.shape):
        raise MultiLevelGradientDiagnosticError(
            f"texture artifact shape mismatch: {tuple(value.shape)} != "
            f"{tuple(parameter.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise MultiLevelGradientDiagnosticError("texture artifact is non-finite")
    with torch.no_grad():
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return {
        "absolute_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
