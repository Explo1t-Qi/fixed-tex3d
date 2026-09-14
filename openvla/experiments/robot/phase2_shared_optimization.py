"""Phase 2.4 shared-feature training and paired-evaluation contracts.

The real model and renderer entry points inject their frame-specific forward
functions here.  Keeping the optimizer mechanics independent of those heavy
dependencies makes the frozen scientific substrate and parameterized optimizer
testable on CPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class Phase2SharedOptimizationError(RuntimeError):
    """Raised at the earliest invalid Phase 2.4 training/evaluation stage."""


@dataclass(frozen=True)
class SharedOptimizationProtocol:
    task_suite: str = "libero_spatial"
    task_id: int = 0
    object_name: str = "akita_black_bowl"
    num_train_init_states: int = 10
    train_frames_per_state: int = 1
    num_frames_to_attack: int = 20
    attack_iterations: int = 500
    pgd_step: float = 0.05
    seed: int = 7

    def validate_frozen_pilot(self) -> None:
        configuration = self.validate_training_configuration()
        if configuration["changes_from_baseline"]:
            raise Phase2SharedOptimizationError(
                "frozen Phase 2.4 pilot must use baseline optimization parameters"
            )

    def validate_training_configuration(self) -> dict[str, Any]:
        """Validate the frozen substrate while allowing optimization parameters."""

        expected = SharedOptimizationProtocol()
        adjustable = frozenset({"attack_iterations", "pgd_step"})
        actual_values = asdict(self)
        expected_values = asdict(expected)
        changes = {
            name: {"baseline": expected_values[name], "actual": value}
            for name, value in actual_values.items()
            if value != expected_values[name]
        }
        frozen_changes = set(changes) - adjustable
        if frozen_changes:
            raise Phase2SharedOptimizationError(
                "Phase 2.4 run changed frozen protocol fields: "
                f"{sorted(frozen_changes)}"
            )
        if not np.isfinite(self.pgd_step) or self.pgd_step <= 0:
            raise Phase2SharedOptimizationError("PGD step must be finite and positive")
        if self.attack_iterations < 1:
            raise Phase2SharedOptimizationError("attack iterations must be positive")
        return {
            "adjustable_parameters": {
                name: {
                    "baseline": expected_values[name],
                    "actual": actual_values[name],
                    "changed": name in changes,
                }
                for name in sorted(adjustable)
            },
            "frozen_protocol_fields_equal": True,
            "changes_from_baseline": changes,
        }


@dataclass(frozen=True)
class FrozenCleanReference:
    h_o: torch.Tensor
    h_p: torch.Tensor

    @classmethod
    def from_tensors(
        cls, h_o: torch.Tensor, h_p: torch.Tensor
    ) -> "FrozenCleanReference":
        values = []
        for name, value in (("h_o", h_o), ("h_p", h_p)):
            if not isinstance(value, torch.Tensor):
                raise Phase2SharedOptimizationError(f"clean {name} is not a tensor")
            if value.ndim != 3 or tuple(value.shape[1:]) != (256, 262):
                raise Phase2SharedOptimizationError(
                    f"clean {name} must be [B,256,262], got {tuple(value.shape)}"
                )
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise Phase2SharedOptimizationError(
                    f"clean {name} must be finite floating point"
                )
            values.append(value.detach().clone())
        return cls(*values)


@dataclass(frozen=True)
class SharedTrainingFrame:
    frame_id: str
    state_id: int
    pi05_template_id: str
    clean: FrozenCleanReference
    payload: Any


@dataclass(frozen=True)
class PairedTrial:
    state_id: int
    clean_success: bool
    adversarial_success: bool


def paired_source_metrics(trials: Sequence[PairedTrial]) -> dict[str, Any]:
    if not trials:
        raise Phase2SharedOptimizationError("paired evaluation requires trials")
    state_ids = [trial.state_id for trial in trials]
    if len(state_ids) != len(set(state_ids)):
        raise Phase2SharedOptimizationError("paired trial state IDs must be unique")
    clean_successes = sum(int(t.clean_success) for t in trials)
    adversarial_successes = sum(int(t.adversarial_success) for t in trials)
    paired_failures = sum(
        int(t.clean_success and not t.adversarial_success) for t in trials
    )
    count = len(trials)
    return {
        "num_trials": count,
        "state_ids": state_ids,
        "clean_successes": clean_successes,
        "adversarial_successes": adversarial_successes,
        "clean_success_rate": clean_successes / count,
        "adversarial_success_rate": adversarial_successes / count,
        "adversarial_failure_rate": 1.0 - adversarial_successes / count,
        "clean_success_adv_failure_count": paired_failures,
        "conditional_asr": (
            paired_failures / clean_successes if clean_successes else None
        ),
    }


def sign_pgd_update(
    parameter: torch.Tensor, *, step_size: float
) -> tuple[float, float]:
    if step_size <= 0 or not np.isfinite(step_size):
        raise Phase2SharedOptimizationError("PGD step must be finite and positive")
    gradient = parameter.grad
    if gradient is None:
        raise Phase2SharedOptimizationError("texture gradient is missing")
    if not bool(torch.isfinite(gradient).all()):
        raise Phase2SharedOptimizationError("texture gradient is non-finite")
    gradient_norm = float(gradient.norm().detach().item())
    if gradient_norm == 0.0:
        raise Phase2SharedOptimizationError("texture gradient is zero")
    before = parameter.detach().clone()
    with torch.no_grad():
        parameter.data -= step_size * gradient.sign()
    change_linf = float((parameter.detach() - before).abs().max().item())
    if not np.isfinite(change_linf) or change_linf <= 0.0:
        raise Phase2SharedOptimizationError("texture parameter did not change")
    return gradient_norm, change_linf


def validate_texture_budget(renderer: Any, *, tolerance: float = 1e-6) -> float:
    parameter = renderer.adv_noise
    epsilon = float(renderer.epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise Phase2SharedOptimizationError("renderer epsilon is invalid")
    perturbation = torch.tanh(parameter.detach()).abs() * epsilon
    maximum = float(perturbation.max().item())
    if not np.isfinite(maximum) or maximum > epsilon + tolerance:
        raise Phase2SharedOptimizationError(
            f"texture budget violated: {maximum} > {epsilon}"
        )
    return maximum


def _metric_value(result: Any, name: str) -> float:
    value = getattr(result, name)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise Phase2SharedOptimizationError(f"loss result {name} must be scalar")
    scalar = float(value.detach().item())
    if not np.isfinite(scalar):
        raise Phase2SharedOptimizationError(f"loss result {name} is non-finite")
    return scalar


def train_shared_texture(
    *,
    renderer: Any,
    frames: Sequence[SharedTrainingFrame],
    forward_frame: Callable[[SharedTrainingFrame], tuple[Any, torch.Tensor]],
    iterations: int,
    requested_batch_size: int,
    pgd_step: float,
    seed: int,
    metrics_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Accumulate mean frame loss and make exactly one sign update per step."""

    if iterations < 1 or requested_batch_size < 1 or not frames:
        raise Phase2SharedOptimizationError("invalid training loop dimensions")
    frame_ids = [frame.frame_id for frame in frames]
    template_ids = [frame.pi05_template_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise Phase2SharedOptimizationError("frame IDs must be unique")
    if len(template_ids) != len(set(template_ids)):
        raise Phase2SharedOptimizationError(
            "each training frame requires its own PI0Pytorch template"
        )
    parameter = renderer.adv_noise
    if not isinstance(parameter, torch.Tensor) or not parameter.requires_grad:
        raise Phase2SharedOptimizationError("renderer adv_noise must be trainable")
    batch_size = min(requested_batch_size, len(frames))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")

    names = (
        "loss",
        "shared_mse",
        "o2_mse",
        "p2_mse",
        "displacement_cosine_mean",
        "o2_to_p2_mse_ratio",
    )
    for iteration in range(iterations):
        parameter.grad = None
        chosen = rng.choice(len(frames), size=batch_size, replace=False)
        totals = {name: 0.0 for name in names}
        image_gradient_norms = []
        for index in chosen:
            frame = frames[int(index)]
            result, adv_image = forward_frame(frame)
            loss = getattr(result, "loss", None)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise Phase2SharedOptimizationError("shared loss must be scalar")
            if not bool(torch.isfinite(loss)):
                raise Phase2SharedOptimizationError("shared loss is non-finite")
            adv_image.retain_grad()
            (loss / batch_size).backward()
            image_gradient = adv_image.grad
            if image_gradient is None:
                raise Phase2SharedOptimizationError(
                    "adversarial image gradient is missing"
                )
            if not bool(torch.isfinite(image_gradient).all()):
                raise Phase2SharedOptimizationError(
                    "adversarial image gradient is non-finite"
                )
            image_norm = float(image_gradient.norm().item())
            if image_norm == 0.0:
                raise Phase2SharedOptimizationError(
                    "adversarial image gradient is zero"
                )
            image_gradient_norms.append(image_norm)
            for name in names:
                totals[name] += _metric_value(result, name) / batch_size

        grad_norm, change_linf = sign_pgd_update(parameter, step_size=pgd_step)
        maximum_perturbation = validate_texture_budget(renderer)
        row = {
            "iteration": iteration,
            "selected_frame_ids": [frames[int(i)].frame_id for i in chosen],
            "loss_shared": totals.pop("loss"),
            **totals,
            "texture_gradient_norm": grad_norm,
            "image_gradient_norm_mean": float(np.mean(image_gradient_norms)),
            "parameter_change_linf": change_linf,
            "maximum_texture_perturbation": maximum_perturbation,
        }
        history.append(row)
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
    return history


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
