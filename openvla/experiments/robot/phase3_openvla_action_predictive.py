"""OpenVLA-only action-predictive Phase 3 ablation (no model ensemble)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from phase2_native_gradient_ensemble import _model_gradient, _scalar
from phase2_shared_gradient import O2_SHAPE, _validate_finite_feature
from phase2_shared_optimization import sign_pgd_update, validate_texture_budget
from phase3_action_predictive_objective import (
    ACTION_NAMES,
    CALIBRATION_EPS,
    DIRECTION_EPS,
    ActionPredictiveObjectiveError,
    FrozenActionLinearProbe,
)


@dataclass(frozen=True)
class OpenVLAReference:
    o2: torch.Tensor
    z: torch.Tensor

    @classmethod
    def capture(
        cls, o2: torch.Tensor, probe: FrozenActionLinearProbe
    ) -> OpenVLAReference:
        _validate_finite_feature(o2, name="clean O2", expected_tail=O2_SHAPE)
        with torch.no_grad():
            z = probe(o2)
        return cls(o2.detach().clone(), z.detach().clone())


@dataclass(frozen=True)
class OpenVLALosses:
    loss: torch.Tensor
    loss_mag: torch.Tensor
    loss_dir: torch.Tensor
    action_mse: torch.Tensor
    native_o2_mse: torch.Tensor
    delta_z_norm: torch.Tensor
    delta_z: torch.Tensor


def openvla_action_losses(
    o2_adv: torch.Tensor,
    clean: OpenVLAReference,
    *,
    probe: FrozenActionLinearProbe,
    lambda_dir: float,
) -> OpenVLALosses:
    if not math.isfinite(lambda_dir) or lambda_dir <= 0:
        raise ActionPredictiveObjectiveError("lambda_dir must be finite and positive")
    _validate_finite_feature(o2_adv, name="adversarial O2", expected_tail=O2_SHAPE)
    _validate_finite_feature(clean.o2, name="clean O2", expected_tail=O2_SHAPE)
    if clean.o2.requires_grad or clean.z.requires_grad:
        raise ActionPredictiveObjectiveError("clean reference must be detached")
    z_adv = probe(o2_adv)
    if z_adv.shape != clean.z.shape or z_adv.device != clean.z.device:
        raise ActionPredictiveObjectiveError(
            "clean/adversarial Z shape or device differs"
        )
    delta = z_adv - clean.z
    action_mse = delta.square().mean()
    loss_mag = -action_mse
    loss_dir = F.cosine_similarity(z_adv, clean.z, dim=-1, eps=DIRECTION_EPS).mean()
    return OpenVLALosses(
        loss=loss_mag + lambda_dir * loss_dir,
        loss_mag=loss_mag,
        loss_dir=loss_dir,
        action_mse=action_mse,
        native_o2_mse=(o2_adv.float() - clean.o2.float()).square().mean(),
        delta_z_norm=delta.norm(dim=-1).mean(),
        delta_z=delta.mean(dim=0),
    )


@dataclass(frozen=True)
class OpenVLATrainingFrame:
    frame_id: str
    state_id: int
    clean: OpenVLAReference
    payload: Any


def calibrate_openvla_lambda(
    *,
    renderer: Any,
    frames: Sequence[OpenVLATrainingFrame],
    forward_frame: Callable[[OpenVLATrainingFrame, float], OpenVLALosses],
) -> dict[str, Any]:
    """Read-only median of frame-local OpenVLA component gradient norm ratios."""
    if not frames:
        raise ActionPredictiveObjectiveError("calibration frame set is empty")
    parameter = renderer.adv_noise
    before = parameter.detach().clone()
    rows = []
    ratios = []
    for frame in frames:
        losses = forward_frame(frame, 1.0)
        g_mag = _model_gradient(
            losses.loss_mag, parameter, label="OpenVLA magnitude", retain_graph=True
        )
        g_dir = _model_gradient(
            losses.loss_dir, parameter, label="OpenVLA direction", retain_graph=False
        )
        if not bool(torch.any(g_mag != 0)) or not bool(torch.any(g_dir != 0)):
            raise ActionPredictiveObjectiveError("OpenVLA calibration gradient is zero")
        ratio = float(g_mag.norm().item() / (g_dir.norm().item() + CALIBRATION_EPS))
        if not math.isfinite(ratio):
            raise ActionPredictiveObjectiveError(
                "OpenVLA calibration ratio is non-finite"
            )
        ratios.append(ratio)
        rows.append(
            {
                "frame_id": frame.frame_id,
                "g_mag_l2_norm": float(g_mag.norm().item()),
                "g_dir_l2_norm": float(g_dir.norm().item()),
                "norm_ratio": ratio,
            }
        )
    if not torch.equal(before, parameter.detach()):
        raise ActionPredictiveObjectiveError("calibration mutated texture parameter")
    value = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not math.isfinite(value) or value <= 0:
        raise ActionPredictiveObjectiveError("calibrated lambda_dir is unstable")
    return {
        "schema_version": "phase3_openvla_lambda_calibration_v1",
        "status": "PASS",
        "method": "median_frames(norm(g_mag_O)/(norm(g_dir_O)+eps))",
        "epsilon": CALIBRATION_EPS,
        "lambda_dir": value,
        "frames": rows,
        "texture_parameter_unchanged": True,
    }


def train_openvla_action_predictive(
    *,
    renderer: Any,
    frames: Sequence[OpenVLATrainingFrame],
    forward_frame: Callable[[OpenVLATrainingFrame, float], OpenVLALosses],
    lambda_dir: float,
    iterations: int,
    requested_batch_size: int,
    pgd_step: float,
    seed: int,
    metrics_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if iterations < 1 or requested_batch_size < 1 or not frames:
        raise ActionPredictiveObjectiveError("invalid training loop dimensions")
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise ActionPredictiveObjectiveError("frame IDs must be unique")
    parameter = renderer.adv_noise
    batch_size = min(requested_batch_size, len(frames))
    rng = np.random.default_rng(seed)
    history = []
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
    for iteration in range(iterations):
        parameter.grad = None
        chosen = rng.choice(len(frames), size=batch_size, replace=False)
        gradient_sum = torch.zeros_like(parameter)
        names = (
            "loss",
            "loss_mag",
            "loss_dir",
            "action_mse",
            "native_o2_mse",
            "delta_z_norm",
        )
        totals = {name: 0.0 for name in names}
        signed = torch.zeros(len(ACTION_NAMES))
        absolute = torch.zeros(len(ACTION_NAMES))
        for index in chosen:
            losses = forward_frame(frames[int(index)], lambda_dir)
            gradient_sum.add_(
                _model_gradient(
                    losses.loss,
                    parameter,
                    label="OpenVLA action-aware",
                    retain_graph=False,
                )
            )
            for name in names:
                totals[name] += _scalar(getattr(losses, name), label=name) / batch_size
            signed += losses.delta_z.detach().cpu() / batch_size
            absolute += losses.delta_z.detach().abs().cpu() / batch_size
        gradient = (
            gradient_sum / batch_size
        )  # g = mean_frame(g_openvla); no model normalization.
        parameter.grad = gradient.detach().clone()
        gradient_norm, change_linf = sign_pgd_update(parameter, step_size=pgd_step)
        maximum = validate_texture_budget(renderer)
        row = {
            "iteration": iteration,
            "selected_frame_ids": [frames[int(i)].frame_id for i in chosen],
            "lambda_dir": lambda_dir,
            **totals,
            "openvla_action_coordinate_mse": totals["action_mse"],
            "o2_mse": totals["native_o2_mse"],
            "openvla_texture_gradient_l2_norm": gradient_norm,
            "parameter_change_linf": change_linf,
            "maximum_texture_perturbation": maximum,
            "texture_budget_respected": maximum <= float(renderer.epsilon) + 1e-6,
        }
        for index, name in enumerate(ACTION_NAMES):
            row[f"openvla_delta_{name}"] = float(signed[index])
            row[f"openvla_abs_delta_{name}"] = float(absolute[index])
        history.append(row)
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
    return history
