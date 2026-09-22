"""Action-predictive O2/P2 objective on the frozen native-GE substrate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase2_native_gradient_ensemble import (
    ModelGradientAccumulator,
    NativeFeatures,
    _gradient_cosine,
    _model_gradient,
    _scalar,
)
from phase2_shared_gradient import (
    O2_SHAPE,
    P2_SHAPE,
    _validate_finite_feature,
    extract_pi05_p2_from_preprocessed_base_image,
)
from phase2_shared_optimization import sign_pgd_update, validate_texture_budget
from phase2_action_representation import pi05_p2_identity_metrics


ACTION_NAMES = ("x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper")
ACTION_DIMENSION = 7
DIRECTION_EPS = 1e-8
CALIBRATION_EPS = 1e-12
PHASE2B_SCHEMA = "phase2b_primary_action_probes_v2"
PHASE2_EXPANDED_SEED7_PROVISIONAL_SCHEMA = (
    "phase2_expanded_seed7_provisional_action_probes_v1"
)
PHASE2_EXPANDED_SEED7_PROVISIONAL_STATUS = (
    "PHASE2_EXPANDED_SEED7_PROVISIONAL_FROZEN"
)


class ActionPredictiveObjectiveError(RuntimeError):
    """Raised at the first invalid Phase 3 objective stage."""


def validate_pi05_probe_runtime_identity(
    *, model: Any, preprocessed_base_image: torch.Tensor
) -> dict[str, Any]:
    """Gate native and live P2 on the *same* model input and autocast context."""

    try:
        with torch.no_grad():
            authoritative_p2 = model.paligemma_with_expert.embed_image(
                preprocessed_base_image
            )
            runtime_p2 = extract_pi05_p2_from_preprocessed_base_image(
                model, preprocessed_base_image
            )
        result = pi05_p2_identity_metrics(
            extractor=authoritative_p2,
            embed_image=runtime_p2,
            prefix=authoritative_p2,
        )
        result["comparison_input"] = "same_preprocessed_base_0_rgb_tensor"
        return result
    except Exception as error:
        raise ActionPredictiveObjectiveError(
            "Phase 3 PI0.5 probe/runtime P2 identity mismatch"
        ) from error


def pi05_preprocessing_input_difference(
    official_base_image: torch.Tensor, differentiable_base_image: torch.Tensor
) -> dict[str, Any]:
    """Report preprocessing gap; this is not a P2 implementation gate."""

    if (
        not isinstance(official_base_image, torch.Tensor)
        or not isinstance(differentiable_base_image, torch.Tensor)
        or official_base_image.shape != differentiable_base_image.shape
        or not official_base_image.is_floating_point()
        or not differentiable_base_image.is_floating_point()
        or not bool(torch.isfinite(official_base_image).all())
        or not bool(torch.isfinite(differentiable_base_image).all())
    ):
        raise ActionPredictiveObjectiveError(
            "PI0.5 preprocessing inputs must be finite, floating, and shape-matched"
        )
    official = official_base_image.detach().float()
    difference = differentiable_base_image.detach().float() - official
    return {
        "shape": list(official.shape),
        "official_dtype": str(official_base_image.dtype),
        "differentiable_dtype": str(differentiable_base_image.dtype),
        "max_abs_difference": float(difference.abs().max()),
        "mean_abs_difference": float(difference.abs().mean()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(official).clamp_min(1e-12)
        ),
        "hard_gate": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class FrozenActionLinearProbe(nn.Module):
    """Frozen Linear(D, 7, bias=False) mapping used as action coordinates."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        safe_action_std: torch.Tensor,
        model: str,
        node: str,
    ) -> None:
        super().__init__()
        if model not in {"openvla", "pi05"}:
            raise ActionPredictiveObjectiveError(f"unsupported probe model: {model}")
        expected_node = "o2" if model == "openvla" else "p2"
        expected_width = O2_SHAPE[1] if model == "openvla" else P2_SHAPE[1]
        if node != expected_node:
            raise ActionPredictiveObjectiveError(
                f"{model} Phase 3 probe must use {expected_node}, got {node}"
            )
        if (
            not isinstance(weight, torch.Tensor)
            or not weight.is_floating_point()
            or tuple(weight.shape) != (ACTION_DIMENSION, expected_width)
            or not bool(torch.isfinite(weight).all())
        ):
            raise ActionPredictiveObjectiveError(
                f"{model}/{node} W must be finite [{ACTION_DIMENSION},{expected_width}]"
            )
        statistics = {
            "action_mean": action_mean,
            "action_std": action_std,
            "safe_action_std": safe_action_std,
        }
        for name, value in statistics.items():
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or tuple(value.shape) != (ACTION_DIMENSION,)
                or not bool(torch.isfinite(value).all())
            ):
                raise ActionPredictiveObjectiveError(
                    f"{model}/{node} {name} must be finite [7]"
                )
        if bool(torch.any(safe_action_std <= 0)):
            raise ActionPredictiveObjectiveError("safe action std must be positive")
        self.model = model
        self.node = node
        self.register_buffer("weight", weight.detach().float().clone())
        self.register_buffer("action_mean", action_mean.detach().float().clone())
        self.register_buffer("action_std", action_std.detach().float().clone())
        self.register_buffer(
            "safe_action_std", safe_action_std.detach().float().clone()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected_tail = O2_SHAPE if self.model == "openvla" else P2_SHAPE
        _validate_finite_feature(
            features,
            name=f"{self.model} {self.node} probe input",
            expected_tail=expected_tail,
        )
        return features.float().mean(dim=1) @ self.weight.t()


@dataclass(frozen=True)
class FrozenProbeArtifact:
    root: Path
    metadata: dict[str, Any]
    openvla: FrozenActionLinearProbe
    pi05: FrozenActionLinearProbe
    hashes: dict[str, str]


def _load_probe_node(
    root: Path,
    *,
    model: str,
    node: str,
    device: torch.device,
    inventory: dict[str, Any],
) -> tuple[FrozenActionLinearProbe, dict[str, str]]:
    node_dir = root / model / node
    required = ("W.pt", "action_stats.npz", "probe.pt", "metrics.json")
    hashes: dict[str, str] = {}
    for filename in required:
        path = node_dir / filename
        if not path.is_file():
            raise ActionPredictiveObjectiveError(f"missing probe artifact: {path}")
        relative = str(path.relative_to(root))
        actual = _sha256(path)
        recorded = inventory.get(relative, {}).get("sha256")
        if recorded != actual:
            raise ActionPredictiveObjectiveError(
                f"probe artifact hash mismatch: {relative}"
            )
        hashes[relative] = actual
    weight = torch.load(node_dir / "W.pt", map_location="cpu", weights_only=True)
    with np.load(node_dir / "action_stats.npz", allow_pickle=False) as archive:
        if not {"mean", "std", "safe_std", "near_constant"}.issubset(archive.files):
            raise ActionPredictiveObjectiveError("action_stats.npz schema mismatch")
        mean = torch.from_numpy(np.asarray(archive["mean"], dtype=np.float32))
        std = torch.from_numpy(np.asarray(archive["std"], dtype=np.float32))
        safe_std = torch.from_numpy(np.asarray(archive["safe_std"], dtype=np.float32))
    probe = FrozenActionLinearProbe(
        weight,
        action_mean=mean,
        action_std=std,
        safe_action_std=safe_std,
        model=model,
        node=node,
    ).to(device)
    probe.eval()
    if tuple(probe.parameters()):
        raise ActionPredictiveObjectiveError("frozen probe must not own parameters")
    return probe, hashes


def load_frozen_primary_probes(
    artifact_dir: str | Path,
    *,
    openvla_device: torch.device | str,
    pi05_device: torch.device | str,
) -> FrozenProbeArtifact:
    root = Path(artifact_dir).expanduser().resolve(strict=True)
    metadata_path = root / "metadata.json"
    inventory_path = root / "artifact_inventory.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    historical_phase2b = (
        metadata.get("schema_version") == PHASE2B_SCHEMA
        and metadata.get("status") == "PHASE_2B_PRIMARY_FROZEN"
        and metadata.get("split_rule") == "pilot-v0.2-c5-split-v1"
    )
    expanded_provisional = (
        metadata.get("schema_version") == PHASE2_EXPANDED_SEED7_PROVISIONAL_SCHEMA
        and metadata.get("status") == PHASE2_EXPANDED_SEED7_PROVISIONAL_STATUS
        and metadata.get("artifact_id")
        == "phase2-expanded-seed7-provisional-action-probes-v1"
        and metadata.get("split_rule") == "pilot-v0.3-expanded-split-v1"
        and metadata.get("phase2_status", {}).get("authoritative_phase2b_v3")
        == "NOT_FROZEN_BLOCKED"
        and metadata.get("phase2_status", {}).get(
            "phase3_exploratory_pipeline_feasibility"
        )
        == "AUTHORIZED_FIXED_SEED7_ONLY"
    )
    if not (historical_phase2b or expanded_provisional) or metadata.get("seed") != 7:
        raise ActionPredictiveObjectiveError("invalid Phase 2 probe metadata")
    pi05_projected = (
        metadata.get("models", {})
        .get("pi05", {})
        .get("representation_nodes", {})
        .get("projected", {})
    )
    if (
        pi05_projected.get("definition_id")
        != "pi05_p2_embed_image_no_manual_scaling_v2"
    ):
        raise ActionPredictiveObjectiveError(
            "Phase 2 PI0.5 P2 is not runtime-identity-corrected"
        )
    if (
        historical_phase2b
        and metadata.get("provenance", {}).get("openvla_W_unchanged") is not True
    ):
        raise ActionPredictiveObjectiveError("historical Phase 2B O2 provenance failed")
    if expanded_provisional and metadata.get("provenance", {}).get("promotion") != (
        "byte-identical copy; no probe refitting"
    ):
        raise ActionPredictiveObjectiveError(
            "expanded provisional probe provenance failed"
        )
    expected_probe = {
        "architecture": "Linear(D,7,bias=False)",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "steps": 2000,
        "action_std_epsilon": 1e-6,
        "probe_reg": 1e-4,
    }
    for key, expected in expected_probe.items():
        if metadata.get("probe", {}).get(key) != expected:
            raise ActionPredictiveObjectiveError(
                f"Phase 2B probe contract mismatch for {key}"
            )
    inventory_without_self = {
        key: value
        for key, value in inventory.items()
        if key != "artifact_inventory.json"
    }
    if "metadata.json" not in inventory_without_self:
        raise ActionPredictiveObjectiveError(
            "metadata is absent from artifact inventory"
        )
    if inventory_without_self["metadata.json"].get("sha256") != _sha256(metadata_path):
        raise ActionPredictiveObjectiveError("Phase 2B metadata hash mismatch")
    openvla, o_hashes = _load_probe_node(
        root,
        model="openvla",
        node="o2",
        device=torch.device(openvla_device),
        inventory=inventory,
    )
    pi05, p_hashes = _load_probe_node(
        root,
        model="pi05",
        node="p2",
        device=torch.device(pi05_device),
        inventory=inventory,
    )
    return FrozenProbeArtifact(
        root=root,
        metadata=metadata,
        openvla=openvla,
        pi05=pi05,
        hashes={**o_hashes, **p_hashes},
    )


@dataclass(frozen=True)
class FrozenActionPredictiveReference:
    o2: torch.Tensor
    p2: torch.Tensor
    z_o: torch.Tensor
    z_p: torch.Tensor

    @classmethod
    def from_tensors(
        cls,
        *,
        o2: torch.Tensor,
        p2: torch.Tensor,
        z_o: torch.Tensor,
        z_p: torch.Tensor,
    ) -> "FrozenActionPredictiveReference":
        _validate_finite_feature(o2, name="clean O2", expected_tail=O2_SHAPE)
        _validate_finite_feature(p2, name="clean P2", expected_tail=P2_SHAPE)
        for name, value in (("Z_O", z_o), ("Z_P", z_p)):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[1] != ACTION_DIMENSION
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ActionPredictiveObjectiveError(f"clean {name} must be [B,7]")
        if not (o2.shape[0] == p2.shape[0] == z_o.shape[0] == z_p.shape[0]):
            raise ActionPredictiveObjectiveError("clean batch sizes differ")
        return cls(*(value.detach().clone() for value in (o2, p2, z_o, z_p)))


@dataclass(frozen=True)
class ActionPredictiveLosses:
    loss_o: torch.Tensor
    loss_p: torch.Tensor
    loss_mag_o: torch.Tensor
    loss_mag_p: torch.Tensor
    loss_dir_o: torch.Tensor
    loss_dir_p: torch.Tensor
    action_mse_o: torch.Tensor
    action_mse_p: torch.Tensor
    native_mse_o: torch.Tensor
    native_mse_p: torch.Tensor
    z_clean_o_norm: torch.Tensor
    z_clean_p_norm: torch.Tensor
    z_adv_o_norm: torch.Tensor
    z_adv_p_norm: torch.Tensor
    delta_z_o_norm: torch.Tensor
    delta_z_p_norm: torch.Tensor
    delta_z_o: torch.Tensor
    delta_z_p: torch.Tensor


def _direction(adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(adv, clean, dim=-1, eps=DIRECTION_EPS).mean()


def action_predictive_losses(
    adversarial: NativeFeatures,
    clean: FrozenActionPredictiveReference,
    *,
    openvla_probe: FrozenActionLinearProbe,
    pi05_probe: FrozenActionLinearProbe,
    lambda_dir: float,
) -> ActionPredictiveLosses:
    if not math.isfinite(lambda_dir) or lambda_dir <= 0:
        raise ActionPredictiveObjectiveError("lambda_dir must be finite and positive")
    for name, value, expected_tail in (
        ("clean O2", clean.o2, O2_SHAPE),
        ("clean P2", clean.p2, P2_SHAPE),
    ):
        _validate_finite_feature(value, name=name, expected_tail=expected_tail)
        if value.requires_grad:
            raise ActionPredictiveObjectiveError(f"{name} must be detached")
    for name, value in (("clean Z_O", clean.z_o), ("clean Z_P", clean.z_p)):
        if value.requires_grad:
            raise ActionPredictiveObjectiveError(f"{name} must be detached")
        if not bool(torch.isfinite(value).all()):
            raise ActionPredictiveObjectiveError(f"{name} is non-finite")
    z_o = openvla_probe(adversarial.o2)
    z_p = pi05_probe(adversarial.p2)
    if (
        z_o.shape != clean.z_o.shape
        or z_p.shape != clean.z_p.shape
        or z_o.device != clean.z_o.device
        or z_p.device != clean.z_p.device
    ):
        raise ActionPredictiveObjectiveError(
            "clean/adversarial coordinate shapes or devices differ"
        )
    delta_o = z_o - clean.z_o
    delta_p = z_p - clean.z_p
    mse_o = delta_o.square().mean()
    mse_p = delta_p.square().mean()
    mag_o = -mse_o
    mag_p = -mse_p
    dir_o = _direction(z_o, clean.z_o)
    dir_p = _direction(z_p, clean.z_p)
    native_o = (adversarial.o2.float() - clean.o2.float()).square().mean()
    native_p = (adversarial.p2.float() - clean.p2.float()).square().mean()
    return ActionPredictiveLosses(
        loss_o=mag_o + lambda_dir * dir_o,
        loss_p=mag_p + lambda_dir * dir_p,
        loss_mag_o=mag_o,
        loss_mag_p=mag_p,
        loss_dir_o=dir_o,
        loss_dir_p=dir_p,
        action_mse_o=mse_o,
        action_mse_p=mse_p,
        native_mse_o=native_o,
        native_mse_p=native_p,
        z_clean_o_norm=clean.z_o.norm(dim=-1).mean(),
        z_clean_p_norm=clean.z_p.norm(dim=-1).mean(),
        z_adv_o_norm=z_o.norm(dim=-1).mean(),
        z_adv_p_norm=z_p.norm(dim=-1).mean(),
        delta_z_o_norm=delta_o.norm(dim=-1).mean(),
        delta_z_p_norm=delta_p.norm(dim=-1).mean(),
        delta_z_o=delta_o.mean(dim=0),
        delta_z_p=delta_p.mean(dim=0),
    )


class DualVLAActionPredictiveAdapter:
    """Attach frozen model-specific probes to existing differentiable O2/P2 paths."""

    def __init__(self, *, native_adapter: Any, probes: FrozenProbeArtifact) -> None:
        self.native_adapter = native_adapter
        self.probes = probes

    def clean_reference(
        self, source_rgb: torch.Tensor
    ) -> FrozenActionPredictiveReference:
        with torch.no_grad():
            features = self.native_adapter.features(source_rgb)
            z_o = self.probes.openvla(features.o2)
            z_p = self.probes.pi05(features.p2)
        return FrozenActionPredictiveReference.from_tensors(
            o2=features.o2, p2=features.p2, z_o=z_o, z_p=z_p
        )

    def losses(
        self,
        source_rgb: torch.Tensor,
        clean: FrozenActionPredictiveReference,
        *,
        lambda_dir: float,
    ) -> tuple[ActionPredictiveLosses, NativeFeatures]:
        features = self.native_adapter.features(source_rgb)
        losses = action_predictive_losses(
            features,
            clean,
            openvla_probe=self.probes.openvla,
            pi05_probe=self.probes.pi05,
            lambda_dir=lambda_dir,
        )
        return losses, features


@dataclass(frozen=True)
class ActionPredictiveTrainingFrame:
    frame_id: str
    state_id: int
    pi05_template_id: str
    clean: FrozenActionPredictiveReference
    payload: Any


def _gradient_metrics(gradient: torch.Tensor) -> dict[str, float]:
    return {
        "l2_norm": float(gradient.norm().item()),
        "mean_abs": float(gradient.abs().mean().item()),
    }


def calibrate_lambda_dir(
    *,
    renderer: Any,
    frames: Sequence[ActionPredictiveTrainingFrame],
    forward_frame: Callable[
        [ActionPredictiveTrainingFrame], tuple[ActionPredictiveLosses, torch.Tensor]
    ],
) -> dict[str, Any]:
    """Calibrate once from frame-local raw component gradients without updates."""

    if not frames:
        raise ActionPredictiveObjectiveError("calibration frame set is empty")
    parameter = renderer.adv_noise
    before = parameter.detach().clone()
    rows: list[dict[str, Any]] = []
    ratios_o: list[float] = []
    ratios_p: list[float] = []
    for frame in frames:
        losses, image = forward_frame(frame)
        gradients = {
            "g_mag_o": _model_gradient(
                losses.loss_mag_o,
                parameter,
                label="calibration OpenVLA magnitude",
                retain_graph=True,
            ),
            "g_dir_o": _model_gradient(
                losses.loss_dir_o,
                parameter,
                label="calibration OpenVLA direction",
                retain_graph=True,
            ),
            "g_mag_p": _model_gradient(
                losses.loss_mag_p,
                parameter,
                label="calibration PI0.5 magnitude",
                retain_graph=True,
            ),
            "g_dir_p": _model_gradient(
                losses.loss_dir_p,
                parameter,
                label="calibration PI0.5 direction",
                retain_graph=False,
            ),
        }
        for label, gradient in gradients.items():
            if not bool(torch.any(gradient != 0)):
                raise ActionPredictiveObjectiveError(f"{label} gradient is zero")
        ratio_o = float(
            gradients["g_mag_o"].norm().item()
            / (gradients["g_dir_o"].norm().item() + CALIBRATION_EPS)
        )
        ratio_p = float(
            gradients["g_mag_p"].norm().item()
            / (gradients["g_dir_p"].norm().item() + CALIBRATION_EPS)
        )
        if not math.isfinite(ratio_o) or not math.isfinite(ratio_p):
            raise ActionPredictiveObjectiveError("calibration ratio is non-finite")
        ratios_o.append(ratio_o)
        ratios_p.append(ratio_p)
        rows.append(
            {
                "frame_id": frame.frame_id,
                "openvla": {
                    "loss_mag": _scalar(losses.loss_mag_o, label="L_mag O"),
                    "loss_dir": _scalar(losses.loss_dir_o, label="L_dir O"),
                    "g_mag": _gradient_metrics(gradients["g_mag_o"]),
                    "g_dir": _gradient_metrics(gradients["g_dir_o"]),
                    "gradient_cosine": _gradient_cosine(
                        gradients["g_mag_o"], gradients["g_dir_o"]
                    ),
                    "norm_ratio": ratio_o,
                },
                "pi05": {
                    "loss_mag": _scalar(losses.loss_mag_p, label="L_mag P"),
                    "loss_dir": _scalar(losses.loss_dir_p, label="L_dir P"),
                    "g_mag": _gradient_metrics(gradients["g_mag_p"]),
                    "g_dir": _gradient_metrics(gradients["g_dir_p"]),
                    "gradient_cosine": _gradient_cosine(
                        gradients["g_mag_p"], gradients["g_dir_p"]
                    ),
                    "norm_ratio": ratio_p,
                },
            }
        )
        del losses, image, gradients
    if not torch.equal(before, parameter.detach()):
        raise ActionPredictiveObjectiveError("calibration mutated texture parameter")
    ratio_o = float(np.median(np.asarray(ratios_o, dtype=np.float64)))
    ratio_p = float(np.median(np.asarray(ratios_p, dtype=np.float64)))
    lambda_dir = math.sqrt(ratio_o * ratio_p)
    if not math.isfinite(lambda_dir) or lambda_dir <= 0:
        raise ActionPredictiveObjectiveError("calibrated lambda_dir is unstable")
    return {
        "schema_version": "phase3_lambda_calibration_v1",
        "status": "PASS",
        "method": "sqrt(median_frame_norm_ratio_openvla * median_frame_norm_ratio_pi05)",
        "epsilon": CALIBRATION_EPS,
        "openvla_median_norm_ratio": ratio_o,
        "pi05_median_norm_ratio": ratio_p,
        "lambda_dir": lambda_dir,
        "frames": rows,
        "texture_parameter_unchanged": True,
    }


def _component_batch_diagnostic(
    sums: dict[str, torch.Tensor], count: int
) -> dict[str, Any]:
    values = {name: value / count for name, value in sums.items()}
    return {
        "openvla": {
            "g_mag": _gradient_metrics(values["g_mag_o"]),
            "g_dir": _gradient_metrics(values["g_dir_o"]),
            "cosine": _gradient_cosine(values["g_mag_o"], values["g_dir_o"]),
        },
        "pi05": {
            "g_mag": _gradient_metrics(values["g_mag_p"]),
            "g_dir": _gradient_metrics(values["g_dir_p"]),
            "cosine": _gradient_cosine(values["g_mag_p"], values["g_dir_p"]),
        },
    }


def train_action_predictive_gradient_ensemble(
    *,
    renderer: Any,
    frames: Sequence[ActionPredictiveTrainingFrame],
    forward_frame: Callable[
        [ActionPredictiveTrainingFrame], tuple[ActionPredictiveLosses, torch.Tensor]
    ],
    lambda_dir: float,
    iterations: int,
    requested_batch_size: int,
    pgd_step: float,
    seed: int,
    component_diagnostic_iterations: Sequence[int] = (0, 50, 100, 250, 499),
    metrics_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Train with the unchanged native-GE aggregation and one sign update."""

    if iterations < 1 or requested_batch_size < 1 or not frames:
        raise ActionPredictiveObjectiveError("invalid training loop dimensions")
    parameter = renderer.adv_noise
    batch_size = min(requested_batch_size, len(frames))
    rng = np.random.default_rng(seed)
    diagnostic_steps = set(component_diagnostic_iterations)
    history: list[dict[str, Any]] = []
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
    for iteration in range(iterations):
        parameter.grad = None
        chosen = rng.choice(len(frames), size=batch_size, replace=False)
        accumulator = ModelGradientAccumulator()
        scalar_names = (
            "loss_o",
            "loss_p",
            "loss_mag_o",
            "loss_mag_p",
            "loss_dir_o",
            "loss_dir_p",
            "action_mse_o",
            "action_mse_p",
            "native_mse_o",
            "native_mse_p",
            "z_clean_o_norm",
            "z_clean_p_norm",
            "z_adv_o_norm",
            "z_adv_p_norm",
            "delta_z_o_norm",
            "delta_z_p_norm",
        )
        totals = {name: 0.0 for name in scalar_names}
        deltas_o = torch.zeros(ACTION_DIMENSION)
        deltas_p = torch.zeros(ACTION_DIMENSION)
        abs_deltas_o = torch.zeros(ACTION_DIMENSION)
        abs_deltas_p = torch.zeros(ACTION_DIMENSION)
        squared_deltas_o = torch.zeros(ACTION_DIMENSION)
        squared_deltas_p = torch.zeros(ACTION_DIMENSION)
        component_sums: dict[str, torch.Tensor] = {}
        diagnostic = iteration in diagnostic_steps
        for index in chosen:
            frame = frames[int(index)]
            losses, image = forward_frame(frame)
            if diagnostic:
                component_gradients = {
                    "g_mag_o": _model_gradient(
                        losses.loss_mag_o,
                        parameter,
                        label="OpenVLA magnitude frame",
                        retain_graph=True,
                    ),
                    "g_dir_o": _model_gradient(
                        losses.loss_dir_o,
                        parameter,
                        label="OpenVLA direction frame",
                        retain_graph=True,
                    ),
                    "g_mag_p": _model_gradient(
                        losses.loss_mag_p,
                        parameter,
                        label="PI0.5 magnitude frame",
                        retain_graph=True,
                    ),
                    "g_dir_p": _model_gradient(
                        losses.loss_dir_p,
                        parameter,
                        label="PI0.5 direction frame",
                        retain_graph=True,
                    ),
                }
                for name, gradient in component_gradients.items():
                    if name not in component_sums:
                        component_sums[name] = gradient.detach().clone()
                    else:
                        component_sums[name].add_(gradient.detach())
            g_o = _model_gradient(
                losses.loss_o,
                parameter,
                label="OpenVLA action-predictive frame",
                retain_graph=True,
            )
            g_p = _model_gradient(
                losses.loss_p,
                parameter,
                label="PI0.5 action-predictive frame",
                retain_graph=False,
            )
            accumulator.add(g_o, g_p)
            for name in scalar_names:
                totals[name] += _scalar(getattr(losses, name), label=name) / batch_size
            deltas_o += losses.delta_z_o.detach().cpu() / batch_size
            deltas_p += losses.delta_z_p.detach().cpu() / batch_size
            abs_deltas_o += losses.delta_z_o.detach().abs().cpu() / batch_size
            abs_deltas_p += losses.delta_z_p.detach().abs().cpu() / batch_size
            squared_deltas_o += losses.delta_z_o.detach().square().cpu() / batch_size
            squared_deltas_p += losses.delta_z_p.detach().square().cpu() / batch_size
            del losses, image, g_o, g_p
        ensemble = accumulator.finalize()
        parameter.grad = ensemble.gradient.detach().clone()
        _, change_linf = sign_pgd_update(parameter, step_size=pgd_step)
        maximum = validate_texture_budget(renderer)
        row: dict[str, Any] = {
            "iteration": iteration,
            "selected_frame_ids": [frames[int(i)].frame_id for i in chosen],
            "lambda_dir": lambda_dir,
            **totals,
            **ensemble.diagnostics,
            "o2_mse": totals["native_mse_o"],
            "p2_mse": totals["native_mse_p"],
            "openvla_action_coordinate_mse": totals["action_mse_o"],
            "pi05_action_coordinate_mse": totals["action_mse_p"],
            "texture_gradient_norm": ensemble.diagnostics["gradient_ensemble_l2_norm"],
            "parameter_change_linf": change_linf,
            "maximum_texture_perturbation": maximum,
            "texture_budget_respected": maximum <= float(renderer.epsilon) + 1e-6,
        }
        for prefix, delta, abs_delta, squared_delta in (
            ("openvla", deltas_o, abs_deltas_o, squared_deltas_o),
            ("pi05", deltas_p, abs_deltas_p, squared_deltas_p),
        ):
            for index, coordinate in enumerate(ACTION_NAMES):
                value = float(delta[index])
                row[f"{prefix}_delta_{coordinate}"] = value
                row[f"{prefix}_abs_delta_{coordinate}"] = float(abs_delta[index])
                row[f"{prefix}_squared_delta_{coordinate}"] = float(
                    squared_delta[index]
                )
        if diagnostic:
            row["component_gradient_diagnostic"] = _component_batch_diagnostic(
                component_sums, batch_size
            )
        history.append(row)
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
    return history
