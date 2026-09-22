from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

import phase3_action_predictive_objective as phase3  # noqa: E402
from scripts import phase3_action_aware_optimization as phase3_entrypoint  # noqa: E402
from phase2_native_gradient_ensemble import NativeFeatures  # noqa: E402
from phase3_action_predictive_objective import (  # noqa: E402
    ActionPredictiveTrainingFrame,
    DualVLAActionPredictiveAdapter,
    FrozenActionLinearProbe,
    FrozenActionPredictiveReference,
    FrozenProbeArtifact,
    action_predictive_losses,
    calibrate_lambda_dir,
    load_frozen_primary_probes,
    pi05_preprocessing_input_difference,
    validate_pi05_probe_runtime_identity,
    train_action_predictive_gradient_ensemble,
)


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "phase2b"
    for model, node, width in (("openvla", "o2", 4096), ("pi05", "p2", 2048)):
        directory = root / model / node
        directory.mkdir(parents=True)
        torch.save(torch.ones(7, width), directory / "W.pt")
        torch.save({}, directory / "probe.pt")
        (directory / "metrics.json").write_text("{}\n")
        np.savez(
            directory / "action_stats.npz",
            mean=np.zeros(7, dtype=np.float32),
            std=np.ones(7, dtype=np.float32),
            safe_std=np.ones(7, dtype=np.float32),
            near_constant=np.zeros(7, dtype=bool),
        )
    metadata = {
        "schema_version": phase3.PHASE2B_SCHEMA,
        "status": "PHASE_2B_PRIMARY_FROZEN",
        "seed": 7,
        "split_rule": "pilot-v0.2-c5-split-v1",
        "probe": {
            "architecture": "Linear(D,7,bias=False)",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "steps": 2000,
            "action_std_epsilon": 1e-6,
            "probe_reg": 1e-4,
        },
        "models": {
            "pi05": {
                "representation_nodes": {
                    "projected": {
                        "definition_id": "pi05_p2_embed_image_no_manual_scaling_v2"
                    }
                }
            }
        },
        "provenance": {"openvla_W_unchanged": True},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    inventory = {
        str(path.relative_to(root)): {"sha256": _file_hash(path)}
        for path in root.rglob("*")
        if path.is_file()
    }
    (root / "artifact_inventory.json").write_text(json.dumps(inventory))
    return root


def test_primary_probe_artifact_loads_with_hash_validation(tmp_path: Path) -> None:
    root = _probe_artifact(tmp_path)
    loaded = load_frozen_primary_probes(root, openvla_device="cpu", pi05_device="cpu")

    assert loaded.openvla.weight.shape == (7, 4096)
    assert loaded.pi05.weight.shape == (7, 2048)
    assert len(loaded.hashes) == 8

    (root / "openvla/o2/metrics.json").write_text('{"changed": true}\n')
    with pytest.raises(phase3.ActionPredictiveObjectiveError, match="hash mismatch"):
        load_frozen_primary_probes(root, openvla_device="cpu", pi05_device="cpu")


class _FakeImageEmbedder:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        self.inputs.append(image)
        return image.mean(dim=(1, 2, 3), keepdim=True).reshape(-1, 1, 1).expand(
            -1, 256, 2048
        )


class _FakePi05Model:
    def __init__(self) -> None:
        self.paligemma_with_expert = _FakeImageEmbedder()


def test_pi05_probe_runtime_identity_uses_same_preprocessed_tensor() -> None:
    preprocessed = torch.ones(1, 3, 224, 224, dtype=torch.bfloat16)
    model = _FakePi05Model()
    result = validate_pi05_probe_runtime_identity(
        model=model, preprocessed_base_image=preprocessed
    )
    assert result["identity_pass"] is True
    assert len(model.paligemma_with_expert.inputs) == 2
    assert all(
        value is preprocessed for value in model.paligemma_with_expert.inputs
    )
    assert result["comparison_input"] == "same_preprocessed_base_0_rgb_tensor"
    assert result["definition_id"] == "pi05_p2_embed_image_no_manual_scaling_v2"
    assert result["comparisons_to_embed_image"]["phase2_extractor"][
        "max_abs_difference"
    ] == 0.0


def test_pi05_preprocessing_gap_is_diagnostic_not_identity_failure() -> None:
    official = torch.ones(1, 3, 224, 224)
    differentiable = official * 0.75
    diagnostic = pi05_preprocessing_input_difference(official, differentiable)
    assert diagnostic["max_abs_difference"] == pytest.approx(0.25)
    assert diagnostic["mean_abs_difference"] == pytest.approx(0.25)
    assert diagnostic["relative_l2_error"] == pytest.approx(0.25)
    assert diagnostic["hard_gate"] is False
    assert validate_pi05_probe_runtime_identity(
        model=_FakePi05Model(), preprocessed_base_image=differentiable
    )["identity_pass"] is True


def test_pi05_probe_runtime_identity_rejects_fixed_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakePi05Model()
    preprocessed = torch.ones(1, 3, 224, 224, dtype=torch.bfloat16)
    monkeypatch.setattr(
        phase3,
        "extract_pi05_p2_from_preprocessed_base_image",
        lambda model, image: model.paligemma_with_expert.embed_image(image)
        / np.sqrt(2048),
    )

    with pytest.raises(
        phase3.ActionPredictiveObjectiveError, match="probe/runtime P2 identity"
    ):
        validate_pi05_probe_runtime_identity(
            model=model, preprocessed_base_image=preprocessed
        )


def test_phase3_wrapper_fixes_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_main(arguments: list[str]) -> int:
        captured.extend(arguments)
        return 0

    monkeypatch.setattr(phase3_entrypoint, "optimization_main", fake_main)

    assert phase3_entrypoint.main(["--output-dir", "/tmp/fresh"]) == 0
    assert captured[:2] == [
        "--objective",
        "action_predictive_gradient_ensemble",
    ]
    with pytest.raises(ValueError, match="fixes the objective"):
        phase3_entrypoint.main(["--objective", "shared_cca"])


def _probe(model: str, width: int, *, scale: float = 1.0):
    weight = torch.zeros(7, width)
    weight[:, :7] = torch.eye(7) * scale
    return FrozenActionLinearProbe(
        weight,
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
        safe_action_std=torch.ones(7),
        model=model,
        node="o2" if model == "openvla" else "p2",
    )


def _features(value: torch.Tensor) -> NativeFeatures:
    batch = value.shape[0]
    o2 = torch.zeros(batch, 256, 4096, device=value.device)
    p2 = torch.zeros(batch, 256, 2048, device=value.device)
    o2[:, :, :7] = value[:, None, :]
    p2[:, :, :7] = (2 * value)[:, None, :]
    return NativeFeatures(o2=o2, p2=p2)


def _reference(value: torch.Tensor | None = None):
    if value is None:
        value = torch.ones(1, 7)
    features = _features(value)
    openvla = _probe("openvla", 4096)
    pi05 = _probe("pi05", 2048)
    return (
        FrozenActionPredictiveReference.from_tensors(
            o2=features.o2,
            p2=features.p2,
            z_o=openvla(features.o2),
            z_p=pi05(features.p2),
        ),
        openvla,
        pi05,
    )


def test_mapping_mean_pools_tokens_and_freezes_weight() -> None:
    probe = _probe("openvla", 4096)
    features = torch.zeros(2, 256, 4096, requires_grad=True)
    features.data[0, :, :7] = torch.arange(7)
    features.data[1, :128, :7] = 2.0

    coordinates = probe(features)

    assert coordinates.shape == (2, 7)
    assert torch.equal(coordinates[0], torch.arange(7, dtype=torch.float32))
    assert torch.equal(coordinates[1], torch.ones(7))
    assert list(probe.parameters()) == []
    coordinates.sum().backward()
    assert features.grad is not None
    assert bool(torch.any(features.grad != 0))
    assert probe.weight.grad is None


def test_magnitude_direction_and_equal_coordinate_weighting() -> None:
    clean, probe_o, probe_p = _reference()
    adversarial = _features(torch.tensor([[2.0] + [1.0] * 6]))
    losses = action_predictive_losses(
        adversarial,
        clean,
        openvla_probe=probe_o,
        pi05_probe=probe_p,
        lambda_dir=0.25,
    )

    assert losses.action_mse_o.item() == pytest.approx(1.0 / 7.0)
    assert losses.action_mse_p.item() == pytest.approx(4.0 / 7.0)
    assert losses.loss_mag_o.item() == pytest.approx(-1.0 / 7.0)
    assert losses.loss_o.item() == pytest.approx(
        losses.loss_mag_o.item() + 0.25 * losses.loss_dir_o.item()
    )
    assert losses.loss_p.item() == pytest.approx(
        losses.loss_mag_p.item() + 0.25 * losses.loss_dir_p.item()
    )


@pytest.mark.parametrize(
    ("adv", "expected"),
    [
        (torch.ones(1, 7), 1.0),
        (torch.tensor([[0.0, 1, 0, 0, 0, 0, 0.0]]), 1 / np.sqrt(7)),
        (-torch.ones(1, 7), -1.0),
        (torch.zeros(1, 7), 0.0),
    ],
)
def test_direction_loss_geometry_and_small_norm_finiteness(
    adv: torch.Tensor, expected: float
) -> None:
    clean, probe_o, probe_p = _reference()
    losses = action_predictive_losses(
        _features(adv),
        clean,
        openvla_probe=probe_o,
        pi05_probe=probe_p,
        lambda_dir=1.0,
    )
    assert losses.loss_dir_o.item() == pytest.approx(expected, abs=1e-6)
    assert bool(torch.isfinite(losses.loss_dir_o))


def test_clean_reference_and_probe_stay_frozen_while_gradient_reaches_texture() -> None:
    clean, probe_o, probe_p = _reference()
    texture = torch.nn.Parameter(torch.full((1, 7), 0.25))
    before_o = probe_o.weight.clone()
    before_p = probe_p.weight.clone()
    losses = action_predictive_losses(
        _features(texture),
        clean,
        openvla_probe=probe_o,
        pi05_probe=probe_p,
        lambda_dir=0.5,
    )

    gradient = torch.autograd.grad(losses.loss_o + losses.loss_p, texture)[0]

    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.any(gradient != 0))
    assert not clean.o2.requires_grad and not clean.p2.requires_grad
    assert not clean.z_o.requires_grad and not clean.z_p.requires_grad
    assert torch.equal(before_o, probe_o.weight)
    assert torch.equal(before_p, probe_p.weight)


class _NativeAdapter:
    @staticmethod
    def features(image: torch.Tensor) -> NativeFeatures:
        value = image.mean(dim=(1, 2, 3), keepdim=False)[:, None].repeat(1, 7)
        return _features(value)


def test_adapter_clean_cache_and_live_autograd() -> None:
    probes = FrozenProbeArtifact(
        root=Path("/unused"),
        metadata={},
        openvla=_probe("openvla", 4096),
        pi05=_probe("pi05", 2048),
        hashes={},
    )
    adapter = DualVLAActionPredictiveAdapter(
        native_adapter=_NativeAdapter(), probes=probes
    )
    clean_image = torch.zeros(1, 3, 4, 4, requires_grad=True)
    clean = adapter.clean_reference(clean_image)
    adv_image = torch.ones(1, 3, 4, 4, requires_grad=True)

    losses, _ = adapter.losses(adv_image, clean, lambda_dir=0.1)
    gradient = torch.autograd.grad(losses.loss_o + losses.loss_p, adv_image)[0]

    assert clean_image.grad is None
    assert bool(torch.isfinite(gradient).all()) and bool(torch.any(gradient != 0))


class _Renderer:
    def __init__(self) -> None:
        self.adv_noise = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
        self.epsilon = 0.5


def _synthetic_losses(parameter: torch.Tensor, scale: float):
    # Different nonlinear component directions keep calibration well-defined.
    mag_o = -((parameter - scale) ** 2).mean()
    dir_o = (parameter * torch.tensor([1.0, -2.0])).sum()
    mag_p = -((parameter + 2 * scale) ** 2).mean()
    dir_p = (parameter * torch.tensor([-3.0, 1.0])).sum()
    zero = parameter.sum() * 0
    delta = torch.cat((parameter, torch.zeros(5)))
    return phase3.ActionPredictiveLosses(
        loss_o=mag_o,
        loss_p=mag_p,
        loss_mag_o=mag_o,
        loss_mag_p=mag_p,
        loss_dir_o=dir_o,
        loss_dir_p=dir_p,
        action_mse_o=-mag_o,
        action_mse_p=-mag_p,
        native_mse_o=-mag_o,
        native_mse_p=-mag_p,
        z_clean_o_norm=zero + 1,
        z_clean_p_norm=zero + 1,
        z_adv_o_norm=zero + 1,
        z_adv_p_norm=zero + 1,
        delta_z_o_norm=delta.norm(),
        delta_z_p_norm=delta.norm(),
        delta_z_o=delta,
        delta_z_p=delta,
    )


def _frames() -> list[ActionPredictiveTrainingFrame]:
    clean, _, _ = _reference()
    return [
        ActionPredictiveTrainingFrame("f0", 0, "p0", clean, 1.0),
        ActionPredictiveTrainingFrame("f1", 1, "p1", clean, 2.0),
    ]


def test_lambda_calibration_is_read_only_and_uses_geometric_median_ratio() -> None:
    renderer = _Renderer()
    before = renderer.adv_noise.detach().clone()

    def forward(frame: ActionPredictiveTrainingFrame):
        return _synthetic_losses(renderer.adv_noise, frame.payload), renderer.adv_noise

    result = calibrate_lambda_dir(
        renderer=renderer, frames=_frames(), forward_frame=forward
    )

    ratios_o = [row["openvla"]["norm_ratio"] for row in result["frames"]]
    ratios_p = [row["pi05"]["norm_ratio"] for row in result["frames"]]
    expected = np.sqrt(np.median(ratios_o) * np.median(ratios_p))
    assert result["lambda_dir"] == pytest.approx(expected)
    assert result["texture_parameter_unchanged"]
    assert torch.equal(before, renderer.adv_noise)


def test_training_keeps_native_ge_hierarchy_and_updates_once_per_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    renderer = _Renderer()
    updates = 0
    original = phase3.sign_pgd_update

    def counted(parameter: torch.Tensor, *, step_size: float):
        nonlocal updates
        updates += 1
        return original(parameter, step_size=step_size)

    monkeypatch.setattr(phase3, "sign_pgd_update", counted)

    def forward(frame: ActionPredictiveTrainingFrame):
        base = _synthetic_losses(renderer.adv_noise, frame.payload)
        return phase3.ActionPredictiveLosses(
            **{
                **base.__dict__,
                "loss_o": base.loss_mag_o + 0.1 * base.loss_dir_o,
                "loss_p": base.loss_mag_p + 0.1 * base.loss_dir_p,
            }
        ), renderer.adv_noise

    history = train_action_predictive_gradient_ensemble(
        renderer=renderer,
        frames=_frames(),
        forward_frame=forward,
        lambda_dir=0.1,
        iterations=3,
        requested_batch_size=10,
        pgd_step=0.05,
        seed=7,
        component_diagnostic_iterations=(0, 2),
        metrics_path=tmp_path / "metrics.jsonl",
    )

    assert updates == 3
    assert len(history) == 3
    assert "component_gradient_diagnostic" in history[0]
    assert "component_gradient_diagnostic" not in history[1]
    assert "component_gradient_diagnostic" in history[2]
    assert all(row["texture_budget_respected"] for row in history)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 3


def test_calibration_rejects_zero_component_gradient() -> None:
    renderer = _Renderer()

    def forward(frame: ActionPredictiveTrainingFrame):
        del frame
        losses = _synthetic_losses(renderer.adv_noise, 1.0)
        zero = renderer.adv_noise.sum() * 0
        return phase3.ActionPredictiveLosses(
            **{**losses.__dict__, "loss_dir_p": zero}
        ), renderer.adv_noise

    with pytest.raises(phase3.ActionPredictiveObjectiveError, match="zero"):
        calibrate_lambda_dir(renderer=renderer, frames=_frames(), forward_frame=forward)
