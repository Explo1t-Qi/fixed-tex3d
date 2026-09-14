from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from scripts import phase2_shared_optimization as training_entrypoint
from scripts import phase2_source_eval


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROBOT_ROOT))

from phase2_shared_optimization import (  # noqa: E402
    FrozenCleanReference,
    PairedTrial,
    Phase2SharedOptimizationError,
    SharedOptimizationProtocol,
    SharedTrainingFrame,
    paired_source_metrics,
    train_shared_texture,
    validate_texture_budget,
)


class _Renderer:
    def __init__(self, epsilon: float = 0.5):
        self.epsilon = epsilon
        self.adv_noise = torch.nn.Parameter(torch.zeros(2))


def _clean() -> FrozenCleanReference:
    return FrozenCleanReference.from_tensors(
        torch.ones(1, 256, 262, requires_grad=True),
        torch.ones(1, 256, 262, requires_grad=True),
    )


def _frames() -> list[SharedTrainingFrame]:
    return [
        SharedTrainingFrame(f"frame-{i}", i, f"pi-template-{i}", _clean(), i + 1.0)
        for i in range(2)
    ]


def _result(loss: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        loss=loss,
        shared_mse=-loss,
        o2_mse=-loss * 2,
        p2_mse=-loss * 3,
        displacement_cosine_mean=loss * 0,
        o2_to_p2_mse_ratio=(-loss * 2) / (-loss * 3),
    )


def test_optimization_parameters_can_change_independently_or_together() -> None:
    baseline = SharedOptimizationProtocol()
    baseline.validate_frozen_pilot()
    baseline_config = baseline.validate_training_configuration()
    assert baseline_config["changes_from_baseline"] == {}

    step_size = SharedOptimizationProtocol(pgd_step=0.01)
    step_config = step_size.validate_training_configuration()
    assert step_config["changes_from_baseline"] == {
        "pgd_step": {"baseline": 0.05, "actual": 0.01}
    }

    iterations = SharedOptimizationProtocol(attack_iterations=5000)
    iteration_config = iterations.validate_training_configuration()
    assert iteration_config["changes_from_baseline"] == {
        "attack_iterations": {"baseline": 500, "actual": 5000}
    }

    combined = SharedOptimizationProtocol(attack_iterations=5000, pgd_step=0.01)
    combined_config = combined.validate_training_configuration()
    assert set(combined_config["changes_from_baseline"]) == {
        "attack_iterations",
        "pgd_step",
    }
    assert combined_config["adjustable_parameters"]["attack_iterations"] == {
        "baseline": 500,
        "actual": 5000,
        "changed": True,
    }
    assert combined_config["adjustable_parameters"]["pgd_step"] == {
        "baseline": 0.05,
        "actual": 0.01,
        "changed": True,
    }


def test_optimization_parameter_validation_keeps_other_protocol_fields_frozen() -> None:
    with pytest.raises(Phase2SharedOptimizationError, match="frozen protocol"):
        SharedOptimizationProtocol(seed=8).validate_training_configuration()
    with pytest.raises(Phase2SharedOptimizationError, match="positive"):
        SharedOptimizationProtocol(
            attack_iterations=0
        ).validate_training_configuration()
    with pytest.raises(Phase2SharedOptimizationError, match="finite and positive"):
        SharedOptimizationProtocol(
            pgd_step=float("nan")
        ).validate_training_configuration()


def test_training_cli_accepts_combined_iterations_and_step_size() -> None:
    args = training_entrypoint._args(
        [
            "--output-dir",
            "/tmp/output",
            "--shared-feature-root",
            "/tmp/shared",
            "--mapping-dir",
            "/tmp/mapping",
            "--openpi-root",
            "/tmp/openpi",
            "--openvla-checkpoint",
            "/tmp/openvla-checkpoint",
            "--pi05-checkpoint",
            "/tmp/pi05-checkpoint",
            "--libero-root",
            "/tmp/libero",
            "--iterations",
            "5000",
            "--pgd-step",
            "0.01",
        ]
    )

    assert args.iterations == 5000
    assert args.pgd_step == 0.01
    assert not hasattr(args, "diagnostic_kind")


def test_native_mode_does_not_require_or_resolve_shared_artifacts(
    tmp_path: Path,
) -> None:
    existing = {}
    for name in (
        "openpi",
        "openvla-checkpoint",
        "pi05-checkpoint",
        "libero",
    ):
        existing[name] = tmp_path / name
        existing[name].mkdir()
    args = training_entrypoint._args(
        [
            "--objective",
            "native_gradient_ensemble",
            "--output-dir",
            str(tmp_path / "output"),
            "--shared-feature-root",
            str(tmp_path / "missing-shared"),
            "--mapping-dir",
            str(tmp_path / "missing-mapping"),
            "--openpi-root",
            str(existing["openpi"]),
            "--openvla-checkpoint",
            str(existing["openvla-checkpoint"]),
            "--pi05-checkpoint",
            str(existing["pi05-checkpoint"]),
            "--libero-root",
            str(existing["libero"]),
        ]
    )

    paths = training_entrypoint._validate_paths(args)

    assert "shared" not in paths
    assert "mapping" not in paths


def test_native_source_paths_exclude_shared_feature_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot_root = tmp_path / "openvla/experiments/robot"
    openvla_root = tmp_path / "openvla"
    openpi_root = tmp_path / "openpi"
    for directory in (
        robot_root / "libero",
        openvla_root,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(training_entrypoint, "ROBOT_ROOT", robot_root)
    monkeypatch.setattr(training_entrypoint, "OPENVLA_ROOT", openvla_root)
    monkeypatch.setattr(sys, "path", list(sys.path))

    training_entrypoint._add_native_source_paths(openpi_root)

    assert str(openpi_root / "src") in sys.path
    assert not any("shared-feature" in path for path in sys.path[:5])


def test_shared_cca_mode_still_requires_shared_artifacts(tmp_path: Path) -> None:
    existing = {}
    for name in (
        "openpi",
        "openvla-checkpoint",
        "pi05-checkpoint",
        "libero",
    ):
        existing[name] = tmp_path / name
        existing[name].mkdir()
    args = training_entrypoint._args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--openpi-root",
            str(existing["openpi"]),
            "--openvla-checkpoint",
            str(existing["openvla-checkpoint"]),
            "--pi05-checkpoint",
            str(existing["pi05-checkpoint"]),
            "--libero-root",
            str(existing["libero"]),
        ]
    )

    with pytest.raises(ValueError, match="shared_cca requires"):
        training_entrypoint._validate_paths(args)


def test_checkpoint_step_parser_is_explicit_and_bounded() -> None:
    assert training_entrypoint._parse_checkpoint_steps("", iterations=500) == ()
    assert training_entrypoint._parse_checkpoint_steps(
        "500,100,500", iterations=500
    ) == (100, 500)
    with pytest.raises(ValueError, match="within"):
        training_entrypoint._parse_checkpoint_steps("501", iterations=500)
    with pytest.raises(ValueError, match="integers"):
        training_entrypoint._parse_checkpoint_steps("100,bad", iterations=500)


def test_checkpoint_artifact_is_reloadable_and_self_describing(tmp_path: Path) -> None:
    class CheckpointRenderer:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))

        def get_texture_param(self) -> torch.Tensor:
            return self.parameter

        @staticmethod
        def get_baked_adv_texture() -> torch.Tensor:
            return torch.full((1, 4, 5, 3), 0.5)

    row = {"iteration": 99, "shared_mse": 0.25}
    metadata = training_entrypoint._save_texture_checkpoint(
        renderer=CheckpointRenderer(),
        output=tmp_path,
        step=100,
        optimization_parameters={"iterations": 5000, "pgd_step": 0.01},
        row=row,
    )

    checkpoint = tmp_path / "checkpoints/step_000100"
    parameter = torch.load(
        checkpoint / "vertex_noise.pt", map_location="cpu", weights_only=True
    )
    image = Image.open(checkpoint / "attack_texture.png")
    on_disk = json.loads((checkpoint / "metadata.json").read_text())
    assert torch.equal(parameter, torch.tensor([1.0, -1.0]))
    assert image.size == (5, 4)
    assert metadata == on_disk
    assert on_disk["completed_step"] == 100
    assert on_disk["optimization_parameters"] == {
        "iterations": 5000,
        "pgd_step": 0.01,
    }
    assert on_disk["diagnostics"] == row


def test_clean_reference_is_detached_and_frame_templates_are_unique() -> None:
    source = torch.ones(1, 256, 262, requires_grad=True)
    clean = FrozenCleanReference.from_tensors(source, source)
    assert not clean.h_o.requires_grad
    assert not clean.h_p.requires_grad
    with torch.no_grad():
        source.zero_()
    assert bool(torch.all(clean.h_o == 1))
    renderer = _Renderer()
    duplicate = [
        SharedTrainingFrame("a", 0, "same", clean, 1.0),
        SharedTrainingFrame("b", 1, "same", clean, 2.0),
    ]
    with pytest.raises(Phase2SharedOptimizationError, match="own PI0Pytorch"):
        train_shared_texture(
            renderer=renderer,
            frames=duplicate,
            forward_frame=lambda frame: (
                _result(renderer.adv_noise.sum()),
                renderer.adv_noise,
            ),
            iterations=1,
            requested_batch_size=2,
            pgd_step=0.05,
            seed=7,
        )


def test_three_steps_use_mean_frame_loss_and_one_sign_update_each(
    tmp_path: Path,
) -> None:
    renderer = _Renderer()
    update_inputs: list[tuple[str, float]] = []

    def forward(frame: SharedTrainingFrame):
        image = renderer.adv_noise * float(frame.payload) + 1.0
        image.retain_grad()
        loss = -(image.square().mean())
        update_inputs.append((frame.frame_id, float(renderer.adv_noise[0].detach())))
        return _result(loss), image

    metrics_path = tmp_path / "step_metrics.jsonl"
    history = train_shared_texture(
        renderer=renderer,
        frames=_frames(),
        forward_frame=forward,
        iterations=3,
        requested_batch_size=20,
        pgd_step=0.05,
        seed=7,
        metrics_path=metrics_path,
    )
    assert len(update_inputs) == 6
    # Both frames in an iteration observe the same pre-update parameter.
    assert [pair[1] for pair in update_inputs] == pytest.approx(
        [0.0, 0.0, 0.05, 0.05, 0.1, 0.1]
    )
    assert torch.allclose(renderer.adv_noise, torch.full((2,), 0.15))
    assert all(row["parameter_change_linf"] == pytest.approx(0.05) for row in history)
    assert all(len(row["selected_frame_ids"]) == 2 for row in history)
    serialized = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert serialized == history


def test_budget_uses_tanh_parameterization() -> None:
    renderer = _Renderer(epsilon=0.25)
    with torch.no_grad():
        renderer.adv_noise.fill_(100.0)
    assert validate_texture_budget(renderer) == pytest.approx(0.25)


def test_paired_source_metrics_and_conditional_asr() -> None:
    metrics = paired_source_metrics(
        [
            PairedTrial(0, True, False),
            PairedTrial(1, True, True),
            PairedTrial(2, False, False),
            PairedTrial(3, False, True),
        ]
    )
    assert metrics["clean_success_rate"] == 0.5
    assert metrics["adversarial_success_rate"] == 0.5
    assert metrics["adversarial_failure_rate"] == 0.5
    assert metrics["conditional_asr"] == 0.5
    undefined = paired_source_metrics([PairedTrial(0, False, False)])
    assert undefined["conditional_asr"] is None


def test_paired_metrics_reject_duplicate_or_empty_trials() -> None:
    with pytest.raises(Phase2SharedOptimizationError, match="requires trials"):
        paired_source_metrics([])
    with pytest.raises(Phase2SharedOptimizationError, match="unique"):
        paired_source_metrics([PairedTrial(0, True, True), PairedTrial(0, True, False)])


def test_training_localizes_zero_gradient() -> None:
    renderer = _Renderer()

    def zero_image_gradient(frame: SharedTrainingFrame):
        image = renderer.adv_noise + 1
        return _result(image.sum() * 0), image

    with pytest.raises(Phase2SharedOptimizationError, match="image gradient is zero"):
        train_shared_texture(
            renderer=renderer,
            frames=_frames(),
            forward_frame=zero_image_gradient,
            iterations=1,
            requested_batch_size=2,
            pgd_step=0.05,
            seed=7,
        )


def test_server_training_entrypoint_parameterizes_optimizer_and_freezes_artifacts() -> (
    None
):
    source = (PROJECT_ROOT / "scripts/phase2_shared_optimization.py").read_text()
    for required in (
        "default=500",
        "default=0.05",
        "default=10",
        "default=20",
        "default=7",
        '"final_vertex_noise.pt"',
        '"final_attack_texture.png"',
        '"step_metrics.jsonl"',
        '"loss_history.npy"',
        '"training_summary.json"',
        '"--iterations"',
        '"--pgd-step"',
        '"--checkpoint-steps"',
        '"checkpoints"',
        '"run_configuration"',
        '"native_gradient_ensemble"',
    ):
        assert required in source
    assert '"--diagnostic-kind"' not in source
    assert "torch.optim" not in source
    assert "autograd.grad" not in source


def test_source_evaluator_is_paired_and_restores_runtime_texture() -> None:
    source = (PROJECT_ROOT / "scripts/phase2_source_eval.py").read_text()
    assert 'choices=("openvla", "pi05")' in source
    assert "clean_success = rollout(state_id)" in source
    assert "adversarial_success = rollout(state_id)" in source
    assert "temporary_runtime_texture(" in source
    assert "asset XML was not restored" in source
    assert "default=50" in source


def test_openvla_evaluator_converts_checkpoint_path_for_legacy_helper() -> None:
    config = phase2_source_eval._openvla_eval_config(Path("/tmp/openvla-checkpoint"))

    assert isinstance(config.pretrained_checkpoint, str)
    assert config.pretrained_checkpoint == "/tmp/openvla-checkpoint"
