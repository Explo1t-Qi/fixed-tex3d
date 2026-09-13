from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def test_frozen_protocol_rejects_changes() -> None:
    SharedOptimizationProtocol().validate_frozen_pilot()
    with pytest.raises(Phase2SharedOptimizationError, match="frozen"):
        SharedOptimizationProtocol(attack_iterations=501).validate_frozen_pilot()


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


def test_server_training_entrypoint_freezes_protocol_and_artifacts() -> None:
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
    ):
        assert required in source
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
