from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

import phase2_native_gradient_ensemble as native  # noqa: E402
from phase2_native_gradient_ensemble import (  # noqa: E402
    DualVLANativeFeatureAdapter,
    FrozenNativeReference,
    ModelGradientAccumulator,
    NativeDisplacementLosses,
    NativeFeatures,
    NativeGradientEnsembleError,
    NativeTrainingFrame,
    native_displacement_losses,
    train_native_gradient_ensemble,
)


def _reference() -> FrozenNativeReference:
    return FrozenNativeReference.from_tensors(
        torch.zeros(1, 256, 4096, requires_grad=True),
        torch.zeros(1, 256, 2048, requires_grad=True),
    )


def test_clean_native_reference_is_detached_and_loss_keeps_adv_autograd() -> None:
    source_o = torch.ones(1, 256, 4096, requires_grad=True)
    source_p = torch.ones(1, 256, 2048, requires_grad=True)
    clean = FrozenNativeReference.from_tensors(source_o, source_p)
    with torch.no_grad():
        source_o.zero_()
        source_p.zero_()

    assert not clean.o2.requires_grad
    assert not clean.p2.requires_grad
    assert bool(torch.all(clean.o2 == 1))
    assert bool(torch.all(clean.p2 == 1))

    adv_o = torch.zeros_like(clean.o2, requires_grad=True)
    adv_p = torch.zeros_like(clean.p2, requires_grad=True)
    losses = native_displacement_losses(NativeFeatures(adv_o, adv_p), clean)
    assert losses.loss_o == -losses.o2_mse
    assert losses.loss_p == -losses.p2_mse
    torch.autograd.grad(losses.loss_o, adv_o)
    torch.autograd.grad(losses.loss_p, adv_p)
    assert source_o.grad is None
    assert source_p.grad is None


def test_frame_gradients_are_meaned_before_model_normalization() -> None:
    accumulator = ModelGradientAccumulator()
    accumulator.add(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 6.0]))
    accumulator.add(torch.tensor([5.0, 1.0]), torch.tensor([6.0, 2.0]))

    result = accumulator.finalize()

    assert torch.equal(result.g_o, torch.tensor([3.0, 2.0]))
    assert torch.equal(result.g_p, torch.tensor([4.0, 4.0]))
    assert torch.allclose(result.g_o_normalized, torch.tensor([1.2, 0.8]))
    assert torch.allclose(result.g_p_normalized, torch.tensor([1.0, 1.0]))


def test_implementation_does_not_normalize_per_frame() -> None:
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([9.0, 1.0])
    accumulator = ModelGradientAccumulator()
    accumulator.add(first, first)
    accumulator.add(second, second)

    result = accumulator.finalize()
    normalized_after_mean = torch.tensor([5.0, 0.5]) / 2.75
    normalized_per_frame_then_mean = (
        first / first.abs().mean() + second / second.abs().mean()
    ) / 2

    assert torch.allclose(result.g_o_normalized, normalized_after_mean)
    assert not torch.allclose(result.g_o_normalized, normalized_per_frame_then_mean)


def test_models_are_normalized_separately_before_ensemble() -> None:
    accumulator = ModelGradientAccumulator()
    accumulator.add(torch.tensor([100.0, 100.0]), torch.tensor([-1.0, 1.0]))

    result = accumulator.finalize()

    assert torch.allclose(result.g_o_normalized, torch.tensor([1.0, 1.0]))
    assert torch.allclose(result.g_p_normalized, torch.tensor([-1.0, 1.0]))
    assert torch.allclose(result.gradient, torch.tensor([0.0, 1.0]))
    assert result.diagnostics["g_o_mean_abs"] == 100.0
    assert result.diagnostics["g_p_mean_abs"] == 1.0


def test_gradient_ensemble_is_mean_of_normalized_model_gradients() -> None:
    accumulator = ModelGradientAccumulator()
    accumulator.add(torch.tensor([2.0, -1.0]), torch.tensor([1.0, 4.0]))

    result = accumulator.finalize()

    assert torch.allclose(
        result.gradient,
        (result.g_o_normalized + result.g_p_normalized) / 2,
    )
    for key in (
        "g_o_l2_norm",
        "g_p_l2_norm",
        "g_o_mean_abs",
        "g_p_mean_abs",
        "raw_gradient_cosine",
        "g_o_normalized_l2_norm",
        "g_p_normalized_l2_norm",
        "normalized_gradient_cosine",
        "gradient_ensemble_l2_norm",
    ):
        assert key in result.diagnostics


def test_native_trainer_uses_separate_autograd_grad_calls() -> None:
    source = Path(native.__file__).read_text()

    assert "torch.autograd.grad(" in source
    assert ".backward(" not in source
    assert "(g_o_normalized + g_p_normalized) / 2.0" in source


class _Renderer:
    def __init__(self) -> None:
        self.adv_noise = torch.nn.Parameter(torch.zeros(2))
        self.epsilon = 0.5


def _training_frames() -> list[NativeTrainingFrame]:
    clean = _reference()
    return [
        NativeTrainingFrame("frame-0", 0, "pi-template-0", clean, 1.0),
        NativeTrainingFrame("frame-1", 1, "pi-template-1", clean, 2.0),
    ]


def _losses(loss_o: torch.Tensor, loss_p: torch.Tensor) -> NativeDisplacementLosses:
    return NativeDisplacementLosses(
        loss_o=loss_o,
        loss_p=loss_p,
        o2_mse=-loss_o,
        p2_mse=-loss_p,
    )


def test_outer_iteration_makes_exactly_one_texture_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _Renderer()
    updates = 0
    original_update = native.sign_pgd_update

    def counted_update(parameter: torch.Tensor, *, step_size: float):
        nonlocal updates
        updates += 1
        return original_update(parameter, step_size=step_size)

    monkeypatch.setattr(native, "sign_pgd_update", counted_update)
    frame_calls: list[tuple[str, torch.Tensor]] = []

    def forward(frame: NativeTrainingFrame):
        before = renderer.adv_noise.detach().clone()
        frame_calls.append((frame.frame_id, before))
        scale = float(frame.payload)
        image = renderer.adv_noise * scale + 1.0
        loss_o = (image * torch.tensor([1.0, 2.0])).sum()
        loss_p = (image * torch.tensor([3.0, 1.0])).sum()
        return _losses(loss_o, loss_p), image

    history = train_native_gradient_ensemble(
        renderer=renderer,
        frames=_training_frames(),
        forward_frame=forward,
        iterations=3,
        requested_batch_size=10,
        pgd_step=0.05,
        seed=7,
        metrics_path=tmp_path / "metrics.jsonl",
    )

    assert updates == 3
    assert len(frame_calls) == 6
    for offset in range(0, len(frame_calls), 2):
        assert torch.equal(frame_calls[offset][1], frame_calls[offset + 1][1])
    assert len(history) == 3
    assert all(row["parameter_change_linf"] == pytest.approx(0.05) for row in history)
    assert all(row["texture_budget_respected"] for row in history)
    serialized = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert serialized == history


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_frame_gradients_fail_explicitly(invalid: float) -> None:
    with pytest.raises(NativeGradientEnsembleError, match="non-finite"):
        ModelGradientAccumulator().add(torch.tensor([invalid]), torch.tensor([1.0]))


def test_zero_batch_and_missing_model_gradients_fail_explicitly() -> None:
    cancelled = ModelGradientAccumulator()
    cancelled.add(torch.tensor([1.0]), torch.tensor([1.0]))
    cancelled.add(torch.tensor([-1.0]), torch.tensor([1.0]))
    with pytest.raises(NativeGradientEnsembleError, match="OpenVLA batch.*zero"):
        cancelled.finalize()

    renderer = _Renderer()

    def missing(frame: NativeTrainingFrame):
        del frame
        independent = torch.tensor(1.0, requires_grad=True)
        dependent = renderer.adv_noise.sum()
        return _losses(independent, dependent), renderer.adv_noise + 1.0

    with pytest.raises(NativeGradientEnsembleError, match="OpenVLA.*missing"):
        train_native_gradient_ensemble(
            renderer=renderer,
            frames=_training_frames(),
            forward_frame=missing,
            iterations=1,
            requested_batch_size=2,
            pgd_step=0.05,
            seed=7,
        )


def _feature_path(width: int, scale: float):
    def path(image: torch.Tensor) -> torch.Tensor:
        value = image.mean(dim=(1, 2, 3)) * scale
        return value[:, None, None].expand(-1, 256, width)

    return path


def test_native_adapter_preserves_same_image_autograd_without_mapping() -> None:
    adapter = DualVLANativeFeatureAdapter(
        openvla_path=_feature_path(4096, 1.0),
        pi05_path=_feature_path(2048, 2.0),
        openvla_device="cpu",
        pi05_device="cpu",
    )
    clean_image = torch.zeros(1, 3, 512, 512, requires_grad=True)
    clean = adapter.clean_reference(clean_image)
    adversarial = torch.ones(1, 3, 512, 512, requires_grad=True)

    losses, features = adapter.losses(adversarial, clean)
    grad_o = torch.autograd.grad(losses.loss_o, adversarial, retain_graph=True)[0]
    grad_p = torch.autograd.grad(losses.loss_p, adversarial)[0]

    assert features.o2.shape == (1, 256, 4096)
    assert features.p2.shape == (1, 256, 2048)
    assert clean_image.grad is None
    assert not clean.o2.requires_grad
    assert not clean.p2.requires_grad
    for gradient in (grad_o, grad_p):
        assert bool(torch.isfinite(gradient).all())
        assert bool(torch.any(gradient != 0))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_native_model_gradients_cross_devices_to_texture_parameter() -> None:
    adapter = DualVLANativeFeatureAdapter(
        openvla_path=_feature_path(4096, 1.0),
        pi05_path=_feature_path(2048, 2.0),
        openvla_device="cuda:0",
        pi05_device="cuda:1",
    )
    clean = adapter.clean_reference(torch.zeros(1, 3, 512, 512, device="cuda:0"))
    texture = torch.nn.Parameter(torch.tensor(0.25, device="cuda:0"))
    adversarial = texture.expand(1, 3, 512, 512)

    losses, _ = adapter.losses(adversarial, clean)
    grad_o = torch.autograd.grad(losses.loss_o, texture, retain_graph=True)[0]
    grad_p = torch.autograd.grad(losses.loss_p, texture)[0]

    assert grad_o.device == texture.device
    assert grad_p.device == texture.device
    assert bool(torch.isfinite(grad_o)) and bool(grad_o != 0)
    assert bool(torch.isfinite(grad_p)) and bool(grad_p != 0)
