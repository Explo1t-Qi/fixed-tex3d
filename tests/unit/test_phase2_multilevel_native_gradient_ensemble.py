from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

import phase2_multilevel_native_gradient_ensemble as multilevel  # noqa: E402
from phase2_multilevel_native_gradient_ensemble import (  # noqa: E402
    DualVLAMultiLevelNativeFeatureAdapter,
    FrozenMultiLevelNativeReference,
    MultiLevelNativeDisplacementLosses,
    MultiLevelNativeFeatures,
    MultiLevelNativeTrainingFrame,
    NativeGradientEnsembleError,
    extract_openvla_o1_o2_autograd,
    extract_pi05_p1_p2_autograd,
    multilevel_native_displacement_losses,
    train_multilevel_native_gradient_ensemble,
)


def _expanded_feature(source: torch.Tensor, width: int, scale: float) -> torch.Tensor:
    value = source.mean(dim=(1, 2, 3)) * scale
    return value[:, None, None].expand(-1, 256, width)


def _openvla_path(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    o1 = _expanded_feature(image, 1152, 1.0)
    o2 = o1.mean(dim=-1, keepdim=True).expand(-1, 256, 4096) * 2.0
    return o1, o2


def _pi05_path(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p1 = _expanded_feature(image, 1152, 3.0)
    p2 = p1.mean(dim=-1, keepdim=True).expand(-1, 256, 2048) * 2.0
    return p1, p2


def _adapter() -> DualVLAMultiLevelNativeFeatureAdapter:
    return DualVLAMultiLevelNativeFeatureAdapter(
        openvla_path=_openvla_path,
        pi05_path=_pi05_path,
        openvla_device="cpu",
        pi05_device="cpu",
    )


def test_clean_reference_shapes_detach_clone_and_exact_losses() -> None:
    sources = (
        torch.full((1, 256, 1152), 1.0, requires_grad=True),
        torch.full((1, 256, 4096), 2.0, requires_grad=True),
        torch.full((1, 256, 1152), 3.0, requires_grad=True),
        torch.full((1, 256, 2048), 4.0, requires_grad=True),
    )
    clean = FrozenMultiLevelNativeReference.from_tensors(*sources)
    assert [
        tuple(value.shape) for value in (clean.o1, clean.o2, clean.p1, clean.p2)
    ] == [
        (1, 256, 1152),
        (1, 256, 4096),
        (1, 256, 1152),
        (1, 256, 2048),
    ]
    assert all(
        not value.requires_grad for value in (clean.o1, clean.o2, clean.p1, clean.p2)
    )
    with torch.no_grad():
        for source in sources:
            source.zero_()
    assert [
        float(value.flatten()[0]) for value in (clean.o1, clean.o2, clean.p1, clean.p2)
    ] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    adversarial = MultiLevelNativeFeatures(
        o1=torch.full_like(clean.o1, 3.0, requires_grad=True),
        o2=torch.full_like(clean.o2, 5.0, requires_grad=True),
        p1=torch.full_like(clean.p1, 7.0, requires_grad=True),
        p2=torch.full_like(clean.p2, 9.0, requires_grad=True),
    )
    losses = multilevel_native_displacement_losses(adversarial, clean)
    assert losses.o1_mse.item() == pytest.approx(4.0)
    assert losses.o2_mse.item() == pytest.approx(9.0)
    assert losses.p1_mse.item() == pytest.approx(16.0)
    assert losses.p2_mse.item() == pytest.approx(25.0)
    assert losses.loss_o.item() == pytest.approx(-13.0)
    assert losses.loss_p.item() == pytest.approx(-41.0)
    assert losses.o1_to_o2_mse_ratio.item() == pytest.approx(4.0 / 9.0)
    assert losses.p1_to_p2_mse_ratio.item() == pytest.approx(16.0 / 25.0)


def test_four_levels_and_combined_model_losses_preserve_image_autograd() -> None:
    adapter = _adapter()
    clean_image = torch.zeros(1, 3, 512, 512, requires_grad=True)
    clean = adapter.clean_reference(clean_image)
    image = torch.ones(1, 3, 512, 512, requires_grad=True)
    losses, features = adapter.losses(image, clean)

    components = (losses.o1_mse, losses.o2_mse, losses.p1_mse, losses.p2_mse)
    gradients = [
        torch.autograd.grad(component, image, retain_graph=True)[0]
        for component in components
    ]
    gradients.extend(
        (
            torch.autograd.grad(losses.loss_o, image, retain_graph=True)[0],
            torch.autograd.grad(losses.loss_p, image)[0],
        )
    )

    assert [
        tuple(value.shape)
        for value in (features.o1, features.o2, features.p1, features.p2)
    ] == [
        (1, 256, 1152),
        (1, 256, 4096),
        (1, 256, 1152),
        (1, 256, 2048),
    ]
    assert all(
        value.requires_grad
        for value in (features.o1, features.o2, features.p1, features.p2)
    )
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        assert bool(torch.any(gradient != 0))
    assert clean_image.grad is None


def test_loss_is_unnormalized_equal_weight_hierarchical_sum() -> None:
    clean = FrozenMultiLevelNativeReference.from_tensors(
        torch.zeros(1, 256, 1152),
        torch.zeros(1, 256, 4096),
        torch.zeros(1, 256, 1152),
        torch.zeros(1, 256, 2048),
    )
    adversarial = MultiLevelNativeFeatures(
        torch.ones_like(clean.o1),
        torch.full_like(clean.o2, 10.0),
        torch.full_like(clean.p1, 2.0),
        torch.full_like(clean.p2, 20.0),
    )
    losses = multilevel_native_displacement_losses(adversarial, clean)

    assert losses.loss_o == -(1.0 + 100.0)
    assert losses.loss_p == -(4.0 + 400.0)
    assert losses.loss_o != -2 * losses.o1_mse
    assert losses.loss_p != -2 * losses.p1_mse


def test_multilevel_shapes_are_rejected_explicitly() -> None:
    with pytest.raises(RuntimeError, match="O1-S.*shape"):
        FrozenMultiLevelNativeReference.from_tensors(
            torch.zeros(1, 255, 1152),
            torch.zeros(1, 256, 4096),
            torch.zeros(1, 256, 1152),
            torch.zeros(1, 256, 2048),
        )


def test_multilevel_module_has_no_shared_mapping_dependency() -> None:
    source = Path(multilevel.__file__).read_text()

    assert "FrozenSharedCCAMapping" not in source
    assert "shared_feature_loss" not in source
    assert "mapping.npz" not in source


class _ExpandBranch(torch.nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.width = width
        self.scale = scale
        self.calls = 0

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = pixels.mean(dim=(1, 2, 3)) * self.scale
        return value[:, None, None].expand(-1, 256, self.width)


class _OpenVLAProjector(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=-1, keepdim=True).expand(-1, 256, 4096)


def test_openvla_extractor_returns_o1s_o2_from_one_forward_and_removes_hook() -> None:
    dino = _ExpandBranch(1024, 1.0)
    siglip = _ExpandBranch(1152, 2.0)
    model = SimpleNamespace(
        vision_backbone=SimpleNamespace(
            featurizer=dino,
            fused_featurizer=siglip,
        ),
        projector=_OpenVLAProjector(),
    )
    pixels = torch.ones(1, 6, 8, 8, requires_grad=True)

    o1, o2 = extract_openvla_o1_o2_autograd(model, pixels)
    o1_gradient = torch.autograd.grad(o1.square().mean(), pixels, retain_graph=True)[0]
    o2_gradient = torch.autograd.grad(o2.square().mean(), pixels)[0]

    assert dino.calls == 1
    assert siglip.calls == 1
    assert tuple(o1.shape) == (1, 256, 1152)
    assert tuple(o2.shape) == (1, 256, 4096)
    assert len(siglip._forward_hooks) == 0
    for gradient in (o1_gradient, o2_gradient):
        assert bool(torch.isfinite(gradient).all())
        assert bool(torch.any(gradient != 0))


class _PiProjector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, p1: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return p1.mean(dim=-1, keepdim=True).expand(-1, 256, 2048)


class _PiEmbedder:
    def __init__(self, projector: _PiProjector, mode: str = "normal") -> None:
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(multi_modal_projector=projector)
        )
        self.projector = projector
        self.mode = mode

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        p1 = _expanded_feature(image, 1152, 2.0)
        if self.mode == "none":
            return p1.mean(dim=-1, keepdim=True).expand(-1, 256, 2048)
        output = self.projector(p1)
        if self.mode == "twice":
            output = self.projector(p1)
        if self.mode == "raise":
            raise RuntimeError("synthetic embed failure")
        return output


class _PiModel:
    def __init__(self, mode: str = "normal") -> None:
        projector = _PiProjector()
        self.paligemma_with_expert = _PiEmbedder(projector, mode)

    @staticmethod
    def _preprocess_observation(observation: Any, train: bool):
        assert not train
        images = list(observation.images.values())
        masks = list(observation.image_masks.values())
        return images, masks, None, None, None


def _pi_observation(image: torch.Tensor) -> SimpleNamespace:
    images = OrderedDict(
        (
            ("base_0_rgb", image),
            ("left_wrist_0_rgb", torch.zeros_like(image)),
            ("right_wrist_0_rgb", torch.zeros_like(image)),
        )
    )
    masks = OrderedDict(
        (
            ("base_0_rgb", torch.tensor([True])),
            ("left_wrist_0_rgb", torch.tensor([True])),
            ("right_wrist_0_rgb", torch.tensor([False])),
        )
    )
    return SimpleNamespace(images=images, image_masks=masks)


def test_pi05_hook_captures_p1_and_p2_once_with_autograd_then_is_removed() -> None:
    model = _PiModel()
    image = torch.ones(1, 3, 224, 224, requires_grad=True)
    projector = model.paligemma_with_expert.projector

    p1, p2 = extract_pi05_p1_p2_autograd(model, _pi_observation(image))
    p1_gradient = torch.autograd.grad(p1.square().mean(), image, retain_graph=True)[0]
    p2_gradient = torch.autograd.grad(p2.square().mean(), image)[0]

    assert projector.calls == 1
    assert tuple(p1.shape) == (1, 256, 1152)
    assert tuple(p2.shape) == (1, 256, 2048)
    assert p1.requires_grad and p2.requires_grad
    assert len(projector._forward_pre_hooks) == 0
    for gradient in (p1_gradient, p2_gradient):
        assert bool(torch.isfinite(gradient).all())
        assert bool(torch.any(gradient != 0))


@pytest.mark.parametrize("mode", ["none", "twice", "raise"])
def test_pi05_hook_rejects_invalid_capture_counts_and_never_leaks(mode: str) -> None:
    model = _PiModel(mode)
    projector = model.paligemma_with_expert.projector
    observation = _pi_observation(torch.ones(1, 3, 224, 224, requires_grad=True))

    with pytest.raises((NativeGradientEnsembleError, RuntimeError)):
        extract_pi05_p1_p2_autograd(model, observation)
    assert len(projector._forward_pre_hooks) == 0

    model.paligemma_with_expert.mode = "normal"
    p1, p2 = extract_pi05_p1_p2_autograd(model, observation)
    assert tuple(p1.shape) == (1, 256, 1152)
    assert tuple(p2.shape) == (1, 256, 2048)
    assert len(projector._forward_pre_hooks) == 0


class _Renderer:
    def __init__(self) -> None:
        self.adv_noise = torch.nn.Parameter(torch.zeros(2))
        self.epsilon = 0.5


def _reference() -> FrozenMultiLevelNativeReference:
    return FrozenMultiLevelNativeReference.from_tensors(
        torch.zeros(1, 256, 1152),
        torch.zeros(1, 256, 4096),
        torch.zeros(1, 256, 1152),
        torch.zeros(1, 256, 2048),
    )


def _training_losses(
    loss_o: torch.Tensor, loss_p: torch.Tensor
) -> MultiLevelNativeDisplacementLosses:
    one = -loss_o / 2
    two = -loss_p / 2
    return MultiLevelNativeDisplacementLosses(
        loss_o=loss_o,
        loss_p=loss_p,
        o1_mse=one,
        o2_mse=one,
        p1_mse=two,
        p2_mse=two,
        o1_to_o2_mse_ratio=one / (one + 1e-12),
        p1_to_p2_mse_ratio=two / (two + 1e-12),
    )


def test_trainer_aggregates_models_then_updates_texture_once_per_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _Renderer()
    clean = _reference()
    frames = [
        MultiLevelNativeTrainingFrame("frame-0", 0, "pi-0", clean, 1.0),
        MultiLevelNativeTrainingFrame("frame-1", 1, "pi-1", clean, 2.0),
    ]
    update_count = 0
    original_update = multilevel.sign_pgd_update

    def counted_update(parameter: torch.Tensor, *, step_size: float):
        nonlocal update_count
        update_count += 1
        return original_update(parameter, step_size=step_size)

    monkeypatch.setattr(multilevel, "sign_pgd_update", counted_update)
    pre_update_parameters: list[torch.Tensor] = []

    def forward(frame: MultiLevelNativeTrainingFrame):
        pre_update_parameters.append(renderer.adv_noise.detach().clone())
        image = renderer.adv_noise * float(frame.payload) + 1.0
        loss_o = (image * torch.tensor([1.0, 2.0])).sum()
        loss_p = (image * torch.tensor([3.0, 1.0])).sum()
        return _training_losses(loss_o, loss_p), image

    metrics_path = tmp_path / "step_metrics.jsonl"
    history = train_multilevel_native_gradient_ensemble(
        renderer=renderer,
        frames=frames,
        forward_frame=forward,
        iterations=3,
        requested_batch_size=10,
        pgd_step=0.05,
        seed=7,
        metrics_path=metrics_path,
    )

    assert update_count == 3
    assert len(pre_update_parameters) == 6
    for offset in range(0, 6, 2):
        assert torch.equal(
            pre_update_parameters[offset], pre_update_parameters[offset + 1]
        )
    required = {
        "o1_mse",
        "o2_mse",
        "p1_mse",
        "p2_mse",
        "loss_o",
        "loss_p",
        "o1_to_o2_mse_ratio",
        "p1_to_p2_mse_ratio",
        "g_o_l2_norm",
        "g_p_l2_norm",
        "g_o_mean_abs",
        "g_p_mean_abs",
        "raw_gradient_cosine",
        "g_o_normalized_l2_norm",
        "g_p_normalized_l2_norm",
        "normalized_gradient_cosine",
        "gradient_ensemble_l2_norm",
        "texture_gradient_norm",
        "parameter_change_linf",
        "maximum_texture_perturbation",
        "texture_budget_respected",
    }
    assert all(required <= row.keys() for row in history)
    assert all(row["o1_to_o2_mse_ratio"] == pytest.approx(1.0) for row in history)
    assert all(row["p1_to_p2_mse_ratio"] == pytest.approx(1.0) for row in history)
    serialized = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert serialized == history


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_multilevel_model_losses_cross_devices_to_texture_parameter() -> None:
    adapter = DualVLAMultiLevelNativeFeatureAdapter(
        openvla_path=_openvla_path,
        pi05_path=_pi05_path,
        openvla_device="cuda:0",
        pi05_device="cuda:1",
    )
    clean = adapter.clean_reference(torch.zeros(1, 3, 512, 512, device="cuda:0"))
    texture = torch.nn.Parameter(torch.tensor(0.25, device="cuda:0"))
    image = texture.expand(1, 3, 512, 512)

    losses, _ = adapter.losses(image, clean)
    g_o = torch.autograd.grad(losses.loss_o, texture, retain_graph=True)[0]
    g_p = torch.autograd.grad(losses.loss_p, texture)[0]

    assert g_o.device == texture.device and g_p.device == texture.device
    assert bool(torch.isfinite(g_o)) and bool(g_o != 0)
    assert bool(torch.isfinite(g_p)) and bool(g_p != 0)
