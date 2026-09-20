from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import phase2_action_representation_extract as entrypoint


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

from phase2_action_representation import (  # noqa: E402
    ActionRepresentationError,
    extract_openvla_action_representation,
    extract_pi05_action_representation,
    extract_pi05_official_p2,
    midpoint_layer_index,
    pi05_p2_identity_metrics,
)


class _PassLayer(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class _ReusableOutputLayer(torch.nn.Module):
    """Model a CUDA Graph output buffer overwritten by the next invocation."""

    def __init__(self) -> None:
        super().__init__()
        self.output_buffer: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = value + 1
        if self.output_buffer is None:
            self.output_buffer = torch.empty_like(result)
        self.output_buffer.copy_(result)
        return self.output_buffer


class _OpenModel:
    def __init__(self) -> None:
        self.projector = torch.nn.Identity()
        layers = torch.nn.ModuleList([_PassLayer() for _ in range(32)])
        self.language_model = type(
            "LM", (), {"model": type("Core", (), {"layers": layers})()}
        )()

    def predict_action(self, **_kwargs):
        value = self.projector(torch.ones(1, 256, 4096))
        hidden = torch.zeros(1, 260, 4096)
        for layer in self.language_model.model.layers:
            hidden = layer(hidden)
        # Cached generation calls must not create another full-sequence capture.
        self.language_model.model.layers[15](torch.zeros(1, 1, 4096))
        assert value.shape[-1] == 4096
        return np.arange(7, dtype=np.float32)


def test_openvla_capture_uses_derived_midpoint_and_visual_slice() -> None:
    model = _OpenModel()
    result = extract_openvla_action_representation(
        model=model,
        model_inputs={"input_ids": torch.ones(1, 2, dtype=torch.long)},
        unnorm_key="fixture",
        deploy_action=lambda action: action,
    )
    assert result.projected.shape == (1, 256, 4096)
    assert result.deep.shape == (1, 256, 4096)
    assert torch.all(result.deep == 16)
    assert result.deep_identity.total_layers == 32
    assert result.deep_identity.zero_based_layer_index == 15
    assert result.deep_identity.one_based_layer_index == 16


class _Projector(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(value, (0, 896))


class _PiModel:
    def __init__(self) -> None:
        projector = _Projector()
        layers = torch.nn.ModuleList([_PassLayer() for _ in range(18)])
        language_model = type("LM", (), {"layers": layers})()
        model = type("Vision", (), {"multi_modal_projector": projector})()
        config = type(
            "Config",
            (),
            {"text_config": type("TextConfig", (), {"hidden_size": 2048})()},
        )()
        paligemma = type(
            "Pali",
            (),
            {"model": model, "language_model": language_model, "config": config},
        )()
        self.paligemma_with_expert = type("Wrapper", (), {"paligemma": paligemma})()


class _PiPolicy:
    def __init__(self, model: _PiModel) -> None:
        self.model = model

    def infer(self, _observation, *, noise):
        assert noise.shape == (10, 32)
        projector = (
            self.model.paligemma_with_expert.paligemma.model.multi_modal_projector
        )
        for _ in range(3):
            projector(torch.ones(1, 256, 1152))
        hidden = torch.zeros(1, 800, 2048)
        layers = self.model.paligemma_with_expert.paligemma.language_model.layers
        for layer in layers:
            hidden = layer(hidden)
        return {"actions": np.arange(70, dtype=np.float32).reshape(10, 7)}


class _OfficialP2Wrapper:
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :1, :1, :1].reshape(1, 1, 1).expand(1, 256, 2048)


class _OfficialP2Model:
    def __init__(self) -> None:
        self.paligemma_with_expert = _OfficialP2Wrapper()

    def _preprocess_observation(self, observation, *, train: bool):
        assert observation == "fixture-observation"
        assert train is False
        images = [torch.full((1, 3, 224, 224), value) for value in (3.0, 4.0, 5.0)]
        masks = [torch.ones(1, dtype=torch.bool) for _ in images]
        return images, masks, torch.zeros(1, 2), torch.ones(1, 2), torch.zeros(1, 32)


class _OverwritingPiPolicy(_PiPolicy):
    def __init__(self, model: _PiModel) -> None:
        super().__init__(model)
        layers = self.model.paligemma_with_expert.paligemma.language_model.layers
        layers[8] = _ReusableOutputLayer()
        self.calls = 0

    def infer(self, _observation, *, noise):
        assert noise.shape == (10, 32)
        self.calls += 1
        projector = (
            self.model.paligemma_with_expert.paligemma.model.multi_modal_projector
        )
        for _ in range(3):
            projector(torch.ones(1, 256, 1152))
        hidden = torch.full((1, 800, 2048), float(self.calls))
        layers = self.model.paligemma_with_expert.paligemma.language_model.layers
        for layer in layers:
            hidden = layer(hidden)
        return {"actions": np.arange(70, dtype=np.float32).reshape(10, 7)}


def test_pi05_uses_authoritative_p2_provider_and_midpoint_prefix() -> None:
    model = _PiModel()
    result = extract_pi05_action_representation(
        policy=_PiPolicy(model),
        model=model,
        raw_observation={},
        noise=np.zeros((10, 32), dtype=np.float32),
        authoritative_p2_provider=lambda: torch.full((1, 256, 2048), 2.0),
    )
    assert result.projected.shape == (1, 256, 2048)
    assert result.projected[0, 0, 0].item() == pytest.approx(2.0)
    assert result.deep.shape == (1, 256, 2048)
    assert torch.all(result.deep == 9)
    assert result.deep_identity.total_layers == 18
    assert result.deep_identity.zero_based_layer_index == 8
    assert np.array_equal(result.deployed_action, np.arange(7, dtype=np.float32))


def test_pi05_official_p2_uses_base_camera_embed_image_without_scaling() -> None:
    result = extract_pi05_official_p2(
        model=_OfficialP2Model(), observation="fixture-observation"
    )

    assert result.shape == (1, 256, 2048)
    assert torch.all(result == 3.0)


def test_pi05_p2_identity_has_no_manual_sqrt_scaling() -> None:
    authoritative = torch.randn(1, 256, 2048, dtype=torch.bfloat16)
    result = pi05_p2_identity_metrics(
        extractor=authoritative.clone(),
        embed_image=authoritative.clone(),
        prefix=authoritative.clone(),
    )

    assert result["identity_pass"] is True
    assert result["manual_scaling"] == "none"
    assert result["l2_norm"]["phase2_extractor"] == pytest.approx(
        result["l2_norm"]["direct_embed_image"]
    )

    with pytest.raises(ActionRepresentationError, match="identity mismatch"):
        pi05_p2_identity_metrics(
            extractor=authoritative / np.sqrt(2048),
            embed_image=authoritative,
            prefix=authoritative,
        )


def test_pi05_deep_capture_owns_storage_across_repeated_inference() -> None:
    model = _PiModel()
    policy = _OverwritingPiPolicy(model)
    noise = np.zeros((10, 32), dtype=np.float32)

    first = extract_pi05_action_representation(
        policy=policy,
        model=model,
        raw_observation={},
        noise=noise,
        authoritative_p2_provider=lambda: torch.ones(1, 256, 2048),
    )
    first_snapshot = first.deep.clone()
    repeated = extract_pi05_action_representation(
        policy=policy,
        model=model,
        raw_observation={},
        noise=noise,
        authoritative_p2_provider=lambda: torch.ones(1, 256, 2048),
    )

    assert torch.equal(first.deep, first_snapshot)
    assert not torch.equal(first.deep, repeated.deep)


def test_hooks_are_removed_after_failure() -> None:
    model = _OpenModel()
    model.predict_action = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    layer = model.language_model.model.layers[15]
    with pytest.raises(RuntimeError, match="boom"):
        extract_openvla_action_representation(
            model=model,
            model_inputs={},
            unnorm_key="fixture",
            deploy_action=lambda action: action,
        )
    assert not model.projector._forward_hooks
    assert not layer._forward_hooks


def test_midpoint_requires_a_real_tower() -> None:
    assert midpoint_layer_index(32) == 15
    assert midpoint_layer_index(18) == 8
    with pytest.raises(ActionRepresentationError):
        midpoint_layer_index(1)


def test_extraction_cli_separates_one_observation_smoke_from_formal() -> None:
    common = [
        "--model",
        "openvla",
        "--output-dir",
        "/tmp/out",
        "--collection-manifest",
        "/tmp/collection.json",
        "--shared-feature-root",
        "/tmp/shared",
        "--checkpoint",
        "/tmp/model",
    ]
    assert entrypoint._args(common).max_observations == 200
    assert entrypoint._args([*common, "--max-observations", "1"]).max_observations == 1
    with pytest.raises(SystemExit):
        entrypoint._args([*common, "--max-observations", "2"])
