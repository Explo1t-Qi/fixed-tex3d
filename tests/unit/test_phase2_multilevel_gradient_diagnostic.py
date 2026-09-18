from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

import phase2_multilevel_gradient_diagnostic as diagnostic  # noqa: E402
from phase2_multilevel_gradient_diagnostic import (  # noqa: E402
    ComponentGradientAccumulator,
    ComponentLosses,
    GradientDiagnosticFrame,
    MultiLevelGradientDiagnosticError,
    analyze_batch_gradients,
    cancellation_ratio,
    component_losses,
    component_texture_gradients,
    gradient_cosine,
    gradient_magnitude,
    load_texture_parameter,
    pairwise_cosines,
    run_gradient_decomposition,
)
from phase2_multilevel_native_gradient_ensemble import (  # noqa: E402
    FrozenMultiLevelNativeReference,
    MultiLevelNativeFeatures,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/phase2_multilevel_gradient_diagnostic.py"
)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "phase2_multilevel_gradient_diagnostic_entrypoint", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_components(value: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name in diagnostic.COMPONENTS}


def test_frame_gradients_are_meaned_raw_before_analysis() -> None:
    accumulator = ComponentGradientAccumulator()
    accumulator.add(
        {
            "O1": torch.tensor([1.0, 0.0]),
            "O2": torch.tensor([2.0, 2.0]),
            "P1": torch.tensor([-1.0, 3.0]),
            "P2": torch.tensor([4.0, -2.0]),
        }
    )
    accumulator.add(
        {
            "O1": torch.tensor([9.0, 1.0]),
            "O2": torch.tensor([4.0, 6.0]),
            "P1": torch.tensor([3.0, 5.0]),
            "P2": torch.tensor([2.0, 8.0]),
        }
    )

    gradients = accumulator.finalize()

    assert torch.equal(gradients["O1"], torch.tensor([5.0, 0.5]))
    assert torch.equal(gradients["O2"], torch.tensor([3.0, 4.0]))
    assert torch.equal(gradients["P1"], torch.tensor([1.0, 4.0]))
    assert torch.equal(gradients["P2"], torch.tensor([3.0, 3.0]))
    normalized_per_frame = (
        torch.tensor([1.0, 0.0]) / 0.5 + torch.tensor([9.0, 1.0]) / 5.0
    ) / 2
    assert not torch.allclose(gradients["O1"], normalized_per_frame)


def test_gradient_magnitude_is_exact() -> None:
    metrics = gradient_magnitude(torch.tensor([-3.0, 4.0]))

    assert metrics == {
        "l2": 5.0,
        "mean_abs": 3.5,
        "linf": 4.0,
        "finite": True,
        "zero_gradient": False,
    }


def test_pairwise_cosine_matrix_uses_raw_batch_gradients() -> None:
    gradients = {
        "O1": torch.tensor([1.0, 0.0]),
        "O2": torch.tensor([1.0, 0.0]),
        "P1": torch.tensor([-1.0, 0.0]),
        "P2": torch.tensor([0.0, 1.0]),
    }

    pairs, matrix = pairwise_cosines(gradients)

    assert pairs["O1-O2"] == pytest.approx(1.0)
    assert pairs["O1-P1"] == pytest.approx(-1.0)
    assert pairs["O1-P2"] == pytest.approx(0.0)
    assert pairs["O2-P1"] == pytest.approx(-1.0)
    assert pairs["O2-P2"] == pytest.approx(0.0)
    assert pairs["P1-P2"] == pytest.approx(0.0)
    assert matrix["P2"]["P2"] == pytest.approx(1.0)
    assert matrix["O1"]["P1"] == matrix["P1"]["O1"]


def test_zero_gradient_is_valid_and_its_cosines_are_undefined() -> None:
    zero = torch.zeros(2)
    nonzero = torch.tensor([1.0, 0.0])

    metrics = gradient_magnitude(zero)

    assert metrics["zero_gradient"] is True
    assert metrics["l2"] == 0.0
    assert gradient_cosine(zero, nonzero) is None
    assert gradient_cosine(zero, zero) is None
    analysis = analyze_batch_gradients(
        {"O1": zero, "O2": nonzero, "P1": zero, "P2": nonzero}
    )
    assert analysis["pairwise_cosine"]["O1-O2"] is None
    assert analysis["cosine_matrix"]["O1"]["O1"] is None


@pytest.mark.parametrize(
    ("gradient", "message"),
    [
        (None, "missing"),
        (torch.tensor([float("nan")]), "non-finite"),
        (torch.tensor([float("inf")]), "non-finite"),
    ],
)
def test_missing_and_nonfinite_gradients_fail(
    gradient: torch.Tensor | None, message: str
) -> None:
    values = _all_components(torch.tensor([1.0]))
    values["O1"] = gradient

    with pytest.raises(MultiLevelGradientDiagnosticError, match=message):
        ComponentGradientAccumulator().add(values)


def test_combined_reconstruction_and_cancellation_metrics() -> None:
    gradients = {
        "O1": torch.tensor([1.0, 0.0]),
        "O2": torch.tensor([1.0, 0.0]),
        "P1": torch.tensor([-1.0, 0.0]),
        "P2": torch.tensor([0.0, 1.0]),
    }

    analysis = analyze_batch_gradients(gradients)

    assert analysis["combined"]["openvla_gradient_l2"] == pytest.approx(2.0)
    assert analysis["combined"]["pi05_gradient_l2"] == pytest.approx(math.sqrt(2))
    assert analysis["combined"]["combined_model_cosine"] == pytest.approx(
        -1 / math.sqrt(2)
    )
    assert cancellation_ratio(
        torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
    ) == pytest.approx(1.0)
    assert cancellation_ratio(
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    ) == pytest.approx(0.0)
    assert cancellation_ratio(
        torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    ) == pytest.approx(1 / math.sqrt(2))


def test_component_losses_are_negative_mse_and_each_reaches_texture() -> None:
    texture = torch.nn.Parameter(torch.tensor(1.0))
    clean = FrozenMultiLevelNativeReference.from_tensors(
        torch.zeros(1, 256, 1152),
        torch.zeros(1, 256, 4096),
        torch.zeros(1, 256, 1152),
        torch.zeros(1, 256, 2048),
    )
    features = MultiLevelNativeFeatures(
        o1=texture.expand(1, 256, 1152),
        o2=(texture * 2).expand(1, 256, 4096),
        p1=(texture * 3).expand(1, 256, 1152),
        p2=(texture * 4).expand(1, 256, 2048),
    )

    losses = component_losses(features, clean)
    gradients = component_texture_gradients(losses, texture)

    assert {name: losses.mses[name].item() for name in diagnostic.COMPONENTS} == {
        "O1": 1.0,
        "O2": 4.0,
        "P1": 9.0,
        "P2": 16.0,
    }
    for name in diagnostic.COMPONENTS:
        assert losses.losses[name].item() == -losses.mses[name].item()
        assert bool(torch.isfinite(gradients[name]))
        assert bool(gradients[name] != 0)


def test_full_diagnostic_is_read_only_and_uses_four_autograd_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    before = parameter.detach().clone()
    calls = 0
    original_grad = torch.autograd.grad

    def counted_grad(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    frames = [
        GradientDiagnosticFrame("frame-0", 1.0),
        GradientDiagnosticFrame("frame-1", 2.0),
    ]

    def forward(frame: GradientDiagnosticFrame) -> ComponentLosses:
        scale = float(frame.payload)
        values = {
            "O1": -(parameter * torch.tensor([1.0, 2.0]) * scale).sum(),
            "O2": -(parameter * torch.tensor([3.0, 1.0]) * scale).sum(),
            "P1": -(parameter * torch.tensor([-1.0, 4.0]) * scale).sum(),
            "P2": -(parameter * torch.tensor([2.0, -3.0]) * scale).sum(),
        }
        return ComponentLosses(
            losses=values,
            mses={name: -value for name, value in values.items()},
        )

    report = run_gradient_decomposition(
        parameter=parameter,
        frames=frames,
        forward_frame=forward,
    )

    assert calls == len(frames) * 4
    assert torch.equal(parameter.detach(), before)
    assert parameter.grad is None
    assert report["texture_parameter_unchanged"] is True
    assert report["batch_size"] == 2
    assert report["frame_ids"] == ["frame-0", "frame-1"]


def test_texture_mutation_is_detected() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.25]))
    frames = [GradientDiagnosticFrame("frame-0", None)]

    def mutating_forward(_frame: GradientDiagnosticFrame) -> ComponentLosses:
        with torch.no_grad():
            parameter.add_(1)
        loss = parameter.square().sum()
        return ComponentLosses(
            losses={name: loss for name in diagnostic.COMPONENTS},
            mses={name: loss for name in diagnostic.COMPONENTS},
        )

    with pytest.raises(MultiLevelGradientDiagnosticError, match="mutated"):
        run_gradient_decomposition(
            parameter=parameter,
            frames=frames,
            forward_frame=mutating_forward,
        )


def test_texture_artifact_loading_validates_and_records_identity(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "vertex_noise.pt"
    source = torch.tensor([1.0, -2.0])
    torch.save(source, artifact)
    parameter = torch.nn.Parameter(torch.zeros(2))

    metadata = load_texture_parameter(parameter, artifact)

    assert torch.equal(parameter.detach(), source)
    assert metadata["absolute_path"] == str(artifact.resolve())
    assert metadata["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert metadata["shape"] == [2]
    assert metadata["dtype"] == "torch.float32"

    invalid = tmp_path / "invalid.pt"
    torch.save(torch.zeros(3), invalid)
    with pytest.raises(MultiLevelGradientDiagnosticError, match="shape mismatch"):
        load_texture_parameter(parameter, invalid)


def test_cli_contract_requires_no_shared_feature_artifact(tmp_path: Path) -> None:
    entrypoint = _load_entrypoint()
    texture = tmp_path / "vertex_noise.pt"
    texture.write_bytes(b"fixture")
    roots = []
    for name in ("openpi", "openvla", "pi05", "libero"):
        path = tmp_path / name
        path.mkdir()
        roots.append(path)

    args = entrypoint._args(
        [
            "--output-dir",
            str(tmp_path / "fresh-output"),
            "--texture-param",
            str(texture),
            "--label",
            "final",
            "--openpi-root",
            str(roots[0]),
            "--openvla-checkpoint",
            str(roots[1]),
            "--pi05-checkpoint",
            str(roots[2]),
            "--libero-root",
            str(roots[3]),
        ]
    )
    paths = entrypoint._validate_paths(args)

    assert args.label == "final"
    assert paths["texture"] == texture.resolve()
    assert paths["output"] == (tmp_path / "fresh-output").resolve()
    assert not hasattr(args, "mapping_dir")
    assert not hasattr(args, "shared_feature_root")


def test_existing_output_failure_does_not_modify_that_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _load_entrypoint()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("unchanged")
    argv = [
        "--output-dir",
        str(output),
        "--texture-param",
        str(tmp_path / "missing-texture.pt"),
        "--openpi-root",
        str(tmp_path / "missing-openpi"),
        "--openvla-checkpoint",
        str(tmp_path / "missing-openvla"),
        "--pi05-checkpoint",
        str(tmp_path / "missing-pi05"),
        "--libero-root",
        str(tmp_path / "missing-libero"),
    ]

    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), *argv])
    assert entrypoint.main() == 1
    assert marker.read_text() == "unchanged"
    assert not (output / "failure.json").exists()
    assert (tmp_path / "existing.failure.json").exists()


def test_diagnostic_source_has_no_optimizer_or_texture_update() -> None:
    core_source = Path(diagnostic.__file__).read_text()
    script_source = (SCRIPT_PATH).read_text()

    for forbidden in (
        "sign_pgd_update",
        "optimizer.step",
        "parameter.data -=",
        ".backward(",
    ):
        assert forbidden not in core_source
        assert forbidden not in script_source
    assert "torch.autograd.grad(" in core_source
