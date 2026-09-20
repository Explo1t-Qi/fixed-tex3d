from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import phase2_action_probe_stability as runner


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

from phase2_action_probe import (  # noqa: E402
    PROBE_SCHEMA_VERSION,
    ActionProbeError,
    ProbeConfig,
    SampleIdentity,
    build_group_split,
    fit_linear_probe,
    sha256_file,
    training_action_statistics,
)
from phase2_action_probe_stability_core import (  # noqa: E402
    assert_tree_unchanged,
    principal_angle_cosines,
    projection_relative_frobenius_distance,
    signed_action_row_cosines,
    tree_hashes,
)


def test_probe_fit_is_same_seed_deterministic_and_different_seed_independent() -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(24, 12)).astype(np.float32)
    actions = rng.normal(size=(24, 7)).astype(np.float32)

    def fit(seed: int) -> torch.Tensor:
        probe, _, _ = fit_linear_probe(
            features,
            actions,
            config=ProbeConfig(
                seed=seed,
                learning_rate=0.01,
                weight_decay=1e-4,
                steps=10,
            ),
        )
        return probe.weight.detach().clone()

    first = fit(1)
    repeated = fit(1)
    other = fit(2)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)


def test_signed_action_row_cosines_preserve_sign() -> None:
    first = torch.eye(7)
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    second = torch.diag(signs)
    assert torch.allclose(signed_action_row_cosines(first, second), signs)


def test_principal_cosines_are_one_under_row_basis_rotation() -> None:
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(7, 14, generator=generator)
    rotation = torch.linalg.qr(torch.randn(7, 7, generator=generator)).Q
    rotated = rotation @ weight
    values = principal_angle_cosines(weight, rotated)
    assert torch.allclose(values, torch.ones(7, dtype=values.dtype), atol=1e-12)


def test_principal_cosines_detect_orthogonal_subspaces() -> None:
    first = torch.cat((torch.eye(7), torch.zeros(7, 7)), dim=1)
    second = torch.cat((torch.zeros(7, 7), torch.eye(7)), dim=1)
    values = principal_angle_cosines(first, second)
    assert torch.allclose(values, torch.zeros(7, dtype=values.dtype), atol=1e-12)


def test_low_rank_projection_distance_matches_dense_reference() -> None:
    generator = torch.Generator().manual_seed(8)
    first = torch.randn(7, 20, generator=generator, dtype=torch.float64)
    second = torch.randn(7, 20, generator=generator, dtype=torch.float64)
    probe_reg = 1e-4

    def projection(weight: torch.Tensor) -> torch.Tensor:
        gram = weight @ weight.T
        solve = torch.linalg.solve(
            gram + probe_reg * torch.eye(7, dtype=torch.float64), weight
        )
        return weight.T @ solve

    p_first, p_second = projection(first), projection(second)
    expected = float(
        torch.linalg.vector_norm(p_first - p_second)
        / max(
            torch.linalg.vector_norm(p_first),
            torch.linalg.vector_norm(p_second),
        )
    )
    actual = projection_relative_frobenius_distance(first, second, probe_reg=probe_reg)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert projection_relative_frobenius_distance(
        first, first, probe_reg=probe_reg
    ) == pytest.approx(0.0, abs=1e-7)


def test_source_artifact_snapshot_enforces_read_only_invariant(tmp_path: Path) -> None:
    source = tmp_path / "phase2a"
    source.mkdir()
    artifact = source / "W.pt"
    artifact.write_bytes(b"frozen")
    before = tree_hashes(source)

    _ = principal_angle_cosines(torch.eye(7), torch.eye(7))
    assert_tree_unchanged(source, before)

    artifact.write_bytes(b"changed")
    with pytest.raises(ActionProbeError, match="changed during diagnostic"):
        assert_tree_unchanged(source, before)


def test_runner_writes_fresh_outputs_without_modifying_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase2a = tmp_path / "phase2a"
    openvla_root = tmp_path / "representations" / "openvla"
    pi05_root = tmp_path / "representations" / "pi05"
    for directory in (phase2a, openvla_root, pi05_root):
        directory.mkdir(parents=True)
    openvla_manifest = openvla_root / "representation_manifest.json"
    pi05_manifest = pi05_root / "representation_manifest.json"
    openvla_manifest.write_text("{}", encoding="utf-8")
    pi05_manifest.write_text("{}", encoding="utf-8")

    samples = [
        SampleIdentity(
            sample_id=f"t{task}-s{state}-p{progress}",
            task_id=task,
            initial_state_id=state,
            target_progress=(0.1, 0.4, 0.7, 0.9)[progress],
        )
        for task in range(10)
        for state in range(5)
        for progress in range(4)
    ]
    split = build_group_split(samples)
    rng = np.random.default_rng(31)
    features = {
        "projected": rng.normal(size=(200, 8)).astype(np.float32),
        "deep": rng.normal(size=(200, 8)).astype(np.float32),
    }
    actions = rng.normal(size=(200, 7)).astype(np.float32)
    train_indices = np.asarray(
        [
            index
            for index, sample in enumerate(samples)
            if sample.sample_id in split["train_sample_ids"]
        ]
    )
    stats = training_action_statistics(actions[train_indices], epsilon=1e-6)
    fixture_config = ProbeConfig(
        seed=7,
        learning_rate=1e-3,
        weight_decay=1e-4,
        steps=2,
        action_std_epsilon=1e-6,
        probe_reg=1e-4,
    )
    monkeypatch.setattr(runner, "FROZEN_CONFIG", fixture_config)

    metadata = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "stage": "phase2a",
        "status": "PHASE_2A_COMPLETE",
        "dataset": {"observation_count": 200},
        "probe": {
            "learning_rate": fixture_config.learning_rate,
            "weight_decay": fixture_config.weight_decay,
            "steps": fixture_config.steps,
            "action_std_epsilon": fixture_config.action_std_epsilon,
            "probe_reg": fixture_config.probe_reg,
        },
        "models": {
            "openvla": {
                "representation_manifest_sha256": sha256_file(openvla_manifest)
            },
            "pi05": {"representation_manifest_sha256": sha256_file(pi05_manifest)},
        },
    }
    (phase2a / "metadata.json").write_text(
        __import__("json").dumps(metadata), encoding="utf-8"
    )
    (phase2a / "split.json").write_text(
        __import__("json").dumps(split), encoding="utf-8"
    )
    for model, node in (
        ("openvla", "o2"),
        ("openvla", "deep"),
        ("pi05", "p2"),
        ("pi05", "deep"),
    ):
        directory = phase2a / model / node
        directory.mkdir(parents=True)
        np.savez(directory / "action_stats.npz", **stats)
        torch.save(torch.zeros(7, 8), directory / "W.pt")

    def fake_manifest(path: Path, model: str) -> dict[str, object]:
        return {"_path": path, "model": model}

    monkeypatch.setattr(runner.materializer, "_manifest", fake_manifest)
    monkeypatch.setattr(
        runner.materializer,
        "_load_model_data",
        lambda _manifest, _model: (samples, features, actions),
    )
    monkeypatch.setattr(runner, "_head", lambda: "fixture-commit")

    sources_before = {
        "phase2a": tree_hashes(phase2a),
        "openvla": tree_hashes(openvla_root),
        "pi05": tree_hashes(pi05_root),
    }
    output = tmp_path / "stability"
    result = runner._run(
        SimpleNamespace(
            phase2a_dir=phase2a,
            openvla_manifest=openvla_manifest,
            pi05_manifest=pi05_manifest,
            output_dir=output,
            report_path=None,
            seeds=[1, 2, 3, 4, 5, 7],
        )
    )

    assert result["status"] in {"STABLE", "NEEDS_REVIEW"}
    assert (output / "summary.json").is_file()
    assert (output / "per_seed_metrics.json").is_file()
    assert (output / "subspace_similarity.json").is_file()
    assert (output / "phase2-action-probe-stability-report.md").is_file()
    for model, node in (
        ("openvla", "o2"),
        ("openvla", "deep"),
        ("pi05", "p2"),
        ("pi05", "deep"),
    ):
        for seed in (1, 2, 3, 4, 5, 7):
            directory = output / model / node / f"seed_{seed:06d}"
            assert (directory / "W.pt").is_file()
            assert (directory / "P_action.pt").is_file()
            assert (directory / "metrics.json").is_file()
    assert tree_hashes(phase2a) == sources_before["phase2a"]
    assert tree_hashes(openvla_root) == sources_before["openvla"]
    assert tree_hashes(pi05_root) == sources_before["pi05"]

    metadata["selected_nodes"] = ["pi05/p2"]
    (phase2a / "metadata.json").write_text(
        __import__("json").dumps(metadata), encoding="utf-8"
    )
    corrected_output = tmp_path / "corrected-p2-stability"
    corrected = runner._run(
        SimpleNamespace(
            phase2a_dir=phase2a,
            openvla_manifest=openvla_manifest,
            pi05_manifest=pi05_manifest,
            output_dir=corrected_output,
            report_path=None,
            seeds=[1, 2, 3, 4, 5, 7],
            nodes=["pi05/p2"],
        )
    )
    assert corrected["status"] in {
        "CORRECTED_P2_STABLE",
        "CORRECTED_P2_NEEDS_REVIEW",
    }
    corrected_summary = __import__("json").loads(
        (corrected_output / "summary.json").read_text(encoding="utf-8")
    )
    assert corrected_summary["selected_nodes"] == ["pi05/p2"]
    assert set(corrected_summary["nodes"]) == {"pi05/p2"}
