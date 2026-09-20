from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import phase2_action_probe_materialize as materializer


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

from phase2_action_probe import (  # noqa: E402
    ACTION_NAMES,
    ProbeConfig,
    SampleIdentity,
    action_distribution_audit,
    build_group_split,
    fit_linear_probe,
    materialize_action_projection,
    mean_pool_features,
    prediction_metrics,
    training_action_statistics,
)


def _pilot_samples() -> list[SampleIdentity]:
    return [
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


def test_group_split_is_disjoint_balanced_and_deterministic() -> None:
    samples = _pilot_samples()
    first = build_group_split(samples)
    second = build_group_split(samples)
    assert first == second
    assert len(first["train_groups"]) == 40
    assert len(first["heldout_groups"]) == 10
    assert len(first["train_sample_ids"]) == 160
    assert len(first["heldout_sample_ids"]) == 40
    assert not set(first["train_sample_ids"]) & set(first["heldout_sample_ids"])
    assert {row["task_id"] for row in first["heldout_groups"]} == set(range(10))


def test_action_audit_has_required_overall_task_progress_and_gripper_stats() -> None:
    samples = _pilot_samples()
    actions = np.arange(200 * 7, dtype=np.float32).reshape(200, 7) / 100
    actions[:, 6] = np.tile([-1.0, 1.0], 100)
    audit = action_distribution_audit(actions, samples)
    assert audit["overall"]["count"] == 200
    assert set(audit["overall"]["dimensions"]) == set(ACTION_NAMES)
    assert len(audit["by_task_id"]) == 10
    assert set(audit["by_target_progress"]) == {"0.10", "0.40", "0.70", "0.90"}
    assert audit["overall"]["gripper"]["effective_states"] == 2


def test_train_only_normalization_flags_constant_dimension() -> None:
    actions = np.arange(35, dtype=np.float32).reshape(5, 7)
    actions[:, 3] = 0.25
    stats = training_action_statistics(actions, epsilon=1e-6)
    assert stats["near_constant"][3]
    assert stats["safe_std"][3] == pytest.approx(1e-6)
    assert np.all(stats["safe_std"] >= 1e-6)


def test_pool_probe_metrics_isolation_and_serialization(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(32, 256, 8)).astype(np.float32)
    pooled = mean_pool_features(features, width=8)
    true_weight = rng.normal(size=(7, 8)).astype(np.float32)
    actions = pooled @ true_weight.T
    upstream = torch.nn.Parameter(torch.ones(1))
    probe, stats, history = fit_linear_probe(
        pooled,
        actions,
        config=ProbeConfig(steps=200, learning_rate=0.03, weight_decay=0.0),
    )
    assert history[-1] < history[0]
    assert upstream.grad is None
    assert probe.weight.grad is not None
    prediction = probe(torch.from_numpy(pooled)).detach().numpy()
    metrics = prediction_metrics(prediction, actions, stats)
    assert metrics["normalized"]["mse"] < 0.2

    path = tmp_path / "probe.pt"
    torch.save({"state_dict": probe.state_dict()}, path)
    restored = torch.nn.Linear(8, 7, bias=False)
    restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    assert torch.allclose(
        probe(torch.from_numpy(pooled)), restored(torch.from_numpy(pooled))
    )


def test_ridge_projection_shape_symmetry_and_reported_residual() -> None:
    weight = torch.randn(7, 11)
    projection, metrics = materialize_action_projection(weight, probe_reg=1e-4)
    assert projection.shape == (11, 11)
    assert torch.isfinite(projection).all()
    assert torch.allclose(projection, projection.T, atol=2e-4, rtol=2e-4)
    assert metrics["rank_W"] == 7
    assert metrics["symmetry_residual_fro"] < 1e-3
    assert metrics["idempotence_residual_fro"] >= 0


def test_mean_predictor_is_zero_in_train_normalized_space() -> None:
    rng = np.random.default_rng(3)
    actions = rng.normal(size=(20, 7)).astype(np.float32)
    stats = training_action_statistics(actions, epsilon=1e-6)
    metrics = prediction_metrics(np.zeros_like(actions), actions, stats)
    assert metrics["normalized"]["mse"] == pytest.approx(1.0, rel=1e-5)


def test_corrected_pi05_manifest_requires_runtime_identity_contract(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "representation_manifest.json"
    manifest = {
        "schema_version": "phase2_action_representation_manifest_v2",
        "status": "COMPLETE",
        "model": "pi05",
        "count": 200,
        "records": [
            {"sample_id": f"sample-{index}", "archive": "unused", "sha256": "x"}
            for index in range(200)
        ],
        "representation_nodes": {
            "projected": {
                "definition_id": "pi05_p2_embed_image_no_manual_scaling_v2",
                "capture_semantics": "current PI0Pytorch base-camera embed_image() output; no additional manual scaling",
            }
        },
        "pi05_p2_identity": {"identity_pass": True},
        "action_identity": {"pass": True},
    }
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    loaded = materializer._manifest(manifest_path, "pi05")
    assert loaded["representation_nodes"]["projected"]["definition_id"].endswith(
        "no_manual_scaling_v2"
    )

    manifest["pi05_p2_identity"]["identity_pass"] = False
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception, match="representation identity"):
        materializer._manifest(manifest_path, "pi05")


def test_materializer_writes_complete_reloadable_artifact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _pilot_samples()
    rng = np.random.default_rng(11)
    features = {
        "projected": rng.normal(size=(200, 8)).astype(np.float32),
        "deep": rng.normal(size=(200, 8)).astype(np.float32),
    }
    action_weight = rng.normal(size=(7, 8)).astype(np.float32)
    actions = features["projected"] @ action_weight.T
    source_manifest = tmp_path / "representation_manifest.json"
    source_manifest.write_text("{}", encoding="utf-8")

    def fake_manifest(_path: Path, model: str):
        return {
            "_path": source_manifest,
            "backend": "fixture",
            "model_identity": model,
            "checkpoint_path": f"/{model}",
            "projected_node": "O2" if model == "openvla" else "P2",
            "deep_node": {
                "module_path": f"{model}.layers[0]",
                "visual_token_slice": "fixture",
            },
            "representation_nodes": {"projected": {}, "deep": {}},
            "action_target_semantics": "fixture deployed action",
            "saved_dtype": "float32",
            "token_count": 256,
            "collection_manifest_sha256": "sha256:" + "0" * 64,
        }

    monkeypatch.setattr(materializer, "_manifest", fake_manifest)
    monkeypatch.setattr(
        materializer,
        "_load_model_data",
        lambda _manifest, _model: (samples, features, actions),
    )
    monkeypatch.setattr(materializer, "_head", lambda: "fixture-commit")
    output = tmp_path / "phase2_action_probe"
    result = materializer._run(
        SimpleNamespace(
            openvla_manifest=source_manifest,
            pi05_manifest=source_manifest,
            output_dir=output,
            stage="phase2b",
            report_path=None,
            probe_device="cpu",
            seed=7,
            probe_lr=0.01,
            probe_weight_decay=0.0,
            probe_steps=3,
            action_std_epsilon=1e-6,
            probe_reg=1e-4,
        )
    )

    assert result["status"] == "PHASE_2_COMPLETE"
    assert (output / "metadata.json").is_file()
    assert (output / "split.json").is_file()
    assert (output / "action_audit/openvla.json").is_file()
    assert (output / "phase2-action-predictive-representation-report.md").is_file()
    assert (output / "artifact_inventory.json").is_file()
    for model, node in (
        ("openvla", "o2"),
        ("openvla", "deep"),
        ("pi05", "p2"),
        ("pi05", "deep"),
    ):
        node_dir = output / model / node
        assert torch.load(node_dir / "W.pt", weights_only=True).shape == (7, 8)
        assert torch.load(node_dir / "P_action.pt", weights_only=True).shape == (8, 8)
        assert (node_dir / "action_stats.npz").is_file()
        assert (node_dir / "metrics.json").is_file()
        assert (node_dir / "training_history.npy").is_file()
