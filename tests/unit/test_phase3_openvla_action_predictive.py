from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import phase3_openvla_action_optimization as entrypoint

ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

from phase3_action_predictive_objective import (
    ActionPredictiveObjectiveError,
    FrozenActionLinearProbe,
    load_frozen_openvla_probe,
)
from phase3_openvla_action_predictive import (
    OpenVLAReference,
    OpenVLATrainingFrame,
    calibrate_openvla_lambda,
    openvla_action_losses,
    train_openvla_action_predictive,
)


class _Renderer:
    def __init__(self) -> None:
        self.adv_noise = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
        self.epsilon = 128 / 255


def _probe() -> FrozenActionLinearProbe:
    weight = torch.zeros(7, 4096)
    for i in range(7):
        weight[i, i] = 1
    return FrozenActionLinearProbe(
        weight,
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
        safe_action_std=torch.ones(7),
        model="openvla",
        node="o2",
    )


def _fixture() -> tuple[_Renderer, FrozenActionLinearProbe, list[OpenVLATrainingFrame]]:
    renderer = _Renderer()
    probe = _probe()
    clean_o2 = torch.zeros(1, 256, 4096)
    clean_o2[:, :, :7] = torch.tensor([0.2, 0.4, 0.6, -0.2, 0.3, -0.4, 0.5])
    clean = OpenVLAReference.capture(clean_o2, probe)
    frames = [OpenVLATrainingFrame(f"frame-{i}", i, clean, i + 1) for i in range(3)]
    return renderer, probe, frames


def _forward(renderer: _Renderer, probe: FrozenActionLinearProbe):
    def forward(frame: OpenVLATrainingFrame, lam: float):
        o2 = frame.clean.o2.clone()
        o2 = o2 + torch.nn.functional.pad(
            (renderer.adv_noise * float(frame.payload)).repeat(1, 256, 1), (0, 4094)
        )
        return openvla_action_losses(o2, frame.clean, probe=probe, lambda_dir=lam)

    return forward


def test_losses_use_all_seven_coordinates_and_native_o2_metric() -> None:
    _, probe, frames = _fixture()
    clean = frames[0].clean
    adv = clean.o2.clone()
    adv[:, :, :7] += torch.arange(1, 8, dtype=torch.float32)
    losses = openvla_action_losses(adv, clean, probe=probe, lambda_dir=1.2)
    assert losses.action_mse.item() == pytest.approx(
        float(torch.arange(1, 8).square().float().mean())
    )
    assert losses.native_o2_mse.item() > 0
    assert losses.loss.item() == pytest.approx(
        losses.loss_mag.item() + 1.2 * losses.loss_dir.item()
    )
    assert losses.delta_z.shape == (7,)
    assert torch.equal(clean.z, probe(clean.o2))


def test_single_model_calibration_uses_median_and_does_not_update() -> None:
    renderer, probe, frames = _fixture()
    before = renderer.adv_noise.detach().clone()
    result = calibrate_openvla_lambda(
        renderer=renderer, frames=frames, forward_frame=_forward(renderer, probe)
    )
    assert result["status"] == "PASS"
    assert result["lambda_dir"] == pytest.approx(
        np.median([row["norm_ratio"] for row in result["frames"]])
    )
    assert "pi05" not in json.dumps(result)
    assert torch.equal(before, renderer.adv_noise)


def test_training_uses_direct_mean_openvla_gradient_and_records_metrics(
    tmp_path: Path,
) -> None:
    renderer, probe, frames = _fixture()
    forward = _forward(renderer, probe)
    calibration = calibrate_openvla_lambda(
        renderer=renderer, frames=frames, forward_frame=forward
    )
    expected = torch.stack(
        [
            torch.autograd.grad(
                forward(frame, calibration["lambda_dir"]).loss, renderer.adv_noise
            )[0]
            for frame in frames
        ]
    ).mean(0)
    before = renderer.adv_noise.detach().clone()
    history = train_openvla_action_predictive(
        renderer=renderer,
        frames=frames,
        forward_frame=forward,
        lambda_dir=calibration["lambda_dir"],
        iterations=1,
        requested_batch_size=20,
        pgd_step=0.05,
        seed=7,
        metrics_path=tmp_path / "metrics.jsonl",
    )
    row = history[0]
    assert row["openvla_texture_gradient_l2_norm"] == pytest.approx(
        expected.norm().item()
    )
    assert torch.allclose(renderer.adv_noise.detach() - before, -0.05 * expected.sign())
    assert row["texture_budget_respected"]
    assert row["o2_mse"] >= 0 and row["action_mse"] >= 0
    assert len([key for key in row if key.startswith("openvla_delta_")]) == 7
    assert len([key for key in row if key.startswith("openvla_abs_delta_")]) == 7
    assert "pi05" not in json.dumps(row)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "provisional"
    node = root / "openvla/o2"
    node.mkdir(parents=True)
    torch.save(torch.ones(7, 4096), node / "W.pt")
    torch.save({}, node / "probe.pt")
    (node / "metrics.json").write_text("{}")
    np.savez(
        node / "action_stats.npz",
        mean=np.zeros(7, dtype=np.float32),
        std=np.ones(7, dtype=np.float32),
        safe_std=np.ones(7, dtype=np.float32),
        near_constant=np.zeros(7, dtype=bool),
    )
    (root / "split.json").write_text("{}")
    artifact_hashes = {path.name: _hash(path) for path in node.iterdir()}
    metadata = {
        "schema_version": "phase2_expanded_seed7_provisional_action_probes_v1",
        "status": "PHASE2_EXPANDED_SEED7_PROVISIONAL_FROZEN",
        "artifact_id": "phase2-expanded-seed7-provisional-action-probes-v1",
        "split_rule": "pilot-v0.3-expanded-split-v1",
        "seed": 7,
        "provenance": {"promotion": "byte-identical copy; no probe refitting"},
        "phase2_status": {"authoritative_phase2b_v3": "NOT_FROZEN_BLOCKED"},
        "dataset": {"protocol": {"protocol_id": "pilot-v0.3-expanded-v1"}},
        "train_observations": 1914,
        "heldout_observations": 456,
        "train_trajectory_groups": 319,
        "heldout_trajectory_groups": 76,
        "probe": {
            "architecture": "Linear(D,7,bias=False)",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "steps": 2000,
            "action_std_epsilon": 1e-6,
            "probe_reg": 1e-4,
        },
        "models": {
            "openvla": {
                "checkpoint_path": "/fixture/openvla-checkpoint",
                "artifacts": artifact_hashes,
                "representation_nodes": {
                    "projected": {"name": "O2", "shape": [256, 4096]}
                },
            }
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    inventory = {
        str(path.relative_to(root)): {"sha256": _hash(path)}
        for path in root.rglob("*")
        if path.is_file()
    }
    (root / "artifact_inventory.json").write_text(json.dumps(inventory))
    return root


def test_loader_reads_only_openvla_and_rejects_hash_or_protocol_change(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path)
    artifact = load_frozen_openvla_probe(root, device="cpu")
    assert torch.equal(artifact.openvla.weight, torch.ones(7, 4096))
    assert set(artifact.hashes) == {
        "metadata.json",
        "split.json",
        "openvla/o2/W.pt",
        "openvla/o2/probe.pt",
        "openvla/o2/metrics.json",
        "openvla/o2/action_stats.npz",
    }
    (root / "openvla/o2/metrics.json").write_text("tampered")
    with pytest.raises(ActionPredictiveObjectiveError, match="hash mismatch"):
        load_frozen_openvla_probe(root, device="cpu")
    root = _artifact(tmp_path / "other")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["seed"] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ActionPredictiveObjectiveError, match="invalid expanded"):
        load_frozen_openvla_probe(root, device="cpu")

    root = _artifact(tmp_path / "counts")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["train_observations"] = 160
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ActionPredictiveObjectiveError, match="invalid expanded"):
        load_frozen_openvla_probe(root, device="cpu")


def test_entrypoint_requires_fresh_output_and_no_pi05_arguments(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    libero = tmp_path / "libero"
    checkpoint.mkdir()
    libero.mkdir()
    args = entrypoint._args(
        [
            "--output-dir",
            str(tmp_path / "new"),
            "--probe-artifact-dir",
            str(root),
            "--openvla-checkpoint",
            str(checkpoint),
            "--libero-root",
            str(libero),
            "--iterations",
            "10",
            "--smoke",
        ]
    )
    assert entrypoint._validate_paths(args)["output"] == tmp_path / "new"
    assert not hasattr(args, "pi05_checkpoint") and not hasattr(args, "openpi_root")
    (tmp_path / "new").mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        entrypoint._validate_paths(args)


def test_checkpoint_must_match_frozen_probe_provenance(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text())
    entrypoint._validate_checkpoint_provenance(
        Path("/fixture/openvla-checkpoint"), metadata
    )
    with pytest.raises(RuntimeError, match="checkpoint differs"):
        entrypoint._validate_checkpoint_provenance(Path("/fixture/other"), metadata)
