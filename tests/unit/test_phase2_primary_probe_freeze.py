from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import phase2_freeze_primary_action_probes as freeze


def _probe_contract() -> dict[str, object]:
    return {
        "seed": 7,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "steps": 2000,
        "action_std_epsilon": 1e-6,
        "probe_reg": 1e-4,
        "architecture": "Linear(D,7,bias=False)",
        "pooling": "mean over 256 visual tokens",
        "target": "model-specific deployed 7-D clean action",
    }


def _node(root: Path, model: str, node: str, width: int, fill: float) -> None:
    directory = root / model / node
    directory.mkdir(parents=True)
    torch.save({"state_dict": {}}, directory / "probe.pt")
    torch.save(torch.full((7, width), fill), directory / "W.pt")
    torch.save(torch.zeros(1), directory / "P_action.pt")
    np.savez(
        directory / "action_stats.npz",
        mean=np.zeros(7),
        std=np.ones(7),
        safe_std=np.ones(7),
        near_constant=np.zeros(7, dtype=bool),
    )
    (directory / "metrics.json").write_text("{}\n", encoding="utf-8")
    np.save(directory / "training_history.npy", np.zeros(2, dtype=np.float32))


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    value = {
        str(path.relative_to(root)): {
            "sha256": freeze.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    (root / "artifact_inventory.json").write_text(json.dumps(value), encoding="utf-8")
    return value


def test_phase2b_v3_reuses_openvla_and_freezes_corrected_pi05(tmp_path: Path) -> None:
    split = {
        "rule_id": "pilot-v0.2-c5-split-v1",
        "train_sample_ids": [f"train-{index}" for index in range(160)],
        "heldout_sample_ids": [f"held-{index}" for index in range(40)],
    }
    old = tmp_path / "phase2b-v2"
    old.mkdir()
    _node(old, "openvla", "o2", 4096, 1.0)
    _node(old, "pi05", "p2", 2048, 2.0)
    (old / "split.json").write_text(json.dumps(split), encoding="utf-8")
    old_inventory = _inventory(old)
    old_metadata = {
        "schema_version": "phase2b_primary_action_probes_v1",
        "status": "PHASE_2B_PRIMARY_FROZEN",
        "models": {
            "openvla": {
                "artifacts": {"W.pt": old_inventory["openvla/o2/W.pt"]["sha256"]}
            },
            "pi05": {"artifacts": {"W.pt": old_inventory["pi05/p2/W.pt"]["sha256"]}},
        },
    }
    (old / "metadata.json").write_text(json.dumps(old_metadata), encoding="utf-8")
    old_inventory = _inventory(old)

    corrected = tmp_path / "corrected-phase2a"
    corrected.mkdir()
    _node(corrected, "pi05", "p2", 2048, 3.0)
    (corrected / "split.json").write_text(json.dumps(split), encoding="utf-8")
    corrected_metadata = {
        "status": "PHASE_2A_COMPLETE",
        "stage": "phase2a",
        "selected_nodes": ["pi05/p2"],
        "probe": _probe_contract(),
        "dataset": {"observation_count": 200},
        "models": {
            "pi05": {
                "representation_nodes": {
                    "projected": {
                        "definition_id": "pi05_p2_embed_image_no_manual_scaling_v2"
                    }
                }
            }
        },
    }
    (corrected / "metadata.json").write_text(
        json.dumps(corrected_metadata), encoding="utf-8"
    )
    corrected_inventory = _inventory(corrected)

    stability = tmp_path / "stability"
    stability.mkdir()
    stability_summary = {
        "status": "CORRECTED_P2_STABLE",
        "selected_nodes": ["pi05/p2"],
        "phase2a_metadata_sha256": freeze.sha256_file(corrected / "metadata.json"),
        "source_artifacts_unchanged": True,
        "nodes": {"pi05/p2": {"phase2a_seed7_weight_max_abs_difference": 0.0}},
    }
    (stability / "summary.json").write_text(
        json.dumps(stability_summary), encoding="utf-8"
    )

    output = tmp_path / "phase2b-v3"
    result = freeze._run(
        SimpleNamespace(
            phase2a_dir=corrected,
            stability_dir=stability,
            openvla_source_phase2b=old,
            output_dir=output,
        )
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "phase2b_primary_action_probes_v2"
    assert result["openvla_W_sha256"] == old_inventory["openvla/o2/W.pt"]["sha256"]
    assert result["pi05_W_sha256"] == corrected_inventory["pi05/p2/W.pt"]["sha256"]
    assert metadata["provenance"]["openvla_W_unchanged"] is True
    assert (
        metadata["provenance"]["historical_pi05_W_sha256"]
        != metadata["provenance"]["corrected_pi05_W_sha256"]
    )
