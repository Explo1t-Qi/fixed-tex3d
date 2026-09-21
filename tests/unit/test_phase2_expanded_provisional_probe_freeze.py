from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import phase2_freeze_expanded_seed7_provisional_action_probes as freeze


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))
from phase3_action_predictive_objective import load_frozen_primary_probes  # noqa: E402


def _write_node(root: Path, model: str, node: str, width: int, fill: float) -> None:
    directory = root / model / node
    directory.mkdir(parents=True)
    torch.save({"state_dict": {}}, directory / "probe.pt")
    torch.save(torch.full((7, width), fill), directory / "W.pt")
    torch.save(torch.zeros(1), directory / "P_action.pt")
    np.savez(
        directory / "action_stats.npz",
        mean=np.zeros(7, dtype=np.float32),
        std=np.ones(7, dtype=np.float32),
        safe_std=np.ones(7, dtype=np.float32),
        near_constant=np.zeros(7, dtype=bool),
    )
    (directory / "metrics.json").write_text("{}\n", encoding="utf-8")
    np.save(directory / "training_history.npy", np.zeros(2, dtype=np.float32))


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    inventory = {
        str(path.relative_to(root)): {
            "sha256": freeze.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    (root / "artifact_inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    return inventory


def _representation_manifest(
    path: Path, collection_sha256: str, *, model: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "phase2_action_representation_manifest_v3",
        "status": "COMPLETE",
        "collection_manifest_sha256": collection_sha256,
    }
    if model == "pi05":
        payload["projected"] = {
            "definition_id": "pi05_p2_embed_image_no_manual_scaling_v2"
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_artifacts(tmp_path: Path) -> dict[str, Path]:
    collection = tmp_path / "collection_manifest.json"
    collection.write_text(
        json.dumps(
            {
                "schema_version": "pilot_v0_3_expanded_collection_v1",
                "protocol_id": "pilot-v0.3-expanded-v1",
                "coverage": {
                    "actual_total_observations": 2370,
                    "actual_total_groups": 395,
                },
            }
        ),
        encoding="utf-8",
    )
    collection_hash = freeze.sha256_file(collection)
    openvla_representation = (
        tmp_path / "representations/openvla/representation_manifest.json"
    )
    pi05_representation = tmp_path / "representations/pi05/representation_manifest.json"
    _representation_manifest(openvla_representation, collection_hash, model="openvla")
    _representation_manifest(pi05_representation, collection_hash, model="pi05")

    source = tmp_path / "phase2a"
    source.mkdir()
    _write_node(source, "openvla", "o2", 4096, 1.0)
    _write_node(source, "pi05", "p2", 2048, 2.0)
    split = {
        "rule_id": "pilot-v0.3-expanded-split-v1",
        "train_sample_ids": [f"train-{index}" for index in range(1914)],
        "heldout_sample_ids": [f"heldout-{index}" for index in range(456)],
        "train_groups": [{"id": index} for index in range(319)],
        "heldout_groups": [{"id": index} for index in range(76)],
    }
    (source / "split.json").write_text(json.dumps(split), encoding="utf-8")
    metadata = {
        "status": "PHASE_2A_COMPLETE",
        "stage": "phase2a",
        "selected_nodes": ["openvla/o2", "pi05/p2"],
        "probe": freeze.EXPECTED_PROBE,
        "dataset": {
            "collection_manifest_sha256": collection_hash,
            "observation_count": 2370,
            "trajectory_group_count": 395,
            "protocol": {"protocol_id": "pilot-v0.3-expanded-v1"},
        },
        "models": {
            "openvla": {
                "representation_manifest_sha256": freeze.sha256_file(
                    openvla_representation
                ),
                "representation_nodes": {"projected": {"name": "O2"}},
            },
            "pi05": {
                "representation_manifest_sha256": freeze.sha256_file(
                    pi05_representation
                ),
                "representation_nodes": {
                    "projected": {
                        "name": "P2",
                        "definition_id": "pi05_p2_embed_image_no_manual_scaling_v2",
                        "capture_semantics": (
                            "current PI0Pytorch base-camera embed_image() output; "
                            "no additional manual scaling"
                        ),
                    }
                },
            },
        },
        "results": [
            {"model": "openvla", "node": "o2"},
            {"model": "pi05", "node": "p2"},
        ],
    }
    (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _inventory(source)

    stability = tmp_path / "stability"
    stability.mkdir()
    (stability / "summary.json").write_text(
        json.dumps(
            {
                "status": "CORRECTED_P2_NEEDS_REVIEW",
                "selected_nodes": ["pi05/p2"],
                "phase2a_metadata_sha256": freeze.sha256_file(source / "metadata.json"),
                "source_artifacts_unchanged": True,
                "nodes": {
                    "pi05/p2": {"phase2a_seed7_weight_max_abs_difference": 0.0}
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "source": source,
        "stability": stability,
        "collection": collection,
        "openvla_representation": openvla_representation,
        "pi05_representation": pi05_representation,
    }


def _freeze(tmp_path: Path) -> tuple[dict[str, Path], Path, dict[str, object]]:
    paths = _source_artifacts(tmp_path)
    output = tmp_path / "provisional"
    result = freeze._run(
        SimpleNamespace(
            phase2a_dir=paths["source"],
            stability_dir=paths["stability"],
            collection_manifest=paths["collection"],
            openvla_representation_manifest=paths["openvla_representation"],
            pi05_representation_manifest=paths["pi05_representation"],
            output_dir=output,
        )
    )
    return paths, output, result


def test_provisional_freeze_is_byte_identical_and_explicitly_non_phase2b(
    tmp_path: Path,
) -> None:
    paths, output, result = _freeze(tmp_path)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert result["status"] == freeze.PHASE2_EXPANDED_SEED7_PROVISIONAL_STATUS
    assert (
        metadata["artifact_id"]
        == "phase2-expanded-seed7-provisional-action-probes-v1"
    )
    assert (
        metadata["phase2_status"]["authoritative_phase2b_v3"]
        == "NOT_FROZEN_BLOCKED"
    )
    assert (
        metadata["provenance"]["promotion"]
        == "byte-identical copy; no probe refitting"
    )
    for model, node in (("openvla", "o2"), ("pi05", "p2")):
        source = paths["source"] / model / node / "W.pt"
        copied = output / model / node / "W.pt"
        assert freeze.sha256_file(source) == freeze.sha256_file(copied)
    loaded = load_frozen_primary_probes(
        output, openvla_device="cpu", pi05_device="cpu"
    )
    assert loaded.metadata["status"] == freeze.PHASE2_EXPANDED_SEED7_PROVISIONAL_STATUS


def test_provisional_freeze_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    paths = _source_artifacts(tmp_path)
    torch.save(torch.zeros(7, 2048), paths["source"] / "pi05/p2/W.pt")

    with pytest.raises(freeze.ProvisionalProbeFreezeError, match="source hash"):
        freeze._run(
            SimpleNamespace(
                phase2a_dir=paths["source"],
                stability_dir=paths["stability"],
                collection_manifest=paths["collection"],
                openvla_representation_manifest=paths["openvla_representation"],
                pi05_representation_manifest=paths["pi05_representation"],
                output_dir=tmp_path / "provisional",
            )
        )
