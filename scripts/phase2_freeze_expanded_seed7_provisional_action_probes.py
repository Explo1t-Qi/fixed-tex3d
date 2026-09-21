#!/usr/bin/env python3
"""Freeze exact expanded seed-7 probes for exploratory Phase 3 feasibility only."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
if str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))

from phase2_action_probe import atomic_json, sha256_file  # noqa: E402
from phase3_action_predictive_objective import (  # noqa: E402
    PHASE2_EXPANDED_SEED7_PROVISIONAL_SCHEMA,
    PHASE2_EXPANDED_SEED7_PROVISIONAL_STATUS,
)


class ProvisionalProbeFreezeError(RuntimeError):
    """Raised when a source cannot support the explicit provisional contract."""


EXPECTED_PROBE = {
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
REQUIRED_FILES = (
    "probe.pt",
    "W.pt",
    "P_action.pt",
    "action_stats.npz",
    "metrics.json",
    "training_history.npy",
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2a-dir", type=Path, required=True)
    parser.add_argument("--stability-dir", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--openvla-representation-manifest", type=Path, required=True)
    parser.add_argument("--pi05-representation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ProvisionalProbeFreezeError(
            f"{name} mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_source(
    *,
    source: Path,
    source_metadata: dict[str, Any],
    source_inventory: dict[str, Any],
    collection_manifest: Path,
    openvla_representation_manifest: Path,
    pi05_representation_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_equal(source_metadata.get("status"), "PHASE_2A_COMPLETE", name="status")
    _require_equal(source_metadata.get("stage"), "phase2a", name="stage")
    _require_equal(
        source_metadata.get("selected_nodes"),
        ["openvla/o2", "pi05/p2"],
        name="selected_nodes",
    )
    for key, expected in EXPECTED_PROBE.items():
        _require_equal(
            source_metadata.get("probe", {}).get(key),
            expected,
            name=f"probe.{key}",
        )

    collection = _read_json(collection_manifest)
    _require_equal(
        collection.get("schema_version"),
        "pilot_v0_3_expanded_collection_v1",
        name="collection.schema_version",
    )
    _require_equal(
        collection.get("protocol_id"),
        "pilot-v0.3-expanded-v1",
        name="collection.protocol_id",
    )
    _require_equal(
        collection.get("coverage", {}).get("actual_total_observations"),
        2370,
        name="collection observation count",
    )
    _require_equal(
        collection.get("coverage", {}).get("actual_total_groups"),
        395,
        name="collection group count",
    )
    collection_hash = sha256_file(collection_manifest)
    dataset = source_metadata.get("dataset", {})
    _require_equal(
        dataset.get("collection_manifest_sha256"),
        collection_hash,
        name="collection manifest SHA-256",
    )
    _require_equal(
        dataset.get("observation_count"), 2370, name="dataset observation count"
    )
    _require_equal(
        dataset.get("trajectory_group_count"), 395, name="dataset group count"
    )
    _require_equal(
        dataset.get("protocol", {}).get("protocol_id"),
        "pilot-v0.3-expanded-v1",
        name="dataset protocol",
    )

    split = _read_json(source / "split.json")
    _require_equal(
        split.get("rule_id"), "pilot-v0.3-expanded-split-v1", name="split rule"
    )
    _require_equal(
        len(split.get("train_sample_ids", [])), 1914, name="TRAIN observation count"
    )
    _require_equal(
        len(split.get("heldout_sample_ids", [])),
        456,
        name="HELD-OUT observation count",
    )
    _require_equal(len(split.get("train_groups", [])), 319, name="TRAIN group count")
    _require_equal(
        len(split.get("heldout_groups", [])), 76, name="HELD-OUT group count"
    )

    manifests = {
        "openvla": _read_json(openvla_representation_manifest),
        "pi05": _read_json(pi05_representation_manifest),
    }
    for model, manifest_path in (
        ("openvla", openvla_representation_manifest),
        ("pi05", pi05_representation_manifest),
    ):
        manifest = manifests[model]
        _require_equal(
            manifest.get("status"), "COMPLETE", name=f"{model} representation status"
        )
        _require_equal(
            manifest.get("collection_manifest_sha256"),
            collection_hash,
            name=f"{model} representation collection SHA-256",
        )
        _require_equal(
            source_metadata["models"][model].get("representation_manifest_sha256"),
            sha256_file(manifest_path),
            name=f"{model} representation manifest SHA-256",
        )
    projected = source_metadata["models"]["pi05"]["representation_nodes"]["projected"]
    _require_equal(
        projected.get("definition_id"),
        "pi05_p2_embed_image_no_manual_scaling_v2",
        name="corrected PI0.5 P2 definition",
    )
    _require_equal(
        projected.get("capture_semantics"),
        (
            "current PI0Pytorch base-camera embed_image() output; "
            "no additional manual scaling"
        ),
        name="corrected PI0.5 P2 capture semantics",
    )

    for model, node, width in (("openvla", "o2", 4096), ("pi05", "p2", 2048)):
        for filename in REQUIRED_FILES:
            relative = f"{model}/{node}/{filename}"
            path = source / relative
            if not path.is_file():
                raise ProvisionalProbeFreezeError(
                    f"missing source artifact: {relative}"
                )
            _require_equal(
                source_inventory.get(relative, {}).get("sha256"),
                sha256_file(path),
                name=f"source hash {relative}",
            )
        weight = torch.load(
            source / model / node / "W.pt", map_location="cpu", weights_only=True
        )
        if tuple(weight.shape) != (7, width) or not bool(torch.isfinite(weight).all()):
            raise ProvisionalProbeFreezeError(f"invalid source W: {model}/{node}")
    return collection, split, manifests


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.phase2a_dir.expanduser().resolve(strict=True)
    stability = args.stability_dir.expanduser().resolve(strict=True)
    collection_manifest = args.collection_manifest.expanduser().resolve(strict=True)
    openvla_representation_manifest = (
        args.openvla_representation_manifest.expanduser().resolve(strict=True)
    )
    pi05_representation_manifest = (
        args.pi05_representation_manifest.expanduser().resolve(strict=True)
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")

    source_metadata_path = source / "metadata.json"
    source_inventory_path = source / "artifact_inventory.json"
    stability_summary_path = stability / "summary.json"
    source_metadata = _read_json(source_metadata_path)
    source_inventory = _read_json(source_inventory_path)
    stability_summary = _read_json(stability_summary_path)
    collection, split, manifests = _validate_source(
        source=source,
        source_metadata=source_metadata,
        source_inventory=source_inventory,
        collection_manifest=collection_manifest,
        openvla_representation_manifest=openvla_representation_manifest,
        pi05_representation_manifest=pi05_representation_manifest,
    )
    _require_equal(
        stability_summary.get("phase2a_metadata_sha256"),
        sha256_file(source_metadata_path),
        name="stability Phase 2A provenance",
    )
    _require_equal(
        stability_summary.get("selected_nodes"), ["pi05/p2"], name="stability nodes"
    )
    _require_equal(
        stability_summary.get("status"),
        "CORRECTED_P2_NEEDS_REVIEW",
        name="stability status",
    )
    _require_equal(
        stability_summary.get("source_artifacts_unchanged"),
        True,
        name="stability source-artifact invariant",
    )
    _require_equal(
        stability_summary["nodes"]["pi05/p2"].get(
            "phase2a_seed7_weight_max_abs_difference"
        ),
        0.0,
        name="PI0.5 seed-7 stability/source W equality",
    )

    output.mkdir(parents=True)
    shutil.copy2(source / "split.json", output / "split.json")
    copied: dict[str, dict[str, str]] = {}
    for model, node in (("openvla", "o2"), ("pi05", "p2")):
        destination = output / model / node
        destination.mkdir(parents=True)
        copied[f"{model}/{node}"] = {}
        for filename in REQUIRED_FILES:
            relative = f"{model}/{node}/{filename}"
            source_path = source / relative
            source_hash = sha256_file(source_path)
            destination_path = destination / filename
            shutil.copy2(source_path, destination_path)
            _require_equal(
                sha256_file(destination_path),
                source_hash,
                name=f"byte-identical copy {relative}",
            )
            copied[f"{model}/{node}"][filename] = source_hash

    metadata = {
        "schema_version": PHASE2_EXPANDED_SEED7_PROVISIONAL_SCHEMA,
        "status": PHASE2_EXPANDED_SEED7_PROVISIONAL_STATUS,
        "artifact_id": "phase2-expanded-seed7-provisional-action-probes-v1",
        "scope": (
            "Fixed seed-7 action-predictive probe artifact for exploratory Phase 3 "
            "pipeline-feasibility experiments only. It does not establish cross-seed "
            "subspace reproducibility and is not authoritative Phase 2B v3."
        ),
        "code_commit": _head(),
        "seed": 7,
        "split_rule": split["rule_id"],
        "train_observations": len(split["train_sample_ids"]),
        "heldout_observations": len(split["heldout_sample_ids"]),
        "train_trajectory_groups": len(split["train_groups"]),
        "heldout_trajectory_groups": len(split["heldout_groups"]),
        "probe": source_metadata["probe"],
        "dataset": source_metadata["dataset"],
        "models": {
            model: {
                **source_metadata["models"][model],
                "phase3_primary_node": node.upper(),
                "artifacts": copied[f"{model}/{node}"],
            }
            for model, node in (("openvla", "o2"), ("pi05", "p2"))
        },
        "results": source_metadata["results"],
        "phase2_status": {
            "action_predictive_signal": "PASS_STRONGLY_SUPPORTED",
            "seed7_probe_usability": "PASS_FIXED_CANDIDATE",
            "cross_seed_full_subspace_reproducibility": "FAIL_UNRESOLVED",
            "authoritative_phase2b_v3": "NOT_FROZEN_BLOCKED",
            "phase3_exploratory_pipeline_feasibility": "AUTHORIZED_FIXED_SEED7_ONLY",
        },
        "provenance": {
            "phase2a_directory": str(source),
            "phase2a_metadata_sha256": sha256_file(source_metadata_path),
            "phase2a_inventory_sha256": sha256_file(source_inventory_path),
            "stability_directory": str(stability),
            "stability_summary_sha256": sha256_file(stability_summary_path),
            "stability_status": stability_summary["status"],
            "collection_manifest": str(collection_manifest),
            "collection_manifest_sha256": sha256_file(collection_manifest),
            "collection_protocol_id": collection["protocol_id"],
            "openvla_representation_manifest": str(openvla_representation_manifest),
            "openvla_representation_manifest_sha256": sha256_file(
                openvla_representation_manifest
            ),
            "pi05_representation_manifest": str(pi05_representation_manifest),
            "pi05_representation_manifest_sha256": sha256_file(
                pi05_representation_manifest
            ),
            "pi05_p2_identity": {
                "definition_id": source_metadata["models"]["pi05"]
                ["representation_nodes"]["projected"]["definition_id"],
                "capture_semantics": source_metadata["models"]["pi05"]
                ["representation_nodes"]["projected"]["capture_semantics"],
            },
            "promotion": "byte-identical copy; no probe refitting",
            "source_artifacts_unchanged": True,
        },
    }
    atomic_json(output / "metadata.json", metadata)
    inventory = {
        str(path.relative_to(output)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    atomic_json(output / "artifact_inventory.json", inventory)
    return {
        "status": metadata["status"],
        "output_dir": str(output),
        "source_artifacts_unchanged": True,
        "copied_W_sha256": {
            "openvla/o2": copied["openvla/o2"]["W.pt"],
            "pi05/p2": copied["pi05/p2"]["W.pt"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        result = _run(args)
    except Exception as error:
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / "failure.json",
            {
                "status": "BLOCKED",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
