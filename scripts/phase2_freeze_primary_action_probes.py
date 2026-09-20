#!/usr/bin/env python3
"""Promote validated Phase 2A O2/P2 seed-7 probes to Phase 2B authority."""

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
from phase3_action_predictive_objective import PHASE2B_SCHEMA  # noqa: E402


class PrimaryProbeFreezeError(RuntimeError):
    """Raised when Phase 2A inputs do not meet the frozen Phase 2B contract."""


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2a-dir", type=Path, required=True)
    parser.add_argument("--stability-dir", type=Path, required=True)
    parser.add_argument(
        "--openvla-source-phase2b",
        type=Path,
        required=True,
        help="Existing validated Phase 2B v2 tree supplying unchanged OpenVLA O2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_probe_contract(metadata: dict[str, Any]) -> None:
    expected = {
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
    probe = metadata.get("probe", {})
    mismatches = {
        key: {"expected": expected_value, "actual": probe.get(key)}
        for key, expected_value in expected.items()
        if probe.get(key) != expected_value
    }
    if mismatches:
        raise PrimaryProbeFreezeError(f"Phase 2A probe contract mismatch: {mismatches}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.phase2a_dir.expanduser().resolve(strict=True)
    stability = args.stability_dir.expanduser().resolve(strict=True)
    openvla_source = args.openvla_source_phase2b.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")
    source_metadata_path = source / "metadata.json"
    source_inventory_path = source / "artifact_inventory.json"
    stability_summary_path = stability / "summary.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_inventory = json.loads(source_inventory_path.read_text(encoding="utf-8"))
    stability_summary = json.loads(stability_summary_path.read_text(encoding="utf-8"))
    openvla_metadata_path = openvla_source / "metadata.json"
    openvla_inventory_path = openvla_source / "artifact_inventory.json"
    openvla_metadata = json.loads(openvla_metadata_path.read_text(encoding="utf-8"))
    openvla_inventory = json.loads(openvla_inventory_path.read_text(encoding="utf-8"))
    if (
        source_metadata.get("status") != "PHASE_2A_COMPLETE"
        or source_metadata.get("stage") != "phase2a"
    ):
        raise PrimaryProbeFreezeError("source is not a completed Phase 2A artifact")
    _require_probe_contract(source_metadata)
    if source_metadata.get("selected_nodes") != ["pi05/p2"]:
        raise PrimaryProbeFreezeError(
            "corrected Phase 2A source must contain only PI0.5 P2"
        )
    corrected_projected = (
        source_metadata.get("models", {})
        .get("pi05", {})
        .get("representation_nodes", {})
        .get("projected", {})
    )
    if (
        corrected_projected.get("definition_id")
        != "pi05_p2_embed_image_no_manual_scaling_v2"
    ):
        raise PrimaryProbeFreezeError(
            "Phase 2A PI0.5 P2 definition is not runtime-compatible"
        )
    split = json.loads((source / "split.json").read_text(encoding="utf-8"))
    if (
        split.get("rule_id") != "pilot-v0.2-c5-split-v1"
        or len(split.get("train_sample_ids", [])) != 160
        or len(split.get("heldout_sample_ids", [])) != 40
    ):
        raise PrimaryProbeFreezeError("Phase 2A split contract mismatch")
    openvla_split = json.loads(
        (openvla_source / "split.json").read_text(encoding="utf-8")
    )
    if split != openvla_split:
        raise PrimaryProbeFreezeError(
            "corrected PI0.5 and existing OpenVLA splits differ"
        )
    if (
        openvla_metadata.get("schema_version") != "phase2b_primary_action_probes_v1"
        or openvla_metadata.get("status") != "PHASE_2B_PRIMARY_FROZEN"
    ):
        raise PrimaryProbeFreezeError("OpenVLA source is not historical Phase 2B v2")
    source_metadata_hash = sha256_file(source_metadata_path)
    if stability_summary.get("phase2a_metadata_sha256") != source_metadata_hash:
        raise PrimaryProbeFreezeError(
            "stability diagnostic references another Phase 2A"
        )
    if not stability_summary.get("source_artifacts_unchanged"):
        raise PrimaryProbeFreezeError("stability source-artifact invariant failed")
    if stability_summary.get(
        "status"
    ) != "CORRECTED_P2_STABLE" or stability_summary.get("selected_nodes") != [
        "pi05/p2"
    ]:
        raise PrimaryProbeFreezeError("corrected PI0.5 P2 stability did not pass")
    node = stability_summary.get("nodes", {}).get("pi05/p2", {})
    if node.get("phase2a_seed7_weight_max_abs_difference") != 0.0:
        raise PrimaryProbeFreezeError("seed-7 W mismatch for pi05/p2")

    output.mkdir(parents=True)
    shutil.copy2(source / "split.json", output / "split.json")
    copied: dict[str, dict[str, str]] = {}
    for model, node in (("openvla", "o2"), ("pi05", "p2")):
        artifact_source = openvla_source if model == "openvla" else source
        artifact_inventory = (
            openvla_inventory if model == "openvla" else source_inventory
        )
        destination = output / model / node
        destination.mkdir(parents=True)
        copied[f"{model}/{node}"] = {}
        for filename in (
            "probe.pt",
            "W.pt",
            "P_action.pt",
            "action_stats.npz",
            "metrics.json",
            "training_history.npy",
        ):
            relative = f"{model}/{node}/{filename}"
            source_file = artifact_source / relative
            source_hash = sha256_file(source_file)
            if artifact_inventory.get(relative, {}).get("sha256") != source_hash:
                raise PrimaryProbeFreezeError(f"source hash mismatch: {relative}")
            shutil.copy2(source_file, destination / filename)
            if sha256_file(destination / filename) != source_hash:
                raise PrimaryProbeFreezeError(f"copy changed artifact: {relative}")
            copied[f"{model}/{node}"][filename] = source_hash

    for model, node, width in (("openvla", "o2", 4096), ("pi05", "p2", 2048)):
        weight = torch.load(
            output / model / node / "W.pt", map_location="cpu", weights_only=True
        )
        if tuple(weight.shape) != (7, width) or not bool(torch.isfinite(weight).all()):
            raise PrimaryProbeFreezeError(f"invalid frozen W for {model}/{node}")

    metadata = {
        "schema_version": PHASE2B_SCHEMA,
        "status": "PHASE_2B_PRIMARY_FROZEN",
        "scope": "O2/P2 candidate action-predictive probes only; no causal action relevance claim",
        "code_commit": _head(),
        "seed": 7,
        "split_rule": "pilot-v0.2-c5-split-v1",
        "train_observations": 160,
        "heldout_observations": 40,
        "probe": source_metadata["probe"],
        "dataset": source_metadata["dataset"],
        "models": {
            "openvla": {
                **openvla_metadata["models"]["openvla"],
                "phase3_primary_node": "o2",
                "artifacts": copied["openvla/o2"],
            },
            "pi05": {
                **source_metadata["models"]["pi05"],
                "phase3_primary_node": "p2",
                "artifacts": copied["pi05/p2"],
            },
        },
        "provenance": {
            "phase2a_directory": str(source),
            "phase2a_metadata_sha256": source_metadata_hash,
            "phase2a_inventory_sha256": sha256_file(source_inventory_path),
            "stability_directory": str(stability),
            "stability_summary_sha256": sha256_file(stability_summary_path),
            "stability_status": stability_summary["status"],
            "selection_decision": (
                "OpenVLA O2 is copied byte-identically from historical Phase 2B v2. "
                "PI0.5 P2 uses the corrected current embed_image() representation and "
                "passed the frozen six-seed stability thresholds."
            ),
            "openvla_source_phase2b": str(openvla_source),
            "openvla_source_metadata_sha256": sha256_file(openvla_metadata_path),
            "openvla_source_inventory_sha256": sha256_file(openvla_inventory_path),
            "historical_pi05_W_sha256": openvla_metadata["models"]["pi05"]["artifacts"][
                "W.pt"
            ],
            "corrected_pi05_W_sha256": copied["pi05/p2"]["W.pt"],
            "openvla_W_unchanged": (
                copied["openvla/o2"]["W.pt"]
                == openvla_metadata["models"]["openvla"]["artifacts"]["W.pt"]
            ),
        },
        "source_artifacts_unchanged": True,
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
        "openvla_W_sha256": copied["openvla/o2"]["W.pt"],
        "pi05_W_sha256": copied["pi05/p2"]["W.pt"],
        "source_artifacts_unchanged": True,
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
