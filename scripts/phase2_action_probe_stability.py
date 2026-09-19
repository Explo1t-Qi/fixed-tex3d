#!/usr/bin/env python3
"""Run a CPU-only multi-seed action-probe/subspace stability diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for source_root in (ROBOT_ROOT, SCRIPTS_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import phase2_action_probe_materialize as materializer  # noqa: E402
from phase2_action_probe import (  # noqa: E402
    ACTION_NAMES,
    PROBE_SCHEMA_VERSION,
    ActionProbeError,
    ProbeConfig,
    atomic_json,
    build_group_split,
    fit_linear_probe,
    materialize_action_projection,
    prediction_metrics,
    sha256_file,
)
from phase2_action_probe_stability_core import (  # noqa: E402
    DEFAULT_SEEDS,
    REFERENCE_SEED,
    STABILITY_SCHEMA_VERSION,
    assert_tree_unchanged,
    classify_stability,
    pairwise_stability,
    tree_hashes,
    validate_seeds,
)


FROZEN_CONFIG = ProbeConfig(
    seed=REFERENCE_SEED,
    learning_rate=1e-3,
    weight_decay=1e-4,
    steps=2000,
    action_std_epsilon=1e-6,
    probe_reg=1e-4,
)
NODE_KEYS = (
    ("openvla", "projected", "o2", "OpenVLA O2"),
    ("openvla", "deep", "deep", "OpenVLA O-deep"),
    ("pi05", "projected", "p2", "PI0.5 P2"),
    ("pi05", "deep", "deep", "PI0.5 P-deep"),
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2a-dir", type=Path, required=True)
    parser.add_argument("--openvla-manifest", type=Path, required=True)
    parser.add_argument("--pi05-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args(argv)


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _predict(probe: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return probe(torch.from_numpy(features)).cpu().numpy()


def _validate_phase2a(
    directory: Path,
    *,
    openvla_manifest: Path,
    pi05_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != PROBE_SCHEMA_VERSION
        or metadata.get("stage") != "phase2a"
        or metadata.get("status") != "PHASE_2A_COMPLETE"
    ):
        raise ActionProbeError("source artifact is not a complete Phase 2A result")
    expected = {
        "learning_rate": FROZEN_CONFIG.learning_rate,
        "weight_decay": FROZEN_CONFIG.weight_decay,
        "steps": FROZEN_CONFIG.steps,
        "action_std_epsilon": FROZEN_CONFIG.action_std_epsilon,
        "probe_reg": FROZEN_CONFIG.probe_reg,
    }
    actual = metadata.get("probe", {})
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ActionProbeError(
            "Phase 2A probe hyperparameters differ from frozen config"
        )
    manifest_paths = {"openvla": openvla_manifest, "pi05": pi05_manifest}
    for model, path in manifest_paths.items():
        if metadata["models"][model]["representation_manifest_sha256"] != sha256_file(
            path
        ):
            raise ActionProbeError(
                f"{model} manifest does not match the Phase 2A source artifact"
            )
    return metadata, split


def _run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = validate_seeds(args.seeds)
    phase2a = args.phase2a_dir.expanduser().resolve(strict=True)
    openvla_manifest_path = args.openvla_manifest.expanduser().resolve(strict=True)
    pi05_manifest_path = args.pi05_manifest.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path is not None
        else None
    )
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")
    if phase2a == output or phase2a in output.parents:
        raise ActionProbeError(
            "diagnostic output must not be inside Phase 2A artifacts"
        )
    if report_path is not None and report_path.exists():
        raise FileExistsError(f"report path already exists: {report_path}")
    if report_path is not None and (
        report_path == phase2a or phase2a in report_path.parents
    ):
        raise ActionProbeError(
            "diagnostic report must not be inside Phase 2A artifacts"
        )

    metadata, frozen_split = _validate_phase2a(
        phase2a,
        openvla_manifest=openvla_manifest_path,
        pi05_manifest=pi05_manifest_path,
    )
    source_roots = {
        "phase2a": phase2a,
        "openvla_representations": openvla_manifest_path.parent,
        "pi05_representations": pi05_manifest_path.parent,
    }
    source_snapshots = {name: tree_hashes(path) for name, path in source_roots.items()}

    manifests = {
        "openvla": materializer._manifest(openvla_manifest_path, "openvla"),
        "pi05": materializer._manifest(pi05_manifest_path, "pi05"),
    }
    loaded = {
        model: materializer._load_model_data(manifest, model)
        for model, manifest in manifests.items()
    }
    identities = loaded["openvla"][0]
    if identities != loaded["pi05"][0]:
        raise ActionProbeError("OpenVLA and PI0.5 sample identities/order differ")
    split = build_group_split(identities)
    if split != frozen_split:
        raise ActionProbeError("diagnostic split differs from Phase 2A split")
    sample_index = {sample.sample_id: index for index, sample in enumerate(identities)}
    train_index = np.asarray([sample_index[x] for x in split["train_sample_ids"]])
    heldout_index = np.asarray([sample_index[x] for x in split["heldout_sample_ids"]])

    output.mkdir(parents=True)
    per_seed: dict[str, Any] = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "seeds": list(seeds),
        "nodes": {},
    }
    similarity_nodes: dict[str, Any] = {}

    for model, feature_key, artifact_node, display_name in NODE_KEYS:
        _, features, actions = loaded[model]
        node_features = features[feature_key]
        weights: dict[int, torch.Tensor] = {}
        heldout_mse: dict[int, float] = {}
        heldout_r2: dict[int, dict[str, float | None]] = {}
        node_seeds: dict[str, Any] = {}
        with np.load(
            phase2a / model / artifact_node / "action_stats.npz", allow_pickle=False
        ) as archive:
            source_stats = {key: archive[key].copy() for key in archive.files}
        source_weight = torch.load(
            phase2a / model / artifact_node / "W.pt",
            map_location="cpu",
            weights_only=True,
        )

        for seed in seeds:
            config = ProbeConfig(
                seed=seed,
                learning_rate=FROZEN_CONFIG.learning_rate,
                weight_decay=FROZEN_CONFIG.weight_decay,
                steps=FROZEN_CONFIG.steps,
                action_std_epsilon=FROZEN_CONFIG.action_std_epsilon,
                probe_reg=FROZEN_CONFIG.probe_reg,
            )
            probe, stats, history = fit_linear_probe(
                node_features[train_index],
                actions[train_index],
                config=config,
                device="cpu",
            )
            for key in ("mean", "std", "safe_std", "near_constant"):
                if not np.array_equal(stats[key], source_stats[key]):
                    raise ActionProbeError(
                        f"{display_name} TRAIN action statistics changed"
                    )
            train_prediction = _predict(probe, node_features[train_index])
            heldout_prediction = _predict(probe, node_features[heldout_index])
            train_metrics = prediction_metrics(
                train_prediction, actions[train_index], stats
            )
            heldout_metrics = prediction_metrics(
                heldout_prediction, actions[heldout_index], stats
            )
            weight = probe.weight.detach().float().cpu().clone()
            projection, projection_metrics = materialize_action_projection(
                weight, probe_reg=FROZEN_CONFIG.probe_reg
            )
            seed_dir = output / model / artifact_node / f"seed_{seed:06d}"
            seed_dir.mkdir(parents=True)
            torch.save(weight, seed_dir / "W.pt")
            torch.save(projection, seed_dir / "P_action.pt")
            seed_metrics = {
                "seed": seed,
                "train": train_metrics,
                "heldout": heldout_metrics,
                "training_initial_loss": history[0],
                "training_final_loss": history[-1],
                "rank_W": projection_metrics["rank_W"],
                "singular_values_W": projection_metrics["singular_values_W"],
                "projection": projection_metrics,
                "W_path": str((seed_dir / "W.pt").relative_to(output)),
                "P_action_path": str((seed_dir / "P_action.pt").relative_to(output)),
            }
            atomic_json(seed_dir / "metrics.json", seed_metrics)
            node_seeds[str(seed)] = seed_metrics
            weights[seed] = weight
            heldout_mse[seed] = float(heldout_metrics["normalized"]["mse"])
            heldout_r2[seed] = {
                name: heldout_metrics["normalized"]["per_dimension"][name]["r2"]
                for name in ACTION_NAMES
            }
            del projection, probe
            gc.collect()

        reference_weight_difference = float(
            torch.max(torch.abs(weights[REFERENCE_SEED] - source_weight)).item()
        )
        node_key = f"{model}/{artifact_node}"
        per_seed["nodes"][node_key] = {
            "display_name": display_name,
            "input_dimension": int(node_features.shape[1]),
            "phase2a_seed7_weight_max_abs_difference": reference_weight_difference,
            "seeds": node_seeds,
        }
        similarity_nodes[node_key] = {
            "display_name": display_name,
            "phase2a_seed7_weight_max_abs_difference": reference_weight_difference,
            **pairwise_stability(
                weights,
                heldout_mse,
                heldout_r2,
                probe_reg=FROZEN_CONFIG.probe_reg,
            ),
        }

    classification = classify_stability(similarity_nodes)
    for name, path in source_roots.items():
        assert_tree_unchanged(path, source_snapshots[name])

    similarity = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "nodes": similarity_nodes,
    }
    summary = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "status": classification["status"],
        "scope": "prediction and candidate action-predictive subspace initialization stability only",
        "code_commit": _head(),
        "phase2a_source": str(phase2a),
        "phase2a_metadata_sha256": sha256_file(phase2a / "metadata.json"),
        "representation_manifests": {
            "openvla": {
                "path": str(openvla_manifest_path),
                "sha256": sha256_file(openvla_manifest_path),
            },
            "pi05": {
                "path": str(pi05_manifest_path),
                "sha256": sha256_file(pi05_manifest_path),
            },
        },
        "dataset": metadata["dataset"],
        "split_rule": frozen_split["rule_id"],
        "train_observations": len(train_index),
        "heldout_observations": len(heldout_index),
        "seeds": list(seeds),
        "reference_seed": REFERENCE_SEED,
        "probe": {
            "architecture": "Linear(D,7,bias=False)",
            "learning_rate": FROZEN_CONFIG.learning_rate,
            "weight_decay": FROZEN_CONFIG.weight_decay,
            "steps": FROZEN_CONFIG.steps,
            "action_std_epsilon": FROZEN_CONFIG.action_std_epsilon,
            "probe_reg": FROZEN_CONFIG.probe_reg,
            "device": "cpu",
        },
        "classification": classification,
        "nodes": {
            key: {
                "display_name": value["display_name"],
                "heldout_mse": value["heldout_mse"],
                "heldout_mse_coefficient_of_variation": value[
                    "heldout_mse_coefficient_of_variation"
                ],
                "principal_angle_cosine": value["principal_angle_cosine"],
                "projection_relative_frobenius_distance": value[
                    "projection_relative_frobenius_distance"
                ],
                "phase2a_seed7_weight_max_abs_difference": value[
                    "phase2a_seed7_weight_max_abs_difference"
                ],
            }
            for key, value in similarity_nodes.items()
        },
        "source_artifacts_unchanged": True,
    }
    atomic_json(output / "per_seed_metrics.json", per_seed)
    atomic_json(output / "subspace_similarity.json", similarity)
    atomic_json(output / "summary.json", summary)
    _write_report(output / "phase2-action-probe-stability-report.md", summary)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_report(report_path, summary)

    for name, path in source_roots.items():
        assert_tree_unchanged(path, source_snapshots[name])
    return {
        "status": summary["status"],
        "output_dir": str(output),
        "seeds": list(seeds),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for value in summary["nodes"].values():
        mse = value["heldout_mse"]
        principal = value["principal_angle_cosine"]
        projection = value["projection_relative_frobenius_distance"]
        rows.append(
            f"| {value['display_name']} | {mse['mean']:.6f} | {mse['std']:.6f} | "
            f"{mse['min']:.6f} | {mse['max']:.6f} | "
            f"{principal['pairwise_mean']['mean']:.6f} | "
            f"{principal['pairwise_minimum']['min']:.6f} | "
            f"{projection['mean']:.6f} | {projection['max']:.6f} | "
            f"{value['phase2a_seed7_weight_max_abs_difference']:.3g} |"
        )
    failed = [
        name
        for name, value in summary["classification"]["nodes"].items()
        if not value["stable"]
    ]
    interpretation = (
        "Held-out performance and the fitted row spaces satisfy the predeclared "
        "stability thresholds for every node. The current dataset can produce "
        "stable candidate action-predictive subspaces under this probe protocol."
        if not failed
        else "Held-out performance may remain stable while at least one fitted row "
        "space fails a predeclared stability threshold. The representation contains "
        "predictive signal, but the corresponding high-dimensional direction is "
        "under-determined and should be reviewed before Phase 2B freeze."
    )
    text = f"""# Phase 2 — Action-Probe Stability Diagnostic

## Scope

This CPU-only diagnostic measures prediction stability and candidate action-predictive subspace stability across random probe initializations. It does not establish causal action relevance, action controllability, texture effectiveness, or transferability, and it does not modify the Phase 2A/2B protocol.

## Frozen protocol

- Dataset: frozen Pilot v0.2, 200 observations.
- Split: `pilot-v0.2-c5-split-v1`, 160 TRAIN and 40 HELD-OUT observations.
- Probe: `Linear(D,7,bias=False)`.
- AdamW: learning rate `1e-3`, weight decay `1e-4`, 2,000 steps.
- Action normalization: TRAIN statistics only, epsilon `1e-6`.
- Projection ridge: `1e-4`.
- Seeds: {summary["seeds"]}.

## Aggregate stability

| Node | Held-out MSE mean | std | min | max | Mean principal cosine | Minimum principal cosine | Projection distance mean | max | Phase 2A seed-7 W max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Detailed per-seed metrics, signed action-row cosines, all seven principal-angle cosines for every pair, low-rank projection distances, and comparisons against seed 7 are stored in `per_seed_metrics.json` and `subspace_similarity.json`.

## Decision rule

The predeclared `STABLE` rule requires every node to have held-out MSE coefficient of variation at most 0.05, minimum pairwise principal cosine at least 0.90, and maximum projection relative Frobenius distance at most 0.25. These thresholds classify reproducibility; they do not tune the probe or select a node from HELD-OUT performance.

## Result

Status: `{summary["status"]}`. Nodes requiring review: {failed or "none"}.

{interpretation}

The source artifact trees were hash-snapshotted before and after the diagnostic and remained unchanged.
"""
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    output_preexisted = output.exists()
    try:
        result = _run(args)
    except Exception as error:
        if not output_preexisted:
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(
                output / "failure.json",
                {
                    "status": "BLOCKED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
