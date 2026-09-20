#!/usr/bin/env python3
"""Fit, evaluate, and freeze Phase 2 action-predictive linear probes."""

from __future__ import annotations

import argparse
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
ALL_NODE_KEYS = ("openvla/o2", "openvla/deep", "pi05/p2", "pi05/deep")
CORRECTED_PI05_MANIFEST_SCHEMA = "phase2_action_representation_manifest_v2"
CORRECTED_PI05_DEFINITION_ID = "pi05_p2_embed_image_no_manual_scaling_v2"
EXPANDED_MANIFEST_SCHEMA = "phase2_action_representation_manifest_v3"
EXPANDED_ARCHIVE_SCHEMA = "phase2_action_representation_archive_v3"
if str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))

from phase2_action_probe import (  # noqa: E402
    NODE_WIDTHS,
    PROBE_SCHEMA_VERSION,
    EXPANDED_PROBE_SCHEMA_VERSION,
    ActionProbeError,
    ProbeConfig,
    SampleIdentity,
    action_distribution_audit,
    atomic_json,
    build_group_split,
    config_dict,
    feature_matrix_diagnostics,
    fit_linear_probe,
    materialize_action_projection,
    mean_pool_features,
    prediction_metrics,
    sha256_file,
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-manifest", type=Path, required=True)
    parser.add_argument("--pi05-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("phase2a", "phase2b"),
        required=True,
        help="Phase 2A produces candidate diagnostics; Phase 2B freezes reusable artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional second report copy, such as docs/phase2-action-predictive-representation-report.md",
    )
    parser.add_argument("--probe-device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--action-std-epsilon", type=float, default=1e-6)
    parser.add_argument("--probe-reg", type=float, default=1e-4)
    parser.add_argument(
        "--nodes",
        nargs="+",
        choices=ALL_NODE_KEYS,
        default=list(ALL_NODE_KEYS),
        help="Fit only the listed nodes; defaults to the historical four-node run.",
    )
    return parser.parse_args(argv)


def _head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest(path: Path, model: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        value.get("schema_version")
        not in {
            "phase2_action_representation_manifest_v1",
            CORRECTED_PI05_MANIFEST_SCHEMA,
            EXPANDED_MANIFEST_SCHEMA,
        }
        or value.get("status") != "COMPLETE"
        or value.get("model") != model
        or type(value.get("count")) is not int
        or value.get("count") <= 0
        or len(value.get("records", [])) != value.get("count")
    ):
        raise ActionProbeError(f"invalid {model} representation manifest")
    schema = value["schema_version"]
    if schema != EXPANDED_MANIFEST_SCHEMA and value.get("count") != 200:
        raise ActionProbeError(f"legacy {model} manifest must contain 200 records")
    if schema == EXPANDED_MANIFEST_SCHEMA:
        protocol = value.get("dataset_protocol")
        groups_per_task = (
            protocol.get("accepted_groups_per_task")
            if isinstance(protocol, dict)
            else None
        )
        progress = (
            protocol.get("target_relative_progress")
            if isinstance(protocol, dict)
            else None
        )
        if (
            not isinstance(protocol, dict)
            or not isinstance(groups_per_task, dict)
            or set(groups_per_task) != {str(index) for index in range(10)}
            or any(
                type(count) is not int or count < 2
                for count in groups_per_task.values()
            )
            or not isinstance(progress, list)
            or len(set(progress)) != len(progress)
            or protocol.get("collection_schema_version")
            != "pilot_v0_3_expanded_collection_v1"
            or protocol.get("protocol_id") != "pilot-v0.3-expanded-v1"
            or protocol.get("split_rule_id") != "pilot-v0.3-expanded-split-v1"
            or protocol.get("observation_count") != value.get("count")
            or protocol.get("trajectory_group_count") != sum(groups_per_task.values())
            or protocol.get("observations_per_group") != len(progress)
            or value.get("count")
            != sum(groups_per_task.values()) * protocol.get("observations_per_group", 0)
        ):
            raise ActionProbeError(f"invalid expanded {model} dataset protocol")
    if (
        schema in {CORRECTED_PI05_MANIFEST_SCHEMA, EXPANDED_MANIFEST_SCHEMA}
        and model == "pi05"
    ):
        projected = value.get("representation_nodes", {}).get("projected", {})
        if (
            projected.get("definition_id") != CORRECTED_PI05_DEFINITION_ID
            or projected.get("capture_semantics")
            != "current PI0Pytorch base-camera embed_image() output; no additional manual scaling"
            or value.get("pi05_p2_identity", {}).get("identity_pass") is not True
            or value.get("action_identity", {}).get("pass") is not True
        ):
            raise ActionProbeError("invalid corrected PI0.5 P2 representation identity")
    value["_path"] = resolved
    return value


def _load_model_data(
    manifest: dict[str, Any], model: str
) -> tuple[list[SampleIdentity], dict[str, np.ndarray], np.ndarray]:
    widths = {
        "projected": NODE_WIDTHS[(model, "o2" if model == "openvla" else "p2")],
        "deep": NODE_WIDTHS[(model, "deep")],
    }
    pooled: dict[str, list[np.ndarray]] = {"projected": [], "deep": []}
    actions: list[np.ndarray] = []
    samples: list[SampleIdentity] = []
    parent = manifest["_path"].parent
    for entry in manifest["records"]:
        path = (parent / entry["archive"]).resolve(strict=True)
        if sha256_file(path) != entry["sha256"]:
            raise ActionProbeError(f"archive hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata_json", "projected", "deep", "action"}:
                raise ActionProbeError(f"invalid archive keys: {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (
                metadata["sample_id"] != entry["sample_id"]
                or metadata["model"] != model
            ):
                raise ActionProbeError(f"archive identity mismatch: {path}")
            source_hash = metadata.get("source_observation_sha256")
            if (
                not isinstance(source_hash, str)
                or not source_hash.startswith("sha256:")
                or len(source_hash) != 71
            ):
                raise ActionProbeError(f"invalid source observation hash: {path}")
            schema = manifest.get("schema_version")
            expected_archive_schema = (
                EXPANDED_ARCHIVE_SCHEMA
                if schema == EXPANDED_MANIFEST_SCHEMA
                else "phase2_action_representation_archive_v2"
            )
            if (
                schema
                in {
                    CORRECTED_PI05_MANIFEST_SCHEMA,
                    EXPANDED_MANIFEST_SCHEMA,
                }
                and model == "pi05"
                and (
                    metadata.get("schema_version") != expected_archive_schema
                    or metadata.get("representation_definition_id")
                    != CORRECTED_PI05_DEFINITION_ID
                )
            ):
                raise ActionProbeError(
                    f"corrected PI0.5 archive schema mismatch: {path}"
                )
            if (
                schema == EXPANDED_MANIFEST_SCHEMA
                and model == "openvla"
                and metadata.get("schema_version") != EXPANDED_ARCHIVE_SCHEMA
            ):
                raise ActionProbeError(
                    f"expanded OpenVLA archive schema mismatch: {path}"
                )
            samples.append(
                SampleIdentity(
                    sample_id=metadata["sample_id"],
                    task_id=int(metadata["task_id"]),
                    initial_state_id=int(metadata["initial_state_id"]),
                    target_progress=float(metadata["target_progress"]),
                    source_observation_sha256=source_hash,
                )
            )
            for key, width in widths.items():
                value = np.asarray(archive[key], dtype=np.float32)
                pooled[key].append(mean_pool_features(value[None], width=width)[0])
            action = np.asarray(archive["action"], dtype=np.float32)
            if action.shape != (7,) or not np.all(np.isfinite(action)):
                raise ActionProbeError(f"invalid action: {path}")
            actions.append(action)
    return (
        samples,
        {key: np.stack(value) for key, value in pooled.items()},
        np.stack(actions),
    )


def _predict(probe: torch.nn.Module, value: np.ndarray) -> np.ndarray:
    device = next(probe.parameters()).device
    with torch.no_grad():
        return probe(torch.from_numpy(value).to(device)).cpu().numpy()


def _node_name(model: str, key: str) -> str:
    return ("o2" if model == "openvla" else "p2") if key == "projected" else "deep"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")
    second_report = (
        args.report_path.expanduser().resolve()
        if args.report_path is not None
        else None
    )
    if second_report is not None and second_report.exists():
        raise FileExistsError(f"report path already exists: {second_report}")
    config = ProbeConfig(
        seed=args.seed,
        learning_rate=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        steps=args.probe_steps,
        action_std_epsilon=args.action_std_epsilon,
        probe_reg=args.probe_reg,
    )
    config.validate()
    selected_nodes = tuple(getattr(args, "nodes", ALL_NODE_KEYS))
    if len(set(selected_nodes)) != len(selected_nodes):
        raise ActionProbeError("selected probe nodes must be unique")
    device = torch.device(args.probe_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ActionProbeError("requested probe CUDA device is unavailable")
    manifests = {
        "openvla": _manifest(args.openvla_manifest, "openvla"),
        "pi05": _manifest(args.pi05_manifest, "pi05"),
    }
    loaded = {
        model: _load_model_data(manifest, model)
        for model, manifest in manifests.items()
    }
    identities = loaded["openvla"][0]
    if identities != loaded["pi05"][0]:
        raise ActionProbeError("OpenVLA and PI0.5 sample identities/order differ")
    if (
        manifests["openvla"]["collection_manifest_sha256"]
        != manifests["pi05"]["collection_manifest_sha256"]
    ):
        raise ActionProbeError(
            "representation manifests use different source collections"
        )
    dataset_protocol = manifests["openvla"].get("dataset_protocol")
    if dataset_protocol != manifests["pi05"].get("dataset_protocol"):
        raise ActionProbeError(
            "representation manifests use different dataset protocols"
        )
    expanded_dataset = all(
        manifest.get("schema_version") == EXPANDED_MANIFEST_SCHEMA
        for manifest in manifests.values()
    )
    split_rule = (
        dataset_protocol["split_rule_id"]
        if dataset_protocol is not None
        else "pilot-v0.2-c5-split-v1"
    )
    heldout_fraction = (
        float(dataset_protocol["heldout_fraction_per_task"])
        if dataset_protocol is not None
        else 0.20
    )
    split = build_group_split(
        identities,
        rule_id=split_rule,
        heldout_fraction_per_task=heldout_fraction,
    )
    probe_schema = (
        EXPANDED_PROBE_SCHEMA_VERSION if expanded_dataset else PROBE_SCHEMA_VERSION
    )
    sample_index = {sample.sample_id: index for index, sample in enumerate(identities)}
    train_index = np.asarray(
        [sample_index[value] for value in split["train_sample_ids"]]
    )
    held_index = np.asarray(
        [sample_index[value] for value in split["heldout_sample_ids"]]
    )

    output.mkdir(parents=True)
    audit_dir = output / "action_audit"
    audit_dir.mkdir()
    atomic_json(output / "split.json", split)
    result_rows: list[dict[str, Any]] = []
    action_audits: dict[str, Any] = {}

    for model in ("openvla", "pi05"):
        samples, features, actions = loaded[model]
        audit = action_distribution_audit(actions, samples)
        action_audits[model] = audit
        atomic_json(audit_dir / f"{model}.json", audit)
        for key in ("projected", "deep"):
            node = _node_name(model, key)
            if f"{model}/{node}" not in selected_nodes:
                continue
            node_dir = output / model / node
            node_dir.mkdir(parents=True)
            probe, stats, history = fit_linear_probe(
                features[key][train_index],
                actions[train_index],
                config=config,
                device=device,
            )
            train_prediction = _predict(probe, features[key][train_index])
            held_prediction = _predict(probe, features[key][held_index])
            zero_train = np.zeros_like(actions[train_index], dtype=np.float32)
            zero_held = np.zeros_like(actions[held_index], dtype=np.float32)
            metrics = {
                "train": prediction_metrics(
                    train_prediction, actions[train_index], stats
                ),
                "heldout": prediction_metrics(
                    held_prediction, actions[held_index], stats
                ),
                "mean_action_baseline": {
                    "train": prediction_metrics(
                        zero_train, actions[train_index], stats
                    ),
                    "heldout": prediction_metrics(
                        zero_held, actions[held_index], stats
                    ),
                },
                "optimization": {
                    "initial_mse": history[0],
                    "final_mse": history[-1],
                    "steps": len(history),
                    "finite": bool(np.all(np.isfinite(history))),
                },
            }
            weight = probe.weight.detach().float().cpu()
            matrix_diagnostics = feature_matrix_diagnostics(
                features[key][train_index], weight=weight
            )
            metrics["feature_matrix"] = matrix_diagnostics
            metrics["train_heldout_mse_gap_normalized"] = float(
                metrics["heldout"]["normalized"]["mse"]
                - metrics["train"]["normalized"]["mse"]
            )
            projection, projection_metrics = materialize_action_projection(
                weight, probe_reg=config.probe_reg
            )
            state = {
                "schema_version": probe_schema,
                "state_dict": {"weight": weight},
                "input_dimension": int(weight.shape[1]),
                "output_dimension": 7,
                "bias": False,
                "config": config_dict(config),
            }
            torch.save(state, node_dir / "probe.pt")
            torch.save(weight, node_dir / "W.pt")
            torch.save(projection, node_dir / "P_action.pt")
            np.savez(
                node_dir / "action_stats.npz",
                mean=stats["mean"],
                std=stats["std"],
                safe_std=stats["safe_std"],
                near_constant=stats["near_constant"],
            )
            np.save(
                node_dir / "training_history.npy", np.asarray(history, dtype=np.float32)
            )
            atomic_json(
                node_dir / "metrics.json", {**metrics, "projection": projection_metrics}
            )

            loaded_weight = torch.load(
                node_dir / "W.pt", map_location="cpu", weights_only=True
            )
            loaded_projection = torch.load(
                node_dir / "P_action.pt", map_location="cpu", weights_only=True
            )
            if not torch.equal(weight, loaded_weight) or not torch.equal(
                projection, loaded_projection
            ):
                raise ActionProbeError("W/P_action changed after serialization")

            reloaded = torch.load(
                node_dir / "probe.pt", map_location="cpu", weights_only=True
            )
            restored = torch.nn.Linear(weight.shape[1], 7, bias=False)
            restored.load_state_dict(reloaded["state_dict"])
            before = torch.from_numpy(held_prediction)
            after = restored(torch.from_numpy(features[key][held_index])).detach()
            reload_max_abs = float((before - after).abs().max())
            if reload_max_abs > 1e-6:
                raise ActionProbeError("probe predictions changed after serialization")
            result_rows.append(
                {
                    "model": model,
                    "node": node,
                    "dimension": int(weight.shape[1]),
                    "train_mse_normalized": metrics["train"]["normalized"]["mse"],
                    "heldout_mse_normalized": metrics["heldout"]["normalized"]["mse"],
                    "heldout_mean_baseline_mse_normalized": metrics[
                        "mean_action_baseline"
                    ]["heldout"]["normalized"]["mse"],
                    "train_mae_normalized": metrics["train"]["normalized"]["mae"],
                    "heldout_mae_normalized": metrics["heldout"]["normalized"]["mae"],
                    "probe_reload_max_abs": reload_max_abs,
                    "rank_W": projection_metrics["rank_W"],
                    "feature_matrix_rank": matrix_diagnostics["rank"],
                    "feature_matrix_condition_number": matrix_diagnostics[
                        "effective_condition_number"
                    ],
                    "feature_matrix_nullspace_dimension": matrix_diagnostics[
                        "nullspace_dimension"
                    ],
                    "probe_weight_nullspace_norm_fraction": matrix_diagnostics[
                        "probe_weight_nullspace_norm_fraction"
                    ],
                    "train_heldout_mse_gap_normalized": metrics[
                        "train_heldout_mse_gap_normalized"
                    ],
                }
            )

    metadata = {
        "schema_version": probe_schema,
        "status": "PHASE_2A_COMPLETE"
        if args.stage == "phase2a"
        else "PHASE_2_COMPLETE",
        "stage": args.stage,
        "artifact_authority": (
            "candidate_pending_dataset_sufficiency_decision"
            if args.stage == "phase2a"
            else "frozen_phase2_action_predictive_representation"
        ),
        "scope": "action predictability only; no causal action relevance, texture effectiveness, or transferability claim",
        "selected_nodes": list(selected_nodes),
        "code_commit": _head(),
        "dataset": {
            "materialization": (
                "Pilot v0.3 expanded manifest-driven collection"
                if expanded_dataset
                else "Pilot v0.2 frozen 200-observation collection"
            ),
            "collection_manifest_sha256": manifests["openvla"][
                "collection_manifest_sha256"
            ],
            "protocol": dataset_protocol,
            "observation_count": len(identities),
            "trajectory_group_count": len(
                {(sample.task_id, sample.initial_state_id) for sample in identities}
            ),
            "accepted_groups_per_task": {
                str(task): len(
                    {
                        sample.initial_state_id
                        for sample in identities
                        if sample.task_id == task
                    }
                )
                for task in sorted({sample.task_id for sample in identities})
            },
            "state_source": "successful OpenVLA on-policy trajectories",
        },
        "split": {
            "rule_id": split["rule_id"],
            "train_groups": split["train_groups"],
            "heldout_groups": split["heldout_groups"],
            "counts": split.get("counts"),
        },
        "probe": {
            **config_dict(config),
            "architecture": "Linear(D,7,bias=False)",
            "pooling": "mean over 256 visual tokens",
            "target": "model-specific deployed 7-D clean action",
        },
        "models": {
            model: {
                "backend": manifests[model]["backend"],
                "model_identity": manifests[model]["model_identity"],
                "checkpoint_path": manifests[model]["checkpoint_path"],
                "representation_manifest": str(manifests[model]["_path"]),
                "representation_manifest_sha256": sha256_file(
                    manifests[model]["_path"]
                ),
                "projected_node": manifests[model]["projected_node"],
                "deep_node": manifests[model]["deep_node"],
                "representation_nodes": manifests[model]["representation_nodes"],
                "action_semantics": manifests[model]["action_target_semantics"],
                "saved_representation_dtype": manifests[model]["saved_dtype"],
                "token_count": manifests[model]["token_count"],
            }
            for model in ("openvla", "pi05")
        },
        "results": result_rows,
    }
    atomic_json(output / "metadata.json", metadata)
    report_path = output / "phase2-action-predictive-representation-report.md"
    _write_report(report_path, metadata)
    if second_report is not None:
        second_report.parent.mkdir(parents=True, exist_ok=True)
        second_report.write_bytes(report_path.read_bytes())
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
        "stage": args.stage,
        "output_dir": str(output),
        "results": result_rows,
    }


def _write_report(path: Path, metadata: dict[str, Any]) -> None:
    rows = metadata["results"]
    selected_nodes = set(metadata["selected_nodes"])
    table = "\n".join(
        f"| {r['model']} | {r['node']} | {r['dimension']} | "
        f"{r['feature_matrix_rank']} | {r['feature_matrix_condition_number']:.6g} | "
        f"{r['train_mse_normalized']:.6g} | {r['heldout_mse_normalized']:.6g} | "
        f"{r['heldout_mean_baseline_mse_normalized']:.6g} | "
        f"{r['train_heldout_mse_gap_normalized']:.6g} | "
        f"{r['probe_weight_nullspace_norm_fraction']:.6g} |"
        for r in rows
    )
    split_counts = metadata["split"].get("counts")
    if split_counts is None:
        observations_per_group = 4
        split_counts = {
            "train_groups": len(metadata["split"]["train_groups"]),
            "heldout_groups": len(metadata["split"]["heldout_groups"]),
            "train_observations": len(metadata["split"]["train_groups"])
            * observations_per_group,
            "heldout_observations": len(metadata["split"]["heldout_groups"])
            * observations_per_group,
        }
    node_lines = []
    if "openvla/o2" in selected_nodes:
        node_lines.append("- OpenVLA O2: multimodal-projector output, `[256,4096]`.")
    if "openvla/deep" in selected_nodes:
        node_lines.append(
            f"- OpenVLA O-deep: `{metadata['models']['openvla']['deep_node']['module_path']}` output, visual slice `{metadata['models']['openvla']['deep_node']['visual_token_slice']}`."
        )
    if "pi05/p2" in selected_nodes:
        node_lines.append(
            "- PI0.5 P2: current base-camera `embed_image()` output, `[256,2048]`, with no additional manual scaling."
        )
    if "pi05/deep" in selected_nodes:
        node_lines.append(
            f"- PI0.5 P-deep: `{metadata['models']['pi05']['deep_node']['module_path']}` output, visual slice `{metadata['models']['pi05']['deep_node']['visual_token_slice']}`."
        )
    text = f"""# Phase 2 — Action-Predictive Representation Report

## 1. Scope

This experiment evaluates action predictability only. It does not establish causal action relevance, texture effectiveness, or transferability. Materialization stage: `{metadata["stage"]}`; artifact authority: `{metadata["artifact_authority"]}`.

## 2. Dataset

The experiment uses {metadata["dataset"]["observation_count"]} observations from {metadata["dataset"]["trajectory_group_count"]} successful clean OpenVLA on-policy LIBERO-Spatial trajectory groups. Split `{metadata["split"]["rule_id"]}` is deterministic, per-task stratified, and group-aware: {split_counts["train_groups"]} TRAIN groups / {split_counts["train_observations"]} observations and {split_counts["heldout_groups"]} HELD-OUT groups / {split_counts["heldout_observations"]} observations. Model-specific action audits are stored under `action_audit/`.

## 3. Representation Nodes

{chr(10).join(node_lines)}

## 4. Probe

Each node is mean-pooled across all 256 visual tokens. A separate `Linear(D, 7, bias=False)` probe is fitted with AdamW against model-specific TRAIN-normalized deployed actions. Action statistics use TRAIN only. Hyperparameters are recorded in `metadata.json`.

## 5. Results

| Model | Node | D | TRAIN rank | Condition | Train MSE | Held-out MSE | Mean baseline MSE | Gap | W null fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

Per-action MSE, MAE, Pearson correlation, and R², plus singular-value diagnostics and W rank, are stored in each node's `metrics.json`.

## 6. Interpretation

The table measures which representation contains more linearly readable action-predictive information under this fixed protocol. Weak or negative held-out evidence remains a valid scientific result. These measurements do not establish causal action relevance or controllability.

## 7. Decision Boundary

Each node directory contains candidate `probe.pt`, `W.pt`, `P_action.pt`, TRAIN action normalization in `action_stats.npz`, and `metrics.json`. No Phase 2B freeze or Phase 3 step is implied by this Phase 2A materialization.
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
