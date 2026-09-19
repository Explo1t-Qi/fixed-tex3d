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
if str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))

from phase2_action_probe import (  # noqa: E402
    NODE_WIDTHS,
    PROBE_SCHEMA_VERSION,
    ActionProbeError,
    ProbeConfig,
    SampleIdentity,
    action_distribution_audit,
    atomic_json,
    build_group_split,
    config_dict,
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
        value.get("schema_version") != "phase2_action_representation_manifest_v1"
        or value.get("status") != "COMPLETE"
        or value.get("model") != model
        or value.get("count") != 200
        or len(value.get("records", [])) != 200
    ):
        raise ActionProbeError(f"invalid {model} representation manifest")
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
            samples.append(
                SampleIdentity(
                    sample_id=metadata["sample_id"],
                    task_id=int(metadata["task_id"]),
                    initial_state_id=int(metadata["initial_state_id"]),
                    target_progress=float(metadata["target_progress"]),
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
    split = build_group_split(identities)
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
            projection, projection_metrics = materialize_action_projection(
                weight, probe_reg=config.probe_reg
            )
            state = {
                "schema_version": PROBE_SCHEMA_VERSION,
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
                }
            )

    metadata = {
        "schema_version": PROBE_SCHEMA_VERSION,
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
        "code_commit": _head(),
        "dataset": {
            "materialization": "Pilot v0.2 frozen 200-observation collection",
            "collection_manifest_sha256": manifests["openvla"][
                "collection_manifest_sha256"
            ],
            "observation_count": 200,
            "trajectory_group_count": 50,
            "state_source": "successful OpenVLA on-policy trajectories",
        },
        "split": {
            "rule_id": split["rule_id"],
            "train_groups": split["train_groups"],
            "heldout_groups": split["heldout_groups"],
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
    table = "\n".join(
        f"| {r['model']} | {r['node']} | {r['dimension']} | {r['train_mse_normalized']:.6g} | {r['heldout_mse_normalized']:.6g} | {r['heldout_mean_baseline_mse_normalized']:.6g} |"
        for r in rows
    )
    text = f"""# Phase 2 — Action-Predictive Representation Report

## 1. Scope

This experiment evaluates action predictability only. It does not establish causal action relevance, texture effectiveness, or transferability. Materialization stage: `{metadata["stage"]}`; artifact authority: `{metadata["artifact_authority"]}`.

## 2. Dataset

The experiment uses 200 frozen Pilot v0.2 observations from 50 successful OpenVLA on-policy LIBERO-Spatial trajectory groups. The deterministic group-aware split contains 40 TRAIN groups (160 observations) and 10 HELD-OUT groups (40 observations), with one held-out group per task. Model-specific action audits are stored under `action_audit/`.

## 3. Representation Nodes

- OpenVLA O2: multimodal-projector output, `[256,4096]`.
- OpenVLA O-deep: `{metadata["models"]["openvla"]["deep_node"]["module_path"]}` output, visual slice `{metadata["models"]["openvla"]["deep_node"]["visual_token_slice"]}`.
- PI0.5 P2: base-camera PaliGemma-ready projected tokens, `[256,2048]`.
- PI0.5 P-deep: `{metadata["models"]["pi05"]["deep_node"]["module_path"]}` output, visual slice `{metadata["models"]["pi05"]["deep_node"]["visual_token_slice"]}`.

## 4. Probe

Each node is mean-pooled across all 256 visual tokens. A separate `Linear(D, 7, bias=False)` probe is fitted with AdamW against model-specific TRAIN-normalized deployed actions. Action statistics use TRAIN only. Hyperparameters are recorded in `metadata.json`.

## 5. Results

| Model | Node | D | Train MSE | Held-out MSE | Mean baseline MSE |
|---|---|---:|---:|---:|---:|
{table}

Per-action MSE, MAE, Pearson correlation, and R² are stored in each node's `metrics.json`.

## 6. Interpretation

The table measures which representation contains more linearly readable action-predictive information under this fixed protocol. Weak or negative held-out evidence remains a valid scientific result. These measurements do not establish causal action relevance or controllability.

## 7. Phase 3 Inputs

Each node directory contains frozen `probe.pt`, `W.pt`, `P_action.pt`, TRAIN action normalization in `action_stats.npz`, and `metrics.json`. Their provenance and exact representation nodes are recorded in `metadata.json`.
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
