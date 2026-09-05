"""Step 1B completion gate and post-hoc use of the unchanged Step 1 analyzer.

No model is loaded here. `audit` must pass before starting pair collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openvla/experiments/robot"))
sys.path.insert(0, str(ROOT))
from step1b_trajectory import (  # noqa: E402
    CHECKPOINT_SCHEDULE, audit_checkpoints, read_json, write_json,
)
from scripts import step1_analyze_transfer as transfer  # noqa: E402
from scripts import step1_openvla_analysis as source  # noqa: E402
from scripts import step1_pi05_witness as witness  # noqa: E402


def validate_inputs(run_root: Path, iteration: int, manifest: dict) -> Path:
    record = next(r for r in manifest["checkpoints"] if r["iteration"] == iteration)
    directory = run_root / "trajectory" / f"iter_{iteration:06d}"
    pairs, loaded = source._load_pairs(directory / "heldout_pairs")
    _, witness_loaded = witness._load_pairs(directory / "heldout_pairs")
    expected = list(range(10, 20))
    if (pairs["heldout_state_ids"] != expected or pairs["num_pairs"] != 10
            or [r["state_id"] for r in loaded] != expected
            or [r["state_id"] for r in witness_loaded] != expected
            or pairs["texture_sha256"] != record["texture_sha256"]
            or pairs.get("xml_restored") is not True
            or pairs.get("clean_texture_unchanged") is not True):
        raise ValueError("Step 1B checkpoint pair provenance mismatch")
    for item, advertised in zip(loaded, pairs["records"], strict=True):
        metadata = item["metadata"]
        if (metadata["texture_sha256"] != record["texture_sha256"]
                or advertised != {**metadata,
                                  "metadata_path": f"state_{item['state_id']}/metadata.json"}):
            raise ValueError("pair manifest/metadata mismatch")
    o = read_json(directory / "openvla/consumer_summary.json")
    p = read_json(directory / "pi05/consumer_summary.json")
    transfer._validate_pi05_contract(p)
    if (o["pair_texture_sha256"] != record["texture_sha256"]
            or p["pair_texture_sha256"] != record["texture_sha256"]
            or o["versions"]["torch"] != "2.2.0+cu121"
            or o["versions"]["transformers"] != "4.40.1"
            or o["versions"]["tokenizers"] != "0.19.1"
            or o["repository_commit"] != manifest["source_git_commit"]):
        raise ValueError("authoritative consumer provenance mismatch")
    if not (transfer._identity_rows(o["consumer_records"])
            == transfer._identity_rows(p["consumer_records"])
            == transfer._identity_rows(pairs["records"])):
        raise ValueError("consumer sample/hash identity mismatch")
    # The original analyzer validates algebra/shapes, but not every feature's
    # finiteness explicitly. Fail closed before invoking its unchanged metrics.
    for subdir, prefix in (("openvla", "o2"), ("pi05", "p2")):
        for suffix in ("clean", "adv", "residuals", "token_rms"):
            with np.load(directory / subdir / f"{prefix}_{suffix}.npz",
                         allow_pickle=False) as archive:
                for key in archive.files:
                    value = archive[key]
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                        raise ValueError("non-finite feature artifact")
    return directory


def analyze_checkpoint(run_root: Path, iteration: int) -> dict:
    manifest = audit_checkpoints(run_root)
    directory = validate_inputs(run_root, iteration, manifest)
    # Step 1B's own gate above binds the full 5000-update run to the selected
    # checkpoint. Do not pass the final texture as this checkpoint's provenance,
    # or pretend that a checkpoint was a separate historical Step 1 formal run.
    summary = transfer._run(SimpleNamespace(
        pairs_dir=directory / "heldout_pairs", openvla_dir=directory / "openvla",
        pi05_dir=directory / "pi05", output_dir=directory / "analysis",
        training_dir=None, formal=False,
    ))
    summary.update({
        "experiment_mode": "step1b_mature_trajectory",
        "checkpoint_iteration": iteration,
        "training_total_iterations": manifest["total_iterations"],
        "training_source_git_commit": manifest["source_git_commit"],
        "all_checkpoints_frozen_before_analysis": True,
    })
    write_json(directory / "analysis/summary.json", summary)
    return summary


def summarize(run_root: Path) -> dict:
    manifest = audit_checkpoints(run_root)
    output = run_root / "trajectory"
    if any((output / f"trajectory_summary.{ext}").exists() for ext in ("csv", "json")):
        raise FileExistsError("trajectory summary must be fresh")
    rows = []
    for iteration in CHECKPOINT_SCHEDULE:
        directory = validate_inputs(run_root, iteration, manifest)
        summary = read_json(directory / "analysis/summary.json")
        record = next(r for r in manifest["checkpoints"] if r["iteration"] == iteration)
        if (summary.get("checkpoint_iteration") != iteration
                or summary.get("training_total_iterations") != 5000
                or summary.get("all_checkpoints_frozen_before_analysis") is not True
                or summary.get("training_source_git_commit") != manifest["source_git_commit"]
                or summary["texture_sha256"] != record["texture_sha256"]
                or summary["num_states"] != 10
                or summary["heldout_state_ids"] != list(range(10, 20))):
            raise ValueError("checkpoint analysis identity mismatch")
        for filename in ("displacement_summary.csv", "token_spearman.csv"):
            if not (directory / "analysis" / filename).is_file():
                raise FileNotFoundError(filename)
        row = {"iteration": iteration}
        for model in ("o2", "p2"):
            values = summary[f"state_displacement_{model}"]
            if len(values) != 10 or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid displacement distribution")
            row[f"mean_d_{model.upper()}"] = statistics.mean(values)
            row[f"median_d_{model.upper()}"] = statistics.median(values)
        for key in ("state_spearman_rho", "state_spearman_p", "token_spearman_mean",
                    "token_spearman_median", "token_spearman_min", "token_spearman_max"):
            if not math.isfinite(summary[key]):
                raise ValueError("non-finite trajectory metric")
            row[key] = summary[key]
        row["texture_sha256"] = record["texture_sha256"]
        rows.append(row)
    with (output / "trajectory_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"schema_version": "step1b_trajectory_summary_v1", "checkpoints": rows,
              "checkpoint_schedule": list(CHECKPOINT_SCHEDULE),
              "source_git_commit": manifest["source_git_commit"],
              "token_metric_label": "token-index displacement trend"}
    write_json(output / "trajectory_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "analyze", "summarize"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, choices=CHECKPOINT_SCHEDULE)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.command == "analyze":
        if args.iteration is None:
            parser.error("analyze requires --iteration")
        result = analyze_checkpoint(root, args.iteration)
    elif args.command == "summarize":
        result = summarize(root)
    else:
        result = audit_checkpoints(root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
