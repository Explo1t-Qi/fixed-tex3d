#!/usr/bin/env python3
"""Audit capacity or collect the Pilot v0.3 expanded Phase 2 dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
OPENVLA_ROOT = PROJECT_ROOT / "openvla"
for source in (ROBOT_ROOT, OPENVLA_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from phase2_expanded_dataset import (  # noqa: E402
    CHECKPOINT_IDENTITY,
    GLOBAL_SEED,
    UNNORM_KEY,
    capacity_report,
    collect_expanded_observations,
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--shared-feature-root", type=Path, required=True)
    audit.add_argument("--libero-revision", required=True)
    audit.add_argument("--output-path", type=Path, required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--shared-feature-root", type=Path, required=True)
    collect.add_argument("--pretrained-checkpoint", type=Path, required=True)
    collect.add_argument("--libero-revision", required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _prepare_source(shared_feature_root: Path) -> Path:
    shared = shared_feature_root.expanduser().resolve(strict=True)
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return shared


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    shared = _prepare_source(args.shared_feature_root)
    from shared_feature import libero_collector as collector

    runtime = collector._load_official_runtime()
    suite = runtime.benchmark.get_benchmark_dict()[collector.PILOT_SUITE]()
    audit = capacity_report(suite, libero_revision=args.libero_revision)
    if args.command == "audit":
        output = args.output_path.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"capacity audit output must be fresh: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {**audit, "output_path": str(output)}

    checkpoint = args.pretrained_checkpoint.expanduser().resolve(strict=True)
    import torch
    from openvla_utils import get_processor
    from robot_utils import get_model, set_seed_everywhere

    if not torch.cuda.is_available():
        raise RuntimeError("expanded collection requires CUDA")
    set_seed_everywhere(GLOBAL_SEED)
    config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    result = collect_expanded_observations(
        model=get_model(config),
        processor=get_processor(config),
        pretrained_checkpoint=checkpoint,
        libero_revision=args.libero_revision,
        code_commit=_git_head(PROJECT_ROOT),
        shared_feature_commit=_git_head(shared),
        output_dir=args.output_dir,
    )
    return {
        "status": result.status,
        "split_readiness": result.split_readiness,
        "manifest_path": str(result.manifest_path),
        "feasibility_report": str(
            result.manifest_path.parent / "feasibility_report.json"
        ),
        "observation_count": len(result.sample_paths),
        "checkpoint_identity": CHECKPOINT_IDENTITY,
        "capacity": audit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
