#!/usr/bin/env python3
"""Extract action-paired O2/deep or P2/deep representations for Pilot v0.2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla/experiments/robot"
OPENVLA_ROOT = PROJECT_ROOT / "openvla"
UNNORM_KEY = "libero_spatial_no_noops"
PI05_CONFIG = "pi05_libero"
CORRECTED_PI05_MANIFEST_SCHEMA = "phase2_action_representation_manifest_v2"
CORRECTED_PI05_MATERIALIZATION_ID = "pi05_p2_runtime_identity_v2"


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("openvla", "pi05"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--shared-feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--reference-action-manifest",
        type=Path,
        help=(
            "Historical PI0.5 representation manifest used only to verify that "
            "the corrected extraction preserves all deployed action targets."
        ),
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        choices=(1, 200),
        default=200,
        help="Use 1 only for a real-checkpoint smoke; formal extraction requires 200.",
    )
    return parser.parse_args(argv)


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_paths(args: argparse.Namespace) -> None:
    paths = [ROBOT_ROOT / "libero", ROBOT_ROOT, OPENVLA_ROOT, args.shared_feature_root]
    if args.openpi_root is not None:
        paths.extend(
            [args.openpi_root / "packages/openpi-client/src", args.openpi_root / "src"]
        )
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"required source roots are missing: {missing}")
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _target_progress(manifest_path: Path) -> dict[str, float]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for task in manifest["task_results"]:
        for group in task["accepted_groups"]:
            for sample in group["samples"]:
                result[sample["sample_id"]] = float(sample["target_relative_progress"])
    return result


def _noise(seed: int, sample_id: str) -> tuple[np.ndarray, int]:
    digest = hashlib.sha256(f"phase2-action-probe|{seed}|{sample_id}".encode()).digest()
    sample_seed = int.from_bytes(digest[:8], "little") % (2**32)
    value = (
        np.random.default_rng(sample_seed).standard_normal((10, 32)).astype(np.float32)
    )
    return value, sample_seed


def _save_archive(
    path: Path,
    *,
    metadata: dict[str, Any],
    projected: torch.Tensor,
    deep: torch.Tensor,
    action: np.ndarray,
) -> None:
    arrays = {
        "projected": projected[0].detach().float().cpu().numpy(),
        "deep": deep[0].detach().float().cpu().numpy(),
        "action": np.asarray(action, dtype=np.float32),
    }
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError("refusing to save non-finite representation archive")
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )


def _reference_actions(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    manifest_path = path.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if manifest.get("model") != "pi05" or len(records) != 200:
        raise RuntimeError("reference action manifest must contain 200 PI0.5 records")
    order: list[str] = []
    actions: dict[str, np.ndarray] = {}
    for record in records:
        archive_path = (manifest_path.parent / record["archive"]).resolve(strict=True)
        if _sha(archive_path) != record["sha256"]:
            raise RuntimeError(f"reference archive hash mismatch: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            sample_id = str(metadata["sample_id"])
            action = np.asarray(archive["action"], dtype=np.float32)
        if sample_id != record["sample_id"] or action.shape != (7,):
            raise RuntimeError(f"reference action identity mismatch: {archive_path}")
        if sample_id in actions or not np.all(np.isfinite(action)):
            raise RuntimeError("reference actions must be unique and finite")
        order.append(sample_id)
        actions[sample_id] = action
    return order, actions


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    shared = args.shared_feature_root.expanduser().resolve(strict=True)
    manifest_path = args.collection_manifest.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be fresh: {output}")
    if args.model == "pi05" and args.openpi_root is None:
        raise ValueError("--openpi-root is required for PI0.5 extraction")
    if (
        args.model == "pi05"
        and args.max_observations == 200
        and args.reference_action_manifest is None
    ):
        raise ValueError(
            "formal corrected PI0.5 extraction requires --reference-action-manifest"
        )
    if args.openpi_root is not None:
        args.openpi_root = args.openpi_root.expanduser().resolve(strict=True)
    _source_paths(args)

    common_path = shared / "scripts/_full_feature_extraction_common.py"
    spec = importlib.util.spec_from_file_location(
        "phase2_shared_full_feature_common", common_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load collection validator: {common_path}")
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    from shared_feature import PilotObservation
    from phase2_action_representation import (
        PI05_P2_DEFINITION_ID,
        extract_openvla_action_representation,
        extract_pi05_action_representation,
        extract_pi05_official_p2,
        pi05_p2_identity_metrics,
    )

    source = common.load_source_collection(manifest_path)
    progress = _target_progress(manifest_path)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal representation extraction requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output.mkdir(parents=True)
    features_dir = output / "features"
    features_dir.mkdir()

    if args.model == "openvla":
        from PIL import Image
        import shared_feature.openvla_features as ov_features
        from openvla_model_inputs import ensure_trailing_empty_token
        from openvla_utils import get_processor, get_vla
        from robot_utils import invert_gripper_action, normalize_gripper_action

        if device != torch.device("cuda:0"):
            raise ValueError("current OpenVLA loader is fixed to logical cuda:0")
        cfg = SimpleNamespace(
            pretrained_checkpoint=checkpoint, load_in_8bit=False, load_in_4bit=False
        )
        model = get_vla(cfg).eval()
        processor = get_processor(cfg)
        runtime = ov_features._load_preprocessing_runtime()

        def extract(observation: Any, sample_id: str) -> Any:
            image = ov_features._build_policy_image(
                observation.base_rgb_raw, runtime, center_crop=True
            )
            if not isinstance(image, Image.Image):
                raise RuntimeError("OpenVLA policy image is not PIL RGB")
            prompt = ov_features._build_prompt(
                observation.prompt, str(checkpoint), runtime.openvla_v01_system_prompt
            )
            inputs = processor(prompt, image, return_tensors="pt").to(
                device, dtype=torch.bfloat16
            )
            inputs = ensure_trailing_empty_token(inputs)

            def deploy(raw: np.ndarray) -> np.ndarray:
                return invert_gripper_action(
                    normalize_gripper_action(raw, binarize=True)
                )

            return extract_openvla_action_representation(
                model=model,
                model_inputs=inputs,
                unnorm_key=UNNORM_KEY,
                deploy_action=deploy,
            )

        backend = "OpenVLAForActionPrediction"
        model_identity = "openvla/openvla-7b-finetuned-libero-spatial"
        noise_contract = None
    else:
        from phase2_shared_gradient import Pi05BaseImageAdapter
        from phase2_shared_gradient_smoke import _load_pi05
        import shared_feature.pi05_features as pi_features

        pi05 = _load_pi05(args.openpi_root, checkpoint, device)
        runtime = pi_features._load_openpi_runtime()

        def pi05_raw(observation: Any) -> dict[str, Any]:
            return {
                "observation/image": pi_features._preprocess_client_image(
                    observation.base_rgb_raw, runtime
                ),
                "observation/wrist_image": pi_features._preprocess_client_image(
                    observation.wrist_rgb_raw, runtime
                ),
                "observation/state": observation.state.copy(),
                "prompt": observation.prompt,
            }

        def extract(observation: Any, sample_id: str) -> Any:
            raw = pi05_raw(observation)
            noise, _ = _noise(args.seed, sample_id)
            adapter = Pi05BaseImageAdapter(
                policy=pi05.policy,
                observation_type=pi05.observation_type,
                policy_input=raw,
                device=device,
            )
            return extract_pi05_action_representation(
                policy=pi05.policy,
                model=pi05.model,
                raw_observation=raw,
                noise=noise,
                authoritative_p2_provider=lambda: extract_pi05_official_p2(
                    model=pi05.model,
                    observation=adapter.clean_observation,
                ),
            )

        backend = "PI0Pytorch"
        model_identity = PI05_CONFIG
        noise_contract = (
            "sha256-derived per-sample seed; standard_normal [10,32] float32"
        )

    first_observation = PilotObservation.load(
        source.records[0].resolved_observation_path
    )
    first = extract(first_observation, first_observation.sample_id)
    repeated = extract(first_observation, first_observation.sample_id)
    p2_identity = None
    if args.model == "pi05":
        adapter = Pi05BaseImageAdapter(
            policy=pi05.policy,
            observation_type=pi05.observation_type,
            policy_input=pi05_raw(first_observation),
            device=device,
        )
        with torch.inference_mode():
            prepared = pi05.model._preprocess_observation(
                adapter.clean_observation, train=False
            )
            if not isinstance(prepared, tuple) or len(prepared) != 5:
                raise RuntimeError("PI0Pytorch preprocessing contract changed")
            images, image_masks, language, language_masks, _ = prepared
            # torch.compile/CUDA Graph paths may reuse static output buffers.
            # Own each witness before the next model call can overwrite it.
            direct_p2 = extract_pi05_official_p2(
                model=pi05.model,
                observation=adapter.clean_observation,
            )
            prefix, _, _ = pi05.model.embed_prefix(
                images, image_masks, language, language_masks
            )
            prefix_p2 = prefix[:, :256].clone()
        p2_identity = pi05_p2_identity_metrics(
            extractor=first.projected,
            embed_image=direct_p2,
            prefix=prefix_p2,
        )
    expected_layers = 32 if args.model == "openvla" else 18
    if first.deep_identity.total_layers != expected_layers:
        raise RuntimeError(
            f"{args.model} authoritative tower must contain {expected_layers} layers, "
            f"got {first.deep_identity.total_layers}"
        )
    determinism = {
        "sample_id": first_observation.sample_id,
        "projected_max_abs": float(
            (first.projected.float() - repeated.projected.float()).abs().max()
        ),
        "deep_max_abs": float((first.deep.float() - repeated.deep.float()).abs().max()),
        "action_max_abs": float(
            np.max(np.abs(first.deployed_action - repeated.deployed_action))
        ),
    }
    determinism["feature_atol"] = 1e-5
    determinism["action_atol"] = 1e-6
    if (
        determinism["projected_max_abs"] > determinism["feature_atol"]
        or determinism["deep_max_abs"] > determinism["feature_atol"]
        or determinism["action_max_abs"] > determinism["action_atol"]
    ):
        raise RuntimeError(f"repeat-forward determinism check failed: {determinism}")

    reference_order: list[str] | None = None
    reference_actions: dict[str, np.ndarray] | None = None
    if args.reference_action_manifest is not None:
        if args.model != "pi05":
            raise ValueError("reference action comparison is PI0.5-only")
        reference_order, reference_actions = _reference_actions(
            args.reference_action_manifest
        )
        source_order = [record.sample_id for record in source.records]
        if reference_order != source_order:
            raise RuntimeError(
                "reference action manifest sample ordering differs from collection"
            )

    records: list[dict[str, Any]] = []
    node_identity = None
    selected_records = source.records[: args.max_observations]
    action_differences: list[float] = []
    for index, record in enumerate(selected_records):
        observation = PilotObservation.load(record.resolved_observation_path)
        result = first if index == 0 else extract(observation, observation.sample_id)
        if reference_actions is not None:
            difference = float(
                np.max(
                    np.abs(
                        result.deployed_action
                        - reference_actions[observation.sample_id]
                    )
                )
            )
            action_differences.append(difference)
            if difference > 1e-6:
                raise RuntimeError(
                    "corrected extraction changed action target for "
                    f"{observation.sample_id}: {difference}"
                )
        identity = result.deep_identity.__dict__
        if node_identity is None:
            node_identity = identity
        elif identity != node_identity:
            raise RuntimeError(
                "deeper-node architecture identity changed during extraction"
            )
        archive_path = features_dir / f"{observation.sample_id}.npz"
        _, noise_seed = _noise(args.seed, observation.sample_id)
        metadata = {
            "schema_version": (
                "phase2_action_representation_archive_v2"
                if args.model == "pi05"
                else "phase2_action_representation_archive_v1"
            ),
            "sample_id": observation.sample_id,
            "task_id": int(observation.task_id),
            "initial_state_id": observation.initial_state_id,
            "target_progress": progress[observation.sample_id],
            "actual_progress": float(observation.normalized_episode_progress),
            "source_observation_sha256": _sha(record.resolved_observation_path),
            "model": args.model,
            "action_semantics": (
                "current deployed LIBERO 7-D action after gripper binarize/invert"
                if args.model == "openvla"
                else "current deployed PI0.5 action chunk first step, 7-D"
            ),
            "pi05_noise_seed": noise_seed if args.model == "pi05" else None,
            "representation_definition_id": (
                PI05_P2_DEFINITION_ID if args.model == "pi05" else None
            ),
        }
        _save_archive(
            archive_path,
            metadata=metadata,
            projected=result.projected,
            deep=result.deep,
            action=result.deployed_action,
        )
        records.append(
            {
                "sample_id": observation.sample_id,
                "archive": f"features/{archive_path.name}",
                "sha256": _sha(archive_path),
            }
        )

    parameters = (
        model.parameters() if args.model == "openvla" else pi05.model.parameters()
    )
    parameters_with_grad = sum(parameter.grad is not None for parameter in parameters)
    if parameters_with_grad:
        raise RuntimeError(
            "representation extraction populated VLA parameter gradients"
        )

    manifest = {
        "schema_version": (
            CORRECTED_PI05_MANIFEST_SCHEMA
            if args.model == "pi05"
            else "phase2_action_representation_manifest_v1"
        ),
        "materialization_id": (
            CORRECTED_PI05_MATERIALIZATION_ID if args.model == "pi05" else None
        ),
        "status": "COMPLETE" if args.max_observations == 200 else "SMOKE_COMPLETE",
        "run_kind": "formal" if args.max_observations == 200 else "smoke",
        "model": args.model,
        "backend": backend,
        "model_identity": model_identity,
        "checkpoint_path": str(checkpoint),
        "checkpoint_reference": str(args.checkpoint),
        "code_commit": _git_head(PROJECT_ROOT),
        "shared_feature_commit": _git_head(shared),
        "collection_manifest": str(manifest_path),
        "collection_manifest_sha256": _sha(manifest_path),
        "count": len(records),
        "source_collection_count": len(source.records),
        "projected_node": "O2" if args.model == "openvla" else "P2",
        "deep_node": node_identity,
        "representation_nodes": {
            "projected": {
                "name": "O2" if args.model == "openvla" else "P2",
                "shape": [256, 4096 if args.model == "openvla" else 2048],
                "token_selection": "all model-native primary-camera visual tokens",
                "capture_semantics": (
                    "multimodal projector output"
                    if args.model == "openvla"
                    else "current PI0Pytorch base-camera embed_image() output; no additional manual scaling"
                ),
                "definition_id": (
                    PI05_P2_DEFINITION_ID if args.model == "pi05" else None
                ),
            },
            "deep": {
                "name": "O-deep" if args.model == "openvla" else "P-deep",
                "shape": [256, 4096 if args.model == "openvla" else 2048],
                **node_identity,
            },
        },
        "token_count": 256,
        "pooling_for_probe": "arithmetic mean over all 256 visual tokens",
        "saved_dtype": "float32",
        "action_dimension": 7,
        "action_target_semantics": (
            "current decoded OpenVLA action after LIBERO gripper binarization and inversion"
            if args.model == "openvla"
            else "first 7-D deployed action from the current PI0.5 action chunk"
        ),
        "pi05_noise_contract": noise_contract,
        "pi05_p2_identity": p2_identity,
        "action_identity": (
            {
                "reference_manifest": str(
                    args.reference_action_manifest.expanduser().resolve(strict=True)
                ),
                "reference_manifest_sha256": _sha(
                    args.reference_action_manifest.expanduser().resolve(strict=True)
                ),
                "compared_observations": len(action_differences),
                "max_abs_action_difference": max(action_differences, default=0.0),
                "tolerance": 1e-6,
                "pass": bool(action_differences) and max(action_differences) <= 1e-6,
            }
            if reference_actions is not None
            else None
        ),
        "determinism": determinism,
        "probe_isolation": {
            "extraction_context": "torch.inference_mode",
            "model_parameters_with_grad": parameters_with_grad,
            "probe_fitting_process_loads_vla": False,
        },
        "records": records,
    }
    (output / "representation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": manifest["status"],
        "model": args.model,
        "count": len(records),
        "output_dir": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    output_preexisted = output.exists()
    try:
        result = _run(args)
    except Exception as error:
        if not output_preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
