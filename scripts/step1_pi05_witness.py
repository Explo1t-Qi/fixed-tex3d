"""Extract held-out pi0.5 P2 witness features after texture freezing."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "openvla" / "experiments" / "robot"
if str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))

CONFIG_NAME = "pi05_libero"
CHECKPOINT_IDENTITY = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CHECKPOINT = Path(
    "/data/xiaomengqi/checkpoints/pi05_libero/openpi-assets/checkpoints/pi05_libero"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the post-training no-gradient pi0.5 P2 witness."
    )
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--shared-feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args(argv)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_source_roots(openpi_root: Path, shared_root: Path) -> None:
    required = (
        openpi_root / "src/openpi",
        openpi_root / "packages/openpi-client/src/openpi_client",
        shared_root / "shared_feature/pi05_intervention.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"pi0.5 source dependencies missing: {missing}")
    for source in (
        shared_root,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    ):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))


def _load_runtime(openpi_root: Path, shared_root: Path, checkpoint: Path) -> Any:
    _validate_source_roots(openpi_root, shared_root)
    import jax
    import torch
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_client import image_tools
    from shared_feature import prepare_pi05_context

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"pi0.5 witness requires JAX GPU, got {jax.default_backend()}")
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"frozen pi0.5 JAX checkpoint is incomplete: {checkpoint}")
    if (checkpoint / "model.safetensors").exists():
        raise RuntimeError("formal witness requires the validated JAX/NNX backend")
    train_config = config.get_config(CONFIG_NAME)
    model_config = train_config.model
    if (
        train_config.name != CONFIG_NAME
        or model_config.pi05 is not True
        or model_config.action_horizon != 10
        or model_config.action_dim != 32
        or model_config.discrete_state_input is not False
        or model_config.max_token_len != 200
    ):
        raise RuntimeError("pi05_libero TrainConfig violates frozen semantics")
    policy = policy_config.create_trained_policy(train_config, checkpoint)
    if getattr(policy, "_is_pytorch_model", None) is not False:
        raise RuntimeError("formal witness did not load the validated JAX/NNX model")
    return SimpleNamespace(
        jax=jax,
        torch=torch,
        image_tools=image_tools,
        policy=policy,
        prepare_pi05_context=prepare_pi05_context,
    )


def _prepare_image(image: np.ndarray, image_tools: Any) -> np.ndarray:
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    resized = image_tools.resize_with_pad(rotated, 224, 224)
    value = image_tools.convert_to_uint8(resized)
    if value.shape != (224, 224, 3) or value.dtype != np.uint8:
        raise RuntimeError("OpenPI client preprocessing returned malformed RGB")
    return value


def _policy_input(
    *,
    base_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    robot_state: np.ndarray,
    prompt: str,
    image_tools: Any,
) -> dict[str, Any]:
    return {
        "observation/image": _prepare_image(base_rgb, image_tools),
        "observation/wrist_image": _prepare_image(wrist_rgb, image_tools),
        "observation/state": robot_state.copy(),
        "prompt": prompt,
    }


def _load_pairs(pairs_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from step1_o2_p2 import load_pair_metadata, verify_pair_identity

    manifest = json.loads((pairs_dir / "manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("held-out pair manifest has no records")
    loaded: list[dict[str, Any]] = []
    for record in records:
        state_id = int(record["state_id"])
        state_dir = pairs_dir / f"state_{state_id}"
        metadata = load_pair_metadata(state_dir / "metadata.json")
        clean = np.asarray(Image.open(state_dir / "clean.png").convert("RGB"))
        adversarial = np.asarray(
            Image.open(state_dir / "adversarial.png").convert("RGB")
        )
        verify_pair_identity(
            metadata,
            sample_id=str(metadata["sample_id"]),
            clean_rgb=clean,
            adv_rgb=adversarial,
        )
        with np.load(state_dir / "fixed_inputs.npz", allow_pickle=False) as archive:
            if set(archive.files) != {
                "wrist_rgb_raw",
                "robot_state",
                "scene_state",
                "camera_state",
            }:
                raise ValueError("fixed pi0.5 input archive has malformed fields")
            wrist = archive["wrist_rgb_raw"].copy()
            robot_state = archive["robot_state"].copy()
        if wrist.shape != (512, 512, 3) or wrist.dtype != np.uint8:
            raise ValueError("fixed wrist RGB must be uint8 [512,512,3]")
        if robot_state.shape != (8,) or not np.all(np.isfinite(robot_state)):
            raise ValueError("fixed pi0.5 robot state must be finite [8]")
        if (
            metadata.get("openvla_attacked_camera_field") != "agentview_image"
            or metadata.get("pi05_corresponding_image_field") != "base_0_rgb"
        ):
            raise ValueError("OpenVLA to pi0.5 camera-field mapping mismatch")
        loaded.append(
            {
                "sample_id": str(metadata["sample_id"]),
                "state_id": state_id,
                "metadata": metadata,
                "clean": clean,
                "adversarial": adversarial,
                "wrist": wrist,
                "robot_state": robot_state,
            }
        )
    loaded.sort(key=lambda item: item["state_id"])
    if len({item["sample_id"] for item in loaded}) != len(loaded):
        raise ValueError("duplicate pair sample_id")
    return manifest, loaded


def _extract_p2(runtime: Any, observation: dict[str, Any]) -> tuple[np.ndarray, Any]:
    noise = np.zeros((1, 10, 32), dtype=np.float32)
    model = runtime.policy._model
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    with runtime.torch.no_grad():
        if runtime.torch.is_grad_enabled():
            raise RuntimeError("torch.no_grad guard is not active in witness path")
        prepared = runtime.prepare_pi05_context(
            policy=runtime.policy,
            observation=observation,
            noise=noise,
        )
    p2 = np.asarray(runtime.jax.device_get(prepared.base_p2[0]), dtype=np.float32)
    if p2.shape != (256, 2048) or not np.all(np.isfinite(p2)):
        raise RuntimeError("pi0.5 P2 is invalid")
    return p2, prepared


def _save_npz(path: Path, *, key: str, value: np.ndarray, records: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        path,
        **{key: value},
        sample_ids=np.asarray([item["sample_id"] for item in records]),
        state_ids=np.asarray([item["state_id"] for item in records], dtype=np.int64),
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    pairs_dir = args.pairs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    openpi_root = args.openpi_root.expanduser().resolve()
    shared_root = args.shared_feature_root.expanduser().resolve()
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    if not pairs_dir.is_dir():
        raise FileNotFoundError(f"held-out pairs not found: {pairs_dir}")
    if output_dir.exists():
        raise FileExistsError(f"pi0.5 output directory must be fresh: {output_dir}")

    manifest, records = _load_pairs(pairs_dir)
    runtime = _load_runtime(openpi_root, shared_root, checkpoint)
    clean_features: list[np.ndarray] = []
    adversarial_features: list[np.ndarray] = []
    for item in records:
        common = {
            "wrist_rgb": item["wrist"],
            "robot_state": item["robot_state"],
            "prompt": item["metadata"]["task_description"],
            "image_tools": runtime.image_tools,
        }
        clean_input = _policy_input(base_rgb=item["clean"], **common)
        adversarial_input = _policy_input(base_rgb=item["adversarial"], **common)
        clean_p2, clean_context = _extract_p2(runtime, clean_input)
        adversarial_p2, adversarial_context = _extract_p2(
            runtime, adversarial_input
        )
        for name in ("left_p2", "right_p2"):
            clean_fixed = np.asarray(
                runtime.jax.device_get(getattr(clean_context, name))
            )
            adversarial_fixed = np.asarray(
                runtime.jax.device_get(getattr(adversarial_context, name))
            )
            if not np.array_equal(clean_fixed, adversarial_fixed):
                raise RuntimeError(
                    f"pi0.5 fixed non-primary image representation changed: {name}"
                )
        clean_features.append(clean_p2)
        adversarial_features.append(adversarial_p2)

    clean = np.stack(clean_features)
    adversarial = np.stack(adversarial_features)
    residual = adversarial - clean
    d_p2 = np.mean(np.square(residual, dtype=np.float64), axis=(1, 2))
    from step1_o2_p2 import token_rms

    rms = token_rms(residual)
    if not all(np.all(np.isfinite(value)) for value in (residual, d_p2, rms)):
        raise RuntimeError("pi0.5 witness produced non-finite metrics")

    output_dir.mkdir(parents=True)
    _save_npz(output_dir / "p2_clean.npz", key="p2_clean", value=clean, records=records)
    _save_npz(output_dir / "p2_adv.npz", key="p2_adv", value=adversarial, records=records)
    _save_npz(
        output_dir / "p2_residuals.npz", key="delta_p2", value=residual, records=records
    )
    _save_npz(
        output_dir / "p2_token_rms.npz",
        key="p2_token_rms",
        value=rms,
        records=records,
    )
    with (output_dir / "p2_state_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=("sample_id", "state_id", "d_P2"))
        writer.writeheader()
        for item, displacement in zip(records, d_p2, strict=True):
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "state_id": item["state_id"],
                    "d_P2": repr(float(displacement)),
                }
            )

    summary = {
        "witness_model": "pi0.5",
        "checkpoint": CHECKPOINT_IDENTITY,
        "checkpoint_path": str(checkpoint),
        "openpi_commit": _git_head(openpi_root),
        "witness_adapter_commit": _git_head(shared_root),
        "pair_texture_sha256": manifest["texture_sha256"],
        "num_pairs": len(records),
        "p2_shape": list(clean.shape),
        "node": "PaliGemma-ready projected base-camera visual representation",
        "backend": "jax_nnx",
        "inference_contract": {
            "model_eval": "JAX/NNX train=False (no eval() state API)",
            "torch_no_grad_guard": True,
            "jax_gradient_transform_used": False,
            "only_replaced_pi05_field": "base_0_rgb",
            "fixed_fields": [
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
                "state",
                "language",
            ],
        },
        "consumer_records": [
            {
                "sample_id": item["sample_id"],
                "state_id": item["state_id"],
                "clean_rgb_sha256": item["metadata"]["clean_rgb_sha256"],
                "adv_rgb_sha256": item["metadata"]["adv_rgb_sha256"],
            }
            for item in records
        ],
        "versions": {
            "python": platform.python_version(),
            "jax": str(runtime.jax.__version__),
            "jaxlib": _package_version("jaxlib"),
            "torch": str(runtime.torch.__version__),
        },
    }
    (output_dir / "consumer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = _run(_parse_args(argv))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
