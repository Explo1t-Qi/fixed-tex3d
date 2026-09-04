"""Extract held-out pi0.5 P2 witness features after texture freezing."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    "/data/xiaomengqi/checkpoints/pi05_libero_pytorch"
)
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _tree_map(function: Any, tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_map(function, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(function, value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_map(function, value) for value in tree)
    return function(tree)


def _validate_source_roots(openpi_root: Path, shared_root: Path) -> None:
    required = (
        openpi_root / "src/openpi",
        openpi_root / "packages/openpi-client/src/openpi_client",
        shared_root / "shared_feature/pi05_features.py",
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
    import torch
    from openpi.models import model as openpi_model
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_client import image_tools

    if not torch.cuda.is_available():
        raise RuntimeError("pi0.5 witness requires a PyTorch CUDA device")
    if not (checkpoint / "model.safetensors").is_file() or not (
        checkpoint / "assets"
    ).is_dir():
        raise FileNotFoundError(
            f"frozen pi0.5 PyTorch checkpoint is incomplete: {checkpoint}"
        )
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
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        pytorch_device="cuda",
    )
    if getattr(policy, "_is_pytorch_model", None) is not True:
        raise RuntimeError("formal witness did not load the PyTorch pi0.5 model")
    model = policy._model
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("pi0.5 policy model is not a torch.nn.Module")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("pi0.5 PyTorch model was not frozen in eval mode")
    return SimpleNamespace(
        torch=torch,
        image_tools=image_tools,
        observation_type=openpi_model.Observation,
        policy=policy,
        model=model,
        device=torch.device(policy._pytorch_device),
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
    from step1_o2_p2 import load_pair_metadata, sha256_rgb, verify_pair_identity

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
        if any(metadata.get(name) is not True for name in (
            "no_policy_action_between_observations",
            "scene_state_identical",
            "camera_state_identical",
            "robot_state_identical",
            "pi05_wrist_rgb_held_fixed",
        )):
            raise ValueError(f"pair scene contract failed for {metadata['sample_id']}")
        if metadata.get("pi05_wrist_rgb_source") != "clean_capture":
            raise ValueError("pi0.5 fixed wrist must come from the clean capture")
        if metadata.get("pi05_fixed_wrist_rgb_sha256") != sha256_rgb(wrist):
            raise ValueError("pi0.5 fixed wrist RGB identity mismatch")
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


def _extract_p2(
    runtime: Any,
    observation: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model = runtime.model
    model.eval()
    with runtime.torch.no_grad():
        if runtime.torch.is_grad_enabled():
            raise RuntimeError("torch.no_grad guard is not active in witness path")
        inputs = _tree_map(lambda value: value, observation)
        inputs = runtime.policy._input_transform(inputs)
        inputs = _tree_map(
            lambda value: runtime.torch.from_numpy(np.asarray(value).copy())
            .to(runtime.device)[None, ...],
            inputs,
        )
        model_observation = runtime.observation_type.from_dict(inputs)
        prepared = model._preprocess_observation(model_observation, train=False)
        if tuple(prepared.images) != IMAGE_KEYS:
            raise RuntimeError(
                f"pi0.5 image-key ordering mismatch: {tuple(prepared.images)}"
            )
        images = [prepared.images[key] for key in IMAGE_KEYS]
        image_masks = [prepared.image_masks[key] for key in IMAGE_KEYS]
        prefix, _, _ = model.embed_prefix(
            images,
            image_masks,
            prepared.tokenized_prompt,
            prepared.tokenized_prompt_mask,
        )
        direct = {
            key: model.paligemma_with_expert.embed_image(prepared.images[key])
            for key in IMAGE_KEYS
        }
        for index, key in enumerate(IMAGE_KEYS):
            value = direct[key]
            expected_shape = (1, 256, 2048)
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"pi0.5 {key} P2 must have shape {expected_shape}, "
                    f"got {tuple(value.shape)}"
                )
            prefix_slice = prefix[:, index * 256 : (index + 1) * 256]
            if not runtime.torch.equal(value, prefix_slice):
                raise RuntimeError(
                    f"pi0.5 {key} P2 does not match official embed_prefix slice"
                )
            if value.requires_grad or not bool(runtime.torch.isfinite(value).all()):
                raise RuntimeError(f"pi0.5 {key} P2 is not finite and detached")

    arrays = {
        key: value[0].detach().to(device="cpu", dtype=runtime.torch.float32).numpy()
        for key, value in direct.items()
    }
    p2 = arrays["base_0_rgb"]
    if p2.shape != (256, 2048) or not np.all(np.isfinite(p2)):
        raise RuntimeError("pi0.5 P2 is invalid")
    return p2, arrays


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
        clean_p2, clean_image_p2 = _extract_p2(runtime, clean_input)
        adversarial_p2, adversarial_image_p2 = _extract_p2(
            runtime, adversarial_input
        )
        for name in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
            clean_fixed = clean_image_p2[name]
            adversarial_fixed = adversarial_image_p2[name]
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
        "backend": "pytorch",
        "model_class": (
            f"{type(runtime.model).__module__}.{type(runtime.model).__qualname__}"
        ),
        "model_safetensors_sha256": _sha256_file(
            checkpoint / "model.safetensors"
        ),
        "inference_contract": {
            "model_eval": runtime.model.training is False,
            "torch_no_grad_guard": True,
            "jax_gradient_transform_used": False,
            "model_parameters_require_grad": any(
                parameter.requires_grad for parameter in runtime.model.parameters()
            ),
            "p2_extractor": "PI0Pytorch.paligemma_with_expert.embed_image",
            "official_embed_prefix_slice_equal": True,
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
            "torch": str(runtime.torch.__version__),
            "transformers": _package_version("transformers"),
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
