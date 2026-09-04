"""Extract authoritative held-out OpenVLA O2 features for Step 1."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = PROJECT_ROOT / "openvla"
ROBOT_ROOT = OPENVLA_ROOT / "experiments" / "robot"
for source_root in (OPENVLA_ROOT, ROBOT_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


CHECKPOINT_IDENTITY = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial_no_noops"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Step 1 held-out OpenVLA O2 representations."
    )
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, default=Path("/opt/libero"))
    return parser.parse_args(argv)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_runtime(checkpoint: Path) -> Any:
    import torch
    from experiments.robot.openvla_image_transform import (
        ExactForwardSurrogateBackwardOpenVLAImageProcessor,
    )
    from experiments.robot.openvla_policy_view import (
        DEFAULT_DEPLOYMENT_VIEW,
        POLICY_SOURCE_RESOLUTION,
        PolicyViewTransform,
    )
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    from experiments.robot.step1_o2_p2 import extract_openvla_o2
    from experiments.robot.libero.libero_utils import get_libero_image

    config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    set_seed_everywhere(7)
    model = get_model(config)
    processor = get_processor(config)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image_processor = (
        ExactForwardSurrogateBackwardOpenVLAImageProcessor.from_checkpoint(
            model=model, processor=processor
        )
    )
    policy_view = PolicyViewTransform(DEFAULT_DEPLOYMENT_VIEW)

    def extract(raw_rgb: np.ndarray) -> np.ndarray:
        policy_source = get_libero_image(
            {"agentview_image": raw_rgb}, POLICY_SOURCE_RESOLUTION
        )
        image = (
            torch.from_numpy(np.ascontiguousarray(policy_source))
            .to(device=model.device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255.0)
        )
        pixel_values = image_processor(policy_view(image)).to(torch.bfloat16)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                o2 = extract_openvla_o2(model, pixel_values)
        value = o2[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        if value.shape != (256, 4096) or not np.all(np.isfinite(value)):
            raise RuntimeError("authoritative OpenVLA O2 is invalid")
        return value

    return SimpleNamespace(
        torch=torch,
        model=model,
        image_processor=image_processor,
        extract=extract,
    )


def _load_pairs(pairs_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from experiments.robot.step1_o2_p2 import (
        load_pair_metadata,
        verify_pair_identity,
    )

    manifest_path = pairs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("held-out pair manifest has no records")
    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        state_id = int(record["state_id"])
        state_dir = pairs_dir / f"state_{state_id}"
        metadata = load_pair_metadata(state_dir / "metadata.json")
        clean = np.asarray(Image.open(state_dir / "clean.png").convert("RGB"))
        adversarial = np.asarray(
            Image.open(state_dir / "adversarial.png").convert("RGB")
        )
        sample_id = str(metadata["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        verify_pair_identity(
            metadata,
            sample_id=sample_id,
            clean_rgb=clean,
            adv_rgb=adversarial,
        )
        if any(metadata.get(name) is not True for name in (
            "no_policy_action_between_observations",
            "scene_state_identical",
            "camera_state_identical",
            "robot_state_identical",
            "wrist_rgb_identical",
        )):
            raise ValueError(f"pair scene contract failed for {sample_id}")
        loaded.append(
            {
                "sample_id": sample_id,
                "state_id": state_id,
                "metadata": metadata,
                "clean": clean,
                "adversarial": adversarial,
            }
        )
    loaded.sort(key=lambda item: item["state_id"])
    if [item["state_id"] for item in loaded] != sorted(
        int(record["state_id"]) for record in records
    ):
        raise ValueError("pair state identity mismatch")
    return manifest, loaded


def _save_npz(path: Path, *, key: str, value: np.ndarray, records: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        path,
        **{key: value},
        sample_ids=np.asarray([item["sample_id"] for item in records]),
        state_ids=np.asarray([item["state_id"] for item in records], dtype=np.int64),
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    pairs_dir = args.pairs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    libero_root = args.libero_root.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"OpenVLA checkpoint not found: {checkpoint}")
    if not (checkpoint / "dataset_statistics.json").is_file():
        raise FileNotFoundError("OpenVLA dataset_statistics.json is missing")
    if not pairs_dir.is_dir():
        raise FileNotFoundError(f"held-out pairs not found: {pairs_dir}")
    if not libero_root.is_dir():
        raise FileNotFoundError(f"LIBERO root not found: {libero_root}")
    if output_dir.exists():
        raise FileExistsError(f"OpenVLA output directory must be fresh: {output_dir}")
    os.environ["LIBERO_ROOT"] = str(libero_root)
    if str(libero_root) not in sys.path:
        sys.path.append(str(libero_root))

    manifest, records = _load_pairs(pairs_dir)
    runtime = _load_runtime(checkpoint)
    clean = np.stack([runtime.extract(item["clean"]) for item in records])
    adversarial = np.stack(
        [runtime.extract(item["adversarial"]) for item in records]
    )
    residual = adversarial - clean
    d_o2 = np.mean(np.square(residual, dtype=np.float64), axis=(1, 2))
    from experiments.robot.step1_o2_p2 import token_rms

    rms = token_rms(residual)
    arrays = (clean, adversarial, residual, d_o2, rms)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise RuntimeError("OpenVLA analysis produced non-finite metrics")

    output_dir.mkdir(parents=True)
    _save_npz(output_dir / "o2_clean.npz", key="o2_clean", value=clean, records=records)
    _save_npz(output_dir / "o2_adv.npz", key="o2_adv", value=adversarial, records=records)
    _save_npz(
        output_dir / "o2_residuals.npz",
        key="delta_o2",
        value=residual,
        records=records,
    )
    _save_npz(
        output_dir / "o2_token_rms.npz",
        key="o2_token_rms",
        value=rms,
        records=records,
    )
    with (output_dir / "o2_state_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=("sample_id", "state_id", "d_O2"))
        writer.writeheader()
        for item, displacement in zip(records, d_o2, strict=True):
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "state_id": item["state_id"],
                    "d_O2": repr(float(displacement)),
                }
            )

    consumer_records = [
        {
            "sample_id": item["sample_id"],
            "state_id": item["state_id"],
            "clean_rgb_sha256": item["metadata"]["clean_rgb_sha256"],
            "adv_rgb_sha256": item["metadata"]["adv_rgb_sha256"],
        }
        for item in records
    ]
    summary = {
        "source_model": "openvla",
        "checkpoint": CHECKPOINT_IDENTITY,
        "checkpoint_path": str(checkpoint),
        "repository_commit": _git_head(),
        "pair_texture_sha256": manifest["texture_sha256"],
        "num_pairs": len(records),
        "o2_shape": list(clean.shape),
        "node": "multimodal projector output before Llama",
        "consumer_records": consumer_records,
        "versions": {
            "python": platform.python_version(),
            "torch": str(runtime.torch.__version__),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
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
