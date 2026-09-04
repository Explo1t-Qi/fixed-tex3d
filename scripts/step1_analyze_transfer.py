"""Validate Step 1 artifacts and compute the frozen cross-model metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Step 1 O2/P2 state and token-index trends."
    )
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--openvla-dir", type=Path, required=True)
    parser.add_argument("--pi05-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args(argv)


def _load_feature(path: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {key, "sample_ids", "state_ids"}
        if set(archive.files) != expected:
            raise ValueError(f"malformed feature archive {path}: {archive.files}")
        return (
            archive[key].copy(),
            archive["sample_ids"].astype(str),
            archive["state_ids"].astype(np.int64),
        )


def _load_consumer(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("consumer_records"), list):
        raise ValueError(f"malformed consumer summary: {path}")
    return value


def _identity_rows(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            record.get("sample_id"),
            int(record.get("state_id")),
            record.get("clean_rgb_sha256"),
            record.get("adv_rgb_sha256"),
        )
        for record in records
    ]


def _validate_pi05_contract(summary: dict[str, Any]) -> None:
    contract = summary.get("inference_contract", {})
    if (
        summary.get("backend") != "pytorch"
        or contract.get("model_eval") is not True
        or contract.get("torch_no_grad_guard") is not True
        or contract.get("model_parameters_require_grad") is not False
        or contract.get("official_embed_prefix_slice_equal") is not True
    ):
        raise ValueError("pi0.5 witness did not satisfy the frozen PyTorch contract")


def _correlation(left: np.ndarray, right: np.ndarray, name: str) -> tuple[float, float]:
    result = spearmanr(left, right)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(rho) or not np.isfinite(p_value):
        raise RuntimeError(f"{name} Spearman metric is non-finite")
    return rho, p_value


def _run(args: argparse.Namespace) -> dict[str, Any]:
    pairs_dir = args.pairs_dir.expanduser().resolve()
    openvla_dir = args.openvla_dir.expanduser().resolve()
    pi05_dir = args.pi05_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"analysis output directory must be fresh: {output_dir}")

    training_config = None
    if args.training_dir is not None:
        training_dir = args.training_dir.expanduser().resolve()
        run_dir = training_dir.parent
        training_config = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        recorded_texture_hash = (
            (training_dir / "texture_sha256.txt").read_text(encoding="utf-8").strip()
        )
        actual_texture_hash = "sha256:" + hashlib.sha256(
            (training_dir / "final_attack_texture.png").read_bytes()
        ).hexdigest()
        if recorded_texture_hash != actual_texture_hash:
            raise ValueError("training texture artifact SHA256 mismatch")
    elif args.formal:
        raise ValueError("formal analysis requires training_dir provenance")

    pair_manifest = json.loads(
        (pairs_dir / "manifest.json").read_text(encoding="utf-8")
    )
    openvla_summary = _load_consumer(openvla_dir / "consumer_summary.json")
    pi05_summary = _load_consumer(pi05_dir / "consumer_summary.json")
    _validate_pi05_contract(pi05_summary)
    pair_rows = _identity_rows(pair_manifest["records"])
    openvla_rows = _identity_rows(openvla_summary["consumer_records"])
    pi05_rows = _identity_rows(pi05_summary["consumer_records"])
    if pair_rows != openvla_rows or pair_rows != pi05_rows:
        raise ValueError("OpenVLA/pi0.5 raw pair sample or SHA256 identity mismatch")
    texture_hashes = {
        pair_manifest.get("texture_sha256"),
        openvla_summary.get("pair_texture_sha256"),
        pi05_summary.get("pair_texture_sha256"),
    }
    if len(texture_hashes) != 1 or None in texture_hashes:
        raise ValueError("frozen texture SHA256 mismatch across consumers")
    if args.training_dir is not None and next(iter(texture_hashes)) != recorded_texture_hash:
        raise ValueError("pair texture does not match frozen training texture")

    o2_clean, sample_ids, state_ids = _load_feature(
        openvla_dir / "o2_clean.npz", "o2_clean"
    )
    o2_adv, sample_ids_adv, state_ids_adv = _load_feature(
        openvla_dir / "o2_adv.npz", "o2_adv"
    )
    delta_o2, sample_ids_delta, state_ids_delta = _load_feature(
        openvla_dir / "o2_residuals.npz", "delta_o2"
    )
    o2_rms, sample_ids_o2_rms, state_ids_o2_rms = _load_feature(
        openvla_dir / "o2_token_rms.npz", "o2_token_rms"
    )
    p2_clean, sample_ids_p2, state_ids_p2 = _load_feature(
        pi05_dir / "p2_clean.npz", "p2_clean"
    )
    p2_adv, sample_ids_p2_adv, state_ids_p2_adv = _load_feature(
        pi05_dir / "p2_adv.npz", "p2_adv"
    )
    delta_p2, sample_ids_p2_delta, state_ids_p2_delta = _load_feature(
        pi05_dir / "p2_residuals.npz", "delta_p2"
    )
    p2_rms, sample_ids_p2_rms, state_ids_p2_rms = _load_feature(
        pi05_dir / "p2_token_rms.npz", "p2_token_rms"
    )
    identity_arrays = (
        (sample_ids_adv, state_ids_adv),
        (sample_ids_delta, state_ids_delta),
        (sample_ids_o2_rms, state_ids_o2_rms),
        (sample_ids_p2, state_ids_p2),
        (sample_ids_p2_adv, state_ids_p2_adv),
        (sample_ids_p2_delta, state_ids_p2_delta),
        (sample_ids_p2_rms, state_ids_p2_rms),
    )
    if any(
        not np.array_equal(ids, sample_ids) or not np.array_equal(states, state_ids)
        for ids, states in identity_arrays
    ):
        raise ValueError("feature archive sample/state ordering mismatch")
    if sample_ids.tolist() != [str(row[0]) for row in pair_rows]:
        raise ValueError("feature archive ordering differs from pair manifest")

    n = len(sample_ids)
    if o2_clean.shape != (n, 256, 4096) or o2_adv.shape != o2_clean.shape:
        raise ValueError("OpenVLA O2 feature shape mismatch")
    if p2_clean.shape != (n, 256, 2048) or p2_adv.shape != p2_clean.shape:
        raise ValueError("pi0.5 P2 feature shape mismatch")
    if delta_o2.shape != o2_clean.shape or delta_p2.shape != p2_clean.shape:
        raise ValueError("saved residual shape mismatch")
    if o2_rms.shape != (n, 256) or p2_rms.shape != (n, 256):
        raise ValueError("saved token RMS shape mismatch")
    np.testing.assert_array_equal(delta_o2, o2_adv - o2_clean)
    np.testing.assert_array_equal(delta_p2, p2_adv - p2_clean)
    expected_o2_rms = np.sqrt(
        np.mean(np.square(delta_o2, dtype=np.float64), axis=-1)
    )
    expected_p2_rms = np.sqrt(
        np.mean(np.square(delta_p2, dtype=np.float64), axis=-1)
    )
    np.testing.assert_allclose(o2_rms, expected_o2_rms, rtol=0, atol=0)
    np.testing.assert_allclose(p2_rms, expected_p2_rms, rtol=0, atol=0)
    if args.formal and (n != 10 or state_ids.tolist() != list(range(10, 20))):
        raise ValueError("formal analysis requires exactly held-out states 10-19")
    if args.formal and (
        training_config.get("formal_configuration_frozen") is not True
        or training_config.get("attack_objective") != "o2_displacement"
        or training_config.get("train_state_ids") != list(range(10))
        or training_config.get("heldout_state_ids") != list(range(10, 20))
    ):
        raise ValueError("formal training configuration provenance mismatch")

    d_o2 = np.mean(np.square(delta_o2, dtype=np.float64), axis=(1, 2))
    d_p2 = np.mean(np.square(delta_p2, dtype=np.float64), axis=(1, 2))
    state_rho, state_p = _correlation(d_o2, d_p2, "state-level")
    token_results = [
        _correlation(o2_rms[index], p2_rms[index], f"state {state_id} token")
        for index, state_id in enumerate(state_ids)
    ]
    token_rhos = np.asarray([item[0] for item in token_results])
    token_ps = np.asarray([item[1] for item in token_results])

    output_dir.mkdir(parents=True)
    with (output_dir / "displacement_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        fieldnames = (
            "sample_id",
            "state_id",
            "clean_rgb_sha256",
            "adv_rgb_sha256",
            "d_O2",
            "d_P2",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(pair_rows):
            writer.writerow(
                {
                    "sample_id": row[0],
                    "state_id": row[1],
                    "clean_rgb_sha256": row[2],
                    "adv_rgb_sha256": row[3],
                    "d_O2": repr(float(d_o2[index])),
                    "d_P2": repr(float(d_p2[index])),
                }
            )
    with (output_dir / "token_spearman.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=("sample_id", "state_id", "rho", "p_value", "N")
        )
        writer.writeheader()
        for index, (rho, p_value) in enumerate(token_results):
            writer.writerow(
                {
                    "sample_id": sample_ids[index],
                    "state_id": int(state_ids[index]),
                    "rho": repr(rho),
                    "p_value": repr(p_value),
                    "N": 256,
                }
            )

    summary = {
        "source_model": "OpenVLA",
        "witness_model": "pi0.5",
        "source_checkpoint": openvla_summary["checkpoint"],
        "witness_checkpoint": pi05_summary["checkpoint"],
        "source_git_commit": openvla_summary["repository_commit"],
        "witness_adapter_commit": pi05_summary["witness_adapter_commit"],
        "texture_sha256": next(iter(texture_hashes)),
        "train_state_ids": (
            training_config["train_state_ids"]
            if training_config is not None
            else list(range(10))
        ),
        "heldout_state_ids": state_ids.tolist(),
        "state_displacement_o2": d_o2.tolist(),
        "state_displacement_p2": d_p2.tolist(),
        "state_spearman_rho": state_rho,
        "state_spearman_p": state_p,
        "num_states": n,
        "token_spearman_per_state": token_rhos.tolist(),
        "token_spearman_p_per_state": token_ps.tolist(),
        "token_spearman_mean": float(np.mean(token_rhos)),
        "token_spearman_median": float(np.median(token_rhos)),
        "token_spearman_min": float(np.min(token_rhos)),
        "token_spearman_max": float(np.max(token_rhos)),
        "o2_shape": list(o2_clean.shape),
        "p2_shape": list(p2_clean.shape),
        "token_spatial_order_verified": False,
        "token_metric_label": "token-index displacement trend",
        "all_o2_displacements_nonzero": bool(np.all(d_o2 > 0)),
        "all_p2_displacements_nonzero": bool(np.all(d_p2 > 0)),
        "scientific_scope": (
            "OpenVLA-only optimized texture representation response in held-out "
            "OpenVLA O2 and pi0.5 P2; no random/matched baseline and no action "
            "relevance claim."
        ),
    }
    (output_dir / "summary.json").write_text(
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
