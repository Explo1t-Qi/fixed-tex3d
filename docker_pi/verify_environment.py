"""Fail fast when the PI0 attack runtime differs from the validated venv."""

from importlib import metadata
from pathlib import Path
import sys


EXPECTED = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.53.2",
    "draccus": "0.10.0",
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "mujoco": "3.8.1",
    "robosuite": "1.4.1",
    "gym": "0.26.2",
    "nvdiffrast": "0.4.0",
    "omegaconf": "2.4.0.dev13",
    "trimesh": "5.0.0rc1",
}


def main() -> int:
    failures = []
    actual_python = ".".join(map(str, sys.version_info[:3]))
    if actual_python != "3.11.14":
        failures.append(f"Python: expected 3.11.14, got {actual_python}")

    for package, expected in EXPECTED.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            failures.append(f"{package}: not installed")
            continue
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{status:8} {package:20} {actual}")
        if actual != expected:
            failures.append(f"{package}: expected {expected}, got {actual}")

    required_paths = [
        Path("/data/huangsimin/tex3d/pi/attack_pi.py"),
        Path("/data/huangsimin/LIBERO-pi/libero/libero/assets"),
        Path("/data/huangsimin/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT/model.safetensors"),
        Path("/data/huangsimin/openvla/taming-transformers/checkpoints/vqgan_imagenet_f16_16384.ckpt"),
    ]
    for path in required_paths:
        if not path.exists():
            failures.append(f"missing mounted path: {path}")

    if failures:
        print("\nEnvironment verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nEnvironment matches the validated PI0 attack runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
