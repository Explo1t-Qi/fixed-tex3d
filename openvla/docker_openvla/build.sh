#!/bin/bash
# Optional helper for building with custom local forks of LIBERO / nvdiffrast / openvla-oft.
#
# Standard public build (uses git clone inside Dockerfile):
#   docker build -f openvla/docker_openvla/Dockerfile -t tex3d-openvla .
#
# Use this script only if you have local modifications to the dependency repos
# and want to bake them into the image instead of cloning from GitHub.
#
# Usage (from tex3d project root):
#   LIBERO_SRC=/path/to/your/LIBERO-fork bash openvla/docker_openvla/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LIBERO_SRC="${LIBERO_SRC:-}"
NVDIFFRAST_SRC="${NVDIFFRAST_SRC:-}"
OPENVLA_OFT_SRC="${OPENVLA_OFT_SRC:-}"

IMAGE_NAME="${IMAGE_NAME:-tex3d-vla}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

if [ -z "$LIBERO_SRC" ] && [ -z "$NVDIFFRAST_SRC" ] && [ -z "$OPENVLA_OFT_SRC" ]; then
    echo "No local overrides set. Running standard public build..."
    docker build \
        -f "$PROJECT_ROOT/openvla/docker_openvla/Dockerfile" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        "$PROJECT_ROOT"
    exit 0
fi

VENDOR_DIR="$PROJECT_ROOT/vendor"
echo "Staging local repos into $VENDOR_DIR ..."
rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"

# Patch Dockerfile to COPY vendor dirs instead of git clone
PATCHED_DOCKERFILE=$(mktemp)
cp "$PROJECT_ROOT/openvla/docker_openvla/Dockerfile" "$PATCHED_DOCKERFILE"

if [ -n "$LIBERO_SRC" ]; then
    cp -r "$LIBERO_SRC" "$VENDOR_DIR/libero"
    sed -i 's|git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/libero|COPY vendor/libero /opt/libero|' "$PATCHED_DOCKERFILE"
fi
if [ -n "$NVDIFFRAST_SRC" ]; then
    cp -r "$NVDIFFRAST_SRC" "$VENDOR_DIR/nvdiffrast"
    sed -i 's|git clone https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast|COPY vendor/nvdiffrast /opt/nvdiffrast|' "$PATCHED_DOCKERFILE"
fi
if [ -n "$OPENVLA_OFT_SRC" ]; then
    cp -r "$OPENVLA_OFT_SRC" "$VENDOR_DIR/openvla-oft"
    sed -i 's|git clone https://github.com/moojink/openvla-oft.git /opt/openvla-oft|COPY vendor/openvla-oft /opt/openvla-oft|' "$PATCHED_DOCKERFILE"
fi

docker build \
    -f "$PATCHED_DOCKERFILE" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$PROJECT_ROOT"

echo "Cleaning up..."
rm -rf "$VENDOR_DIR" "$PATCHED_DOCKERFILE"

echo "Done. Run with: docker run --gpus all --rm -it ${IMAGE_NAME}:${IMAGE_TAG}"
