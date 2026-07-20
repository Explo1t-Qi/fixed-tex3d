#!/bin/sh
set -eu

# Override with ATTACK_GPU_ID=<physical GPU index> when needed.
ATTACK_GPU_ID="${ATTACK_GPU_ID:-7}"
ATTACK_EGL_DEVICE_ID="${ATTACK_EGL_DEVICE_ID:-$ATTACK_GPU_ID}"

if [ -n "${ATTACK_PYTHON:-}" ]; then
    PYTHON_BIN="$ATTACK_PYTHON"
elif [ -x /data/huangsimin/RLinf/.venv/bin/python ]; then
    PYTHON_BIN=/data/huangsimin/RLinf/.venv/bin/python
else
    PYTHON_BIN=python
fi

export CUDA_VISIBLE_DEVICES="$ATTACK_GPU_ID"
export MUJOCO_GL=egl
# robosuite 1.4 expects the physical ID listed in CUDA_VISIBLE_DEVICES.
export MUJOCO_EGL_DEVICE_ID="$ATTACK_EGL_DEVICE_ID"
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=/data/huangsimin/.cache/openpi
export MPLCONFIGDIR=/tmp/attack_pi_mpl
export PYTHONPATH="/data/huangsimin/taming-transformers-master:/data/huangsimin/openvla/taming-transformers${PYTHONPATH:+:$PYTHONPATH}"

# Use libraries from the selected interpreter. In Docker this must be the
# image's /opt/venv, never the host RLinf venv mounted at /data/huangsimin.
TORCH_LIB_DIR=$(dirname "$PYTHON_BIN")/../lib/python3.11/site-packages/torch/lib
CUDA_SHIM_DIR=/data/huangsimin/cuda-12.4-toolkit-shim/lib64
if [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDA_SHIM_DIR:$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$CUDA_SHIM_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PYTHON_BIN" \
    /data/huangsimin/tex3d/pi/attack_pi.py "$@"
