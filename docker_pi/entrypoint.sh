#!/bin/bash
set -e
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data/.cache/openpi}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
export ATTACK_IN_DOCKER=1
exec "$@"
