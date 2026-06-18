#!/bin/bash
# Activate vla_render conda env and exec the command.
# MUJOCO_GL / PYOPENGL_PLATFORM are already set via ENV in the Dockerfile.
source /opt/conda/etc/profile.d/conda.sh
conda activate vla_render
exec "$@"
