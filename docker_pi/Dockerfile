# PI0/LIBERO attack environment aligned with the validated RLinf venv.
# Checkpoints, modified LIBERO assets, VQGAN weights, and outputs are mounted.
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1 \
    OPENPI_DATA_HOME=/data/huangsimin/.cache/openpi \
    MPLCONFIGDIR=/tmp/attack_pi_mpl \
    PYTHONPATH=/data/huangsimin/tex3d:/data/huangsimin/LIBERO-pi:/data/huangsimin/taming-transformers-master:/data/huangsimin/openvla/taming-transformers:/opt/taming-transformers:/opt/libero

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates \
    build-essential ffmpeg libegl1 libgl1 libglib2.0-0 libsm6 libxext6 \
    libxrender1 libglfw3 patchelf ninja-build && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    uv python install 3.11.14 && \
    uv venv --python 3.11.14 /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN uv pip install --python /opt/venv/bin/python \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Use the same RLinf OpenPI fork and commit as the validated host venv.
ARG OPENPI_COMMIT=c5dc4b9296a1a4739bf52828f28a579f12dce763
RUN uv pip install --python /opt/venv/bin/python \
    "git+https://github.com/RLinf/openpi.git@${OPENPI_COMMIT}"

COPY docker_pi/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /opt/venv/bin/python --prerelease=allow \
    -r /tmp/requirements.txt

# OpenPI's PyTorch implementation requires its patched Transformers modules.
RUN cp -r /opt/venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* \
    /opt/venv/lib/python3.11/site-packages/transformers/

ARG LIBERO_COMMIT=0c5e40cc4ae63e09c14e7df6f74481e9ee8585f7
RUN git clone https://github.com/RLinf/LIBERO.git /opt/libero && \
    git -C /opt/libero checkout "${LIBERO_COMMIT}" && \
    uv pip install --python /opt/venv/bin/python --no-deps -e /opt/libero

RUN git clone --branch v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast && \
    uv pip install --python /opt/venv/bin/python --no-build-isolation /opt/nvdiffrast

# VQGAN encoder implementation used by attack_pi.py; weights are mounted.
RUN git clone https://github.com/CompVis/taming-transformers.git /opt/taming-transformers

WORKDIR /data/huangsimin/tex3d
COPY pi /data/huangsimin/tex3d/pi
COPY docker_pi/entrypoint.sh /entrypoint.sh
COPY docker_pi/verify_environment.py /opt/verify_environment.py
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
