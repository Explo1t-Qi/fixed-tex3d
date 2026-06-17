# tex3d Docker image — OpenVLA + headless GPU rendering
#
# Base: CUDA 12.1 + cuDNN 8 (matches torch==2.2.0+cu121 / RTX 4090 / A100)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── Headless rendering ─────────────────────────────────────────────────────────
# Containers have no DISPLAY by default → MuJoCo will not attempt X11.
# OSMesa is installed via apt (no symlink workarounds needed).
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
# ─────────────────────────────────────────────────────────────────────────────

# ── Path env vars (read by attack_openvla.py) ─────────────────────────────────
ENV LIBERO_ROOT=/opt/libero
ENV TAMING_ROOT=/opt/taming-transformers
# OPENVLA_CKPT is intentionally not set here — pass it via -e or --env-file
# when running the container (model weights should be mounted as a volume).
# ─────────────────────────────────────────────────────────────────────────────

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git ca-certificates build-essential \
        libosmesa6 libosmesa6-dev \
        libgl1-mesa-dev libgles2-mesa-dev \
        libglfw3 libglfw3-dev \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (Python 3.10)
RUN wget -q \
        https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create the vla_render environment
RUN conda create -n vla_render python=3.10.5 -y && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "vla_render", "/bin/bash", "-c"]

# PyTorch — must be installed before packages that build CUDA extensions
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# nvdiffrast (differentiable rasterizer)
RUN git clone https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast && \
    pip install --no-cache-dir -e /opt/nvdiffrast

# LIBERO benchmark
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/libero && \
    pip install --no-cache-dir -e /opt/libero

# OpenVLA-OFT (provides the openvla model interface used by attack_openvla.py)
RUN git clone https://github.com/moojink/openvla-oft.git /opt/openvla-oft && \
    pip install --no-cache-dir -e /opt/openvla-oft

# taming-transformers (VQGAN latent encoder for TAAO frame weighting)
# Note: only the code + configs are baked in; the checkpoint (~1 GB) must be
# downloaded separately and passed via --latent_encoder_ckpt or TAMING_ROOT.
RUN git clone https://github.com/CompVis/taming-transformers.git /opt/taming-transformers && \
    pip install --no-cache-dir -e /opt/taming-transformers

# Remaining pip dependencies
COPY docker_openvla/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy tex3d source
WORKDIR /workspace
COPY . /workspace/tex3d

COPY docker_openvla/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
