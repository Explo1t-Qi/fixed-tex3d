<div align="center">

<a href="https://arxiv.org/abs/2604.01618"><img src="https://img.shields.io/badge/arXiv-2604.01618-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
<img src="https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python" alt="Python">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">

# 🎯 Tex3D

## Objects as Attack Surfaces via Adversarial 3D Textures for Vision-Language-Action Models

<p align="center">
  <strong>Jiawei Chen*</strong><sup>1,2</sup> &nbsp;·&nbsp;
  <strong>Simin Huang*</strong><sup>1</sup> &nbsp;·&nbsp;
  <strong>Jiawei Du</strong><sup>3</sup> &nbsp;·&nbsp;
  <strong>Shuaihang Chen</strong><sup>2,5</sup> &nbsp;·&nbsp;
  <strong>Yu Tian</strong><sup>4</sup> &nbsp;·&nbsp;
  <strong>Mingjie Wei</strong><sup>2,5</sup> &nbsp;·&nbsp;
  <strong>Chao Yu†</strong><sup>4</sup> &nbsp;·&nbsp;
  <strong>Zhaoxia Yin†</strong><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>East China Normal University &nbsp;·&nbsp;
  <sup>2</sup>Zhongguancun Academy &nbsp;·&nbsp;
  <sup>3</sup>CFAR, A*STAR, Singapore<br>
  <sup>4</sup>Tsinghua University &nbsp;·&nbsp;
  <sup>5</sup>Harbin Institute of Technology
</p>

<p align="center">
  <em>* Equal contribution &nbsp; † Corresponding authors</em>
</p>

<p align="center">
  <a href="https://vla-attack.github.io/tex3d">🌐 Project Page</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2604.01618">📄 Paper</a> &nbsp;|&nbsp;
  <a href="#citation">📖 Citation</a>
</p>

---

</div>

## 📢 What's NEW!

- 🎉 **[2026]** Our paper **Tex3D** has been **accepted to ACM MM 2026**! See you at the conference in Rio de Janeiro, Brazil.🚀

---

## 📌 Abstract

Vision-Language-Action (VLA) models have shown strong performance in robotic manipulation, yet their robustness to physically realizable adversarial attacks remains underexplored. Existing studies reveal vulnerabilities through language perturbations and 2D visual attacks, but these attack surfaces are either less representative of real deployment or limited in physical realism.

**Tex3D** is the **first framework** for end-to-end optimization of adversarial 3D textures directly within the VLA simulation environment. By introducing two core techniques:

- 🔧 **Foreground-Background Decoupling (FBD)** — establishes a differentiable optimization path from VLA objectives back to object textures via dual-renderer alignment
- 🎯 **Trajectory-Aware Adversarial Optimization (TAAO)** — prioritizes behaviorally critical frames via latent dynamics and stabilizes optimization with vertex-based parameterization

Experiments across simulation and real-robot settings achieve task **failure rates of up to 96.7%**, exposing critical vulnerabilities of VLA systems to physically grounded 3D adversarial attacks.

---

## 🆚 Comparison with Prior Attack Paradigms

| Attack Type | Interface | Physical Grounding | View Robustness | Perceptibility |
|:-----------:|:---------:|:-----------------:|:---------------:|:--------------:|
| Language-Based | Language-dependent | ❌ Limited | ❌ | ✅ Low |
| 2D Patch-Based | Visual front-end | ⚠️ Moderate | ❌ View-specific | ❌ High |
| **Tex3D (Ours)** | Object-centric | ✅ Strong | ✅ Multi-view | ✅ Naturalistic |

---

## 🏗️ Framework Overview

```
Tex3D Framework
├── Foreground-Background Decoupling (FBD)
│   ├── Environmental background rendering (MuJoCo)
│   ├── Target foreground rendering (Nvdiffrast)
│   ├── Cross-renderer geometric alignment (MVP transform)
│   ├── Lighting alignment (Ia, Id, ρ)
│   └── Scene compositing
│
└── Trajectory-Aware Adversarial Optimization (TAAO)
    ├── Latent dynamics-guided frame weighting
    │   ├── Visual encoder feature extraction
    │   ├── Latent velocity & acceleration (central differences)
    │   └── Criticality scoring + temperature-scaled softmax
    ├── Vertex-based texture parameterization
    ├── Untargeted attack objective
    ├── Targeted attack objective
    └── EoT for physical-world transfer
```

---

## 📁 Repository Structure

```
tex3d/
├── openvla/                        # OpenVLA attack scripts & Docker
│   ├── docker_openvla/             # Docker environment for OpenVLA
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   ├── build.sh
│   │   ├── requirements.txt
│   │   └── patches/
│   └── experiments/
│       └── robot/libero/
│           ├── attack_openvla.py   # Attack script for OpenVLA
│           └── openvla_utils.py    # OpenVLA utility functions
│
├── openvla-oft/                    # OpenVLA-OFT attack scripts & Docker
│   ├── docker_oft/                 # Docker environment for OpenVLA-OFT
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   ├── build.sh
│   │   ├── requirements.txt
│   │   └── patches/
│   └── experiments/
│       └── robot/libero/
│           ├── attack_oft.py       # Attack script for OpenVLA-OFT
│           └── libero_utils.py
│
├── image/                          # Images for project page / paper
├── scripts/                        # Helper scripts
├── video/                          # Demo videos
│
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── index.html                      # Project page
├── pyproject.toml                  # Python project configuration
└── requirements-min.txt            # Minimal dependencies
```

---

## ⚙️ Environment Setup

**Requirements:** Docker ≥ 20.10, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), CUDA 12.1+ driver.

If your user is not in the `docker` group, ask an administrator to run:
```bash
sudo usermod -aG docker $USER   # then re-login
```

### OpenVLA

#### Step 1 — Build the Image

```bash
git clone https://github.com/vla-attack/tex3d.git
cd tex3d

docker build -f openvla/docker_openvla/Dockerfile -t tex3d-openvla .
```

> **What the image includes:** Python 3.10, CUDA 12.1, PyTorch 2.2.0, MuJoCo 3.4, LIBERO (patched), nvdiffrast, OpenVLA. Headless OSMesa rendering and `PYTHONPATH` are pre-configured.

#### Step 2 — Prepare Data Mounts

| What | Container path |
|------|---------------|
| LIBERO assets (mesh / texture / XML) | `/data/libero-eval` |
| OpenVLA fine-tuned checkpoint | `/data/openvla-ckpt` |

#### Step 3 — Launch Container

```bash
docker run --gpus all --rm -it \
    -v /your/libero-eval:/data/libero-eval \
    -v /your/openvla-ckpt:/data/openvla-ckpt \
    -v $(pwd)/experiments/logs:/workspace/tex3d/experiments/logs \
    tex3d-openvla
```

<details>
<summary><strong>Using a custom LIBERO fork</strong></summary>

```bash
LIBERO_SRC=/path/to/your/LIBERO-fork bash openvla/docker_openvla/build.sh
```

</details>

---

### OpenVLA-OFT

#### Step 1 — Build the Image

```bash
git clone https://github.com/vla-attack/tex3d.git
cd tex3d

docker build -f openvla-oft/docker_oft/Dockerfile -t tex3d-oft .
```

> **What the image includes:** Python 3.10, CUDA 12.1, PyTorch 2.2.0, MuJoCo 3.4, LIBERO (patched), nvdiffrast, OpenVLA, OpenVLA-OFT (action head / proprio projector / diffusion head). Conda env is named `torch` to match the server-side setup.

#### Step 2 — Prepare Data Mounts

| What | Container path |
|------|---------------|
| LIBERO assets (mesh / texture / XML) | `/data/libero-eval` |
| OpenVLA-OFT fine-tuned checkpoint | `/data/oft-ckpt` |

#### Step 3 — Launch Container

```bash
docker run --gpus all --rm -it \
    -v /your/libero-eval:/data/libero-eval \
    -v /your/oft-ckpt:/data/oft-ckpt \
    -v $(pwd)/experiments/logs:/workspace/tex3d/experiments/logs \
    tex3d-oft
```

<details>
<summary><strong>Using a custom openvla-oft or LIBERO fork</strong></summary>

```bash
OPENVLA_OFT_SRC=/path/to/your/openvla-oft-fork bash openvla-oft/docker_oft/build.sh
LIBERO_SRC=/path/to/your/LIBERO-fork bash openvla-oft/docker_oft/build.sh
```

</details>

---

## 🚀 Running — OpenVLA (`attack_openvla.py`)

`experiments/robot/libero/attack_openvla.py` handles both adversarial training and evaluation in a single run. After training finishes, it immediately evaluates on 50 init states (500 total if all tasks are tested).

### Clean Evaluation (no attack)

Test the unmodified model on a single task:

```bash
python experiments/robot/libero/attack_openvla.py \
    --pretrained_checkpoint /data/openvla-ckpt \
    --task_suite_name libero_spatial \
    --task_id 0 \
    --enable_attack False
```

Test all tasks in a suite (500 episodes total):

```bash
python experiments/robot/libero/attack_openvla.py \
    --pretrained_checkpoint /data/openvla-ckpt \
    --task_suite_name libero_spatial \
    --task_id None \
    --enable_attack False
```

### Adversarial Attack + Evaluation

Train an adversarial texture on a single task, then evaluate on 50 states:

```bash
python experiments/robot/libero/attack_openvla.py \
    --pretrained_checkpoint /data/openvla-ckpt \
    --object_name akita_black_bowl \
    --task_suite_name libero_spatial \
    --task_id 0 \
    --attack_iters 5000 \
    --run_id_note my_exp
```

Train and evaluate on all tasks in a suite:

```bash
python experiments/robot/libero/attack_openvla.py \
    --pretrained_checkpoint /data/openvla-ckpt \
    --object_name akita_black_bowl \
    --task_suite_name libero_spatial \
    --task_id None \
    --attack_iters 5000
```

Load a pre-trained adversarial texture and skip training:

```bash
python experiments/robot/libero/attack_openvla.py \
    --pretrained_checkpoint /data/openvla-ckpt \
    --object_name akita_black_bowl \
    --task_suite_name libero_spatial \
    --task_id 0 \
    --load_texture_path /path/to/Ep0_Vertex_Noise.pt
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pretrained_checkpoint` | Path to OpenVLA fine-tuned checkpoint | — |
| `--object_name` | Target object to attack | `akita_black_bowl` |
| `--task_suite_name` | LIBERO suite: `libero_spatial`, `libero_object` | `libero_spatial` |
| `--task_id` | Task index within the suite, or `None` for all tasks | `0` |
| `--enable_attack` | `True` to train adversarial texture; `False` for clean eval | `True` |
| `--attack_iters` | Optimization iterations | `5000` |
| `--attack_lr` | SignSGD step size | `0.05` |
| `--num_trials_per_task` | Evaluation episodes per task | `50` |
| `--load_texture_path` | Load a saved `.pt` noise file and skip training | `None` |
| `--local_log_dir` | Output directory for logs and artifacts | `./experiments/logs` |
| `--run_id_note` | Prefix appended to the run ID for easy identification | `None` |
| `--live_test_enabled` | Run a full rollout every N iters during training | `False` |
| `--live_test_every_n_iters` | Interval between live tests | `20` |

---

---

## 🚀 Running — OpenVLA-OFT (`attack_oft.py`)

`openvla-oft/experiments/robot/libero/attack_oft.py` supports L1-regression and diffusion action heads, plus proprioception input.

### Clean Evaluation (no attack)

```bash
python openvla-oft/experiments/robot/libero/attack_oft.py \
    --pretrained_checkpoint /data/oft-ckpt \
    --task_suite_name libero_spatial \
    --task_id 0 \
    --enable_attack False
```

### Adversarial Attack + Evaluation

```bash
python openvla-oft/experiments/robot/libero/attack_oft.py \
    --pretrained_checkpoint /data/oft-ckpt \
    --object_name akita_black_bowl \
    --task_suite_name libero_spatial \
    --task_id 0 \
    --attack_iters 5000
```

Load a pre-trained adversarial texture and skip training:

```bash
python openvla-oft/experiments/robot/libero/attack_oft.py \
    --pretrained_checkpoint /data/oft-ckpt \
    --object_name akita_black_bowl \
    --task_suite_name libero_spatial \
    --task_id 0 \
    --load_texture_path /path/to/Ep0_Vertex_Noise.pt
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pretrained_checkpoint` | Path to OpenVLA-OFT fine-tuned checkpoint | — |
| `--object_name` | Target object to attack | `akita_black_bowl` |
| `--task_suite_name` | LIBERO suite: `libero_spatial`, `libero_object` | `libero_spatial` |
| `--task_id` | Task index, or `None` for all tasks | `0` |
| `--enable_attack` | `True` to train adversarial texture; `False` for clean eval | `True` |
| `--attack_iters` | Optimization iterations | `5000` |
| `--use_l1_regression` | Use L1 regression action head | `True` |
| `--use_diffusion` | Use diffusion action head | `False` |
| `--use_proprio` | Feed robot proprioception (EEF pos/ori + gripper) to the model | `True` |
| `--load_texture_path` | Load a saved `.pt` noise file and skip training | `None` |
| `--local_log_dir` | Output directory for logs and artifacts | `./experiments/logs` |

---

## Running - PI0 / PI0.5 (`pi/attack_pi0.py`)

### 1. Environment

The PI0 attack uses the PyTorch PI0.5-LIBERO implementation from [openpi](https://github.com/Physical-Intelligence/openpi). Ubuntu 22.04, Python 3.11, CUDA 12, and an NVIDIA GPU are recommended.

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install -e third_party/libero
uv pip install scipy trimesh draccus omegaconf imageio imageio-ffmpeg
uv pip install "git+https://github.com/NVlabs/nvdiffrast.git"

cp -r src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
```

The checkpoint must be a converted PI0.5-LIBERO PyTorch checkpoint containing `model.safetensors` and `assets/`. Adversarial training additionally requires the VQGAN config and checkpoint used by `taming-transformers`.

Set the LIBERO and object-asset paths at the top of `pi/attack_pi0.py`, then run the following commands from the `openpi` directory. Replace `/path/to/tex3d` with the Tex3D repository path.

### 2. Commands

#### Train the adversarial texture

```bash
uv run /path/to/tex3d/pi/attack_pi0.py \
    --run_mode train \
    --pretrained_checkpoint /path/to/pi05_libero_pytorch \
    --object_name akita_black_bowl \
    --attack_iters 5000 \
    --attack_lr 0.01 \
    --num_frames 5 \
    --latent_encoder_config /path/to/taming-transformers/configs/vqgan_imagenet_f16_16384.yaml \
    --latent_encoder_ckpt /path/to/taming-transformers/checkpoints/vqgan_imagenet_f16_16384.ckpt \
    --local_log_dir /path/to/outputs/pi0_attacks
```

#### Adversarial evaluation

```bash
uv run /path/to/tex3d/pi/attack_pi0.py \
    --run_mode adv_test \
    --pretrained_checkpoint /path/to/pi05_libero_pytorch \
    --object_name akita_black_bowl \
    --load_texture_path /path/to/Ep0_Texture_Noise.pt \
    --eval_max_steps 400 \
    --num_steps_wait 10 \
    --replan_steps 5 \
    --local_log_dir /path/to/outputs/pi0_attacks
```

#### Clean evaluation

```bash
uv run /path/to/tex3d/pi/attack_pi0.py \
    --run_mode clean_test \
    --pretrained_checkpoint /path/to/pi05_libero_pytorch \
    --object_name akita_black_bowl \
    --eval_max_steps 400 \
    --num_steps_wait 10 \
    --replan_steps 5 \
    --local_log_dir /path/to/outputs/pi0_attacks
```

### 3. Policy server

A separate openpi WebSocket policy server is **not required**. The script loads the PyTorch PI0.5 policy directly in the attack process.

---

## 📈 Main Results

Task failure rates (%) on LIBERO benchmark (higher = stronger attack):

| Model | Clean | Gaussian | Tex3D (Untargeted) | Tex3D (Targeted) |
|-------|:-----:|:--------:|:------------------:|:----------------:|
| OpenVLA | 24.1 | 31.1 | **88.1** | **90.5** |
| OpenVLA-OFT | 4.7 | 6.5 | **76.0** | **79.3** |
| π0 | 4.6 | 10.7 | **71.8** | **73.3** |
| π0.5 | 2.8 | 7.4 | **69.3** | **71.2** |

Peak performance: **96.7%** failure rate on OpenVLA Spatial (targeted attack).

---

## 🧪 Evaluation Protocol

Each task is executed for **50 independent trials**. Performance is measured by **Task Failure Rate (FR)** — proportion of failed task completions over all trials.

## 📐 Perturbation Levels

| Level | ε | Additional Constraint | Notes |
|-------|---|----------------------|-------|
| L0 | 64/255 | + MSE loss (naturalness) | Near-imperceptible, stealthy |
| L1 | 16/255 | — | Low budget |
| L2 | 32/255 | — | Medium budget |
| L3 | 64/255 | — | Full budget (default) |

---

## 🔬 Ablation Study Summary

| FBD Components | TAAO Weighting | Failure Rate | Step Time |
|----------------|---------------|:------------:|:---------:|
| w/o MVP alignment | Dynamics | 65.8% | ~7.2s |
| w/o Lighting | Dynamics | 76.8% | ~7.2s |
| w/o Decoupling | Dynamics | 84.6% | ~24.8s |
| Full FBD | Random | 73.7% | ~7.2s |
| Full FBD | Uniform | 82.1% | ~7.2s |
| **Full FBD** | **Dynamics** | **88.1%** | **~7.2s** |

---

## 📖 Citation

If you find Tex3D useful for your research, please cite:

```bibtex
@misc{chen2026tex3dobjectsattacksurfaces,
      title={Tex3D: Objects as Attack Surfaces via Adversarial 3D Textures for Vision-Language-Action Models}, 
      author={Jiawei Chen and Simin Huang and Jiawei Du and Shuaihang Chen and Yu Tian and Mingjie Wei and Chao Yu and Zhaoxia Yin},
      year={2026},
      eprint={2604.01618},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.01618}, 
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We thank the authors of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [OpenVLA](https://github.com/openvla/openvla), [Nvdiffrast](https://github.com/NVlabs/nvdiffrast), and [MuJoCo](https://github.com/google-deepmind/mujoco) for their open-source contributions that made this work possible.

---

<div align="center">
  <sub>⭐ If you find this work useful, please consider starring the repository!</sub>
</div>
