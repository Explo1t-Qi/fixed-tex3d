<div align="center">

<img src="https://img.shields.io/badge/arXiv-2604.01618-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv">
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
├── experiments/
│   └── robot/                      # Robot experiment scripts
│       ├── __pycache__/
│       ├── bridge/                 # Bridge robot experiments
│       └── libero/                 # LIBERO benchmark experiments
│           ├── __pycache__/
│           ├── attack_oft.py       # Attack script for OpenVLA-OFT
│           ├── attack_openvla.py   # Attack script for OpenVLA
│           ├── attack_pi.py        # Attack script for π0
│           ├── attack_pi05.py      # Attack script for π0.5
│           ├── openvla_utils.py    # OpenVLA utility functions
│           └── robot_utils.py      # Common robot utility functions
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

docker build -f docker_openvla/Dockerfile -t tex3d-openvla .
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
LIBERO_SRC=/path/to/your/LIBERO-fork bash docker_openvla/build.sh
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
| `--live_test_enabled` | Run a full rollout every N iters during training | `True` |
| `--live_test_every_n_iters` | Interval between live tests | `20` |

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
