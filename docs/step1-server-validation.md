# Step 1 server validation

Run the gates in this order. Stop immediately after any nonzero exit or failed
predicate. In particular, do not run or rerun formal training after inspecting
held-out outputs.

## Runtime paths

```bash
REPO=/data/xiaomengqi/src/tex3d-fixed
LIBERO_ROOT_PATH=/data/xiaomengqi/src/LIBERO-joint
LOG_ROOT=/data/xiaomengqi/logs/step1
OPENVLA_PY=/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python
OPENVLA_CKPT=/data/huangsimin/openvla-7b-finetuned-libero-spatial
OPENPI_ROOT=/data/xiaomengqi/src/openpi
JOINT_PY=/data/xiaomengqi/src/shared-feature-tex3d/.venv-joint/bin/python
PI05_CKPT=/data/xiaomengqi/checkpoints/pi05_libero_pytorch
SHARED_ROOT=/data/xiaomengqi/src/shared-feature-tex3d
```

Before running, verify that `JOINT_PY`, `OPENPI_ROOT`, and `SHARED_ROOT` match
the existing validated joint environment. Do not install or upgrade packages
inside the authoritative OpenVLA environment.

Run this preflight first and return its complete output:

```bash
set -eu
for path in \
  "$REPO" \
  "$LIBERO_ROOT_PATH" \
  "$OPENVLA_PY" \
  "$OPENVLA_CKPT" \
  "$OPENPI_ROOT" \
  "$JOINT_PY" \
  "$PI05_CKPT" \
  "$SHARED_ROOT"
do
  test -e "$path"
  printf 'FOUND %s\n' "$path"
done
mkdir -p "$LOG_ROOT"
cd "$REPO"
git status --short --branch
git rev-parse HEAD
"$OPENVLA_PY" - <<'PY'
import torch, tokenizers, transformers
print("openvla torch", torch.__version__)
print("openvla transformers", transformers.__version__)
print("openvla tokenizers", tokenizers.__version__)
print("openvla cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
"$JOINT_PY" - <<'PY'
import torch, transformers
from transformers.models.siglip import check
print("joint torch", torch.__version__)
print("joint transformers", transformers.__version__)
print("joint cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("openpi transformers patch", check.check_whether_transformers_replace_is_installed_correctly())
PY
git -C "$OPENPI_ROOT" rev-parse HEAD
git -C "$SHARED_ROOT" rev-parse HEAD
```

## 1. CPU regression

```bash
cd "$REPO"
git status --short
git rev-parse HEAD
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$OPENVLA_PY" -m pytest -q tests/unit
```

The required result is an empty initial status and all tests passing.

## 2. Source GPU smoke

This is state 0, one training frame, and one O2-only update. It performs no
rollout and cannot load pi0.5.

```bash
cd "$REPO"
RUN_TAG=$(date +%Y%m%d-%H%M%S)
SOURCE_SMOKE="$LOG_ROOT/source-smoke-$RUN_TAG"
test ! -e "$SOURCE_SMOKE"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
LIBERO_ROOT="$LIBERO_ROOT_PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/openvla" \
"$OPENVLA_PY" openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint "$OPENVLA_CKPT" \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --object_name akita_black_bowl \
  --attack_objective o2_displacement \
  --attack_iters 1 \
  --num_train_init_states 1 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 1 \
  --photometric_calib_frames 1 \
  --num_trials_per_task 0 \
  --live_test_enabled False \
  --save_attack_artifacts True \
  --step1_output_dir "$SOURCE_SMOKE" \
  --use_wandb False \
  --run_id_note step1-source-smoke
```

Validate the emitted evidence:

```bash
"$OPENVLA_PY" - "$SOURCE_SMOKE" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch

root = Path(sys.argv[1])
training = root / "training"
summary = json.loads((training / "training_summary.json").read_text())
restoration = json.loads((root / "asset_restoration.json").read_text())
metrics = [json.loads(line) for line in (training / "Ep0_step_metrics.jsonl").read_text().splitlines()]
assert len(metrics) == 1
metric = metrics[0]
assert summary["attack_objective"] == "o2_displacement"
assert summary["o2_clean_finite"] and summary["o2_adversarial_finite"]
assert summary["pi05_loaded_during_training"] is False
assert math.isfinite(metric["total_loss"])
assert math.isfinite(metric["o2_displacement"]) and metric["o2_displacement"] > 0
assert math.isfinite(metric["texture_gradient_norm"]) and metric["texture_gradient_norm"] > 0
assert metric["parameter_change_linf"] > 0
assert metric["maximum_texture_perturbation"] <= summary["renderer_epsilon"] + 1e-6
assert restoration["pass"] is True
parameter = torch.load(training / "parameter.pt", map_location="cpu")
assert tuple(parameter.shape) == (21932, 3)
assert bool(torch.isfinite(parameter).all())
texture = training / "final_attack_texture.png"
actual = "sha256:" + hashlib.sha256(texture.read_bytes()).hexdigest()
assert actual == summary["texture_sha256"] == (training / "texture_sha256.txt").read_text().strip()
losses = np.load(training / "loss_history.npy", allow_pickle=False)
assert losses.shape == (1,) and np.all(np.isfinite(losses))
print(json.dumps({"source_smoke": "PASS", "metric": metric, "restoration": restoration}, sort_keys=True))
PY
```

Also require an empty backup search:

```bash
find "$LIBERO_ROOT_PATH/libero/libero/assets" -type f \
  \( -name '*clean_backup*' -o -name 'texture_clean_backup_*.png' \)
```

## 3. One-pair witness smoke

Generate the pair from MuJoCo twice at the same state, without a policy action:

```bash
cd "$REPO"
"$OPENVLA_PY" scripts/step1_collect_heldout_pairs.py \
  --attack-texture "$SOURCE_SMOKE/training/final_attack_texture.png" \
  --output-dir "$SOURCE_SMOKE/heldout_pairs" \
  --libero-root "$LIBERO_ROOT_PATH" \
  --state-ids 10
```

Run the OpenVLA consumer only in the authoritative OpenVLA environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
LIBERO_ROOT="$LIBERO_ROOT_PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/openvla" \
"$OPENVLA_PY" scripts/step1_openvla_analysis.py \
  --pairs-dir "$SOURCE_SMOKE/heldout_pairs" \
  --output-dir "$SOURCE_SMOKE/openvla" \
  --pretrained-checkpoint "$OPENVLA_CKPT" \
  --libero-root "$LIBERO_ROOT_PATH"
```

Only after that process exits, run the separate PyTorch pi0.5 witness process:

```bash
cd "$REPO"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
"$JOINT_PY" scripts/step1_pi05_witness.py \
  --pairs-dir "$SOURCE_SMOKE/heldout_pairs" \
  --output-dir "$SOURCE_SMOKE/pi05" \
  --openpi-root "$OPENPI_ROOT" \
  --shared-feature-root "$SHARED_ROOT" \
  --checkpoint-dir "$PI05_CKPT"
```

For both consumer outputs, require the saved feature/state metrics to be finite,
the shapes to be `[1,256,4096]` and `[1,256,2048]`, and the consumer identity
rows to be exactly equal. The one-pair smoke does not compute or interpret a
Spearman correlation.

## 4. Formal run (exactly once)

Choose a fresh run ID before any held-out extraction:

```bash
cd "$REPO"
RUN_ID=step1-o2-p2-formal-v1
FORMAL_ROOT="$LOG_ROOT/$RUN_ID"
test ! -e "$FORMAL_ROOT"
```

Run the single frozen configuration:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
LIBERO_ROOT="$LIBERO_ROOT_PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/openvla" \
"$OPENVLA_PY" openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint "$OPENVLA_CKPT" \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --object_name akita_black_bowl \
  --attack_objective o2_displacement \
  --attack_iters 10 \
  --attack_lr 0.05 \
  --num_train_init_states 10 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 20 \
  --photometric_calib_frames 5 \
  --num_trials_per_task 0 \
  --live_test_enabled False \
  --save_attack_artifacts True \
  --step1_output_dir "$FORMAL_ROOT" \
  --step1_formal True \
  --use_wandb False \
  --seed 7 \
  --run_id_note step1-formal
```

This configuration intentionally runs zero policy-rollout episodes. Its final
summary must therefore say `Task success rate: NOT EVALUATED (rollout
disabled)`; neither task success nor attack success rate is a Step 1 metric.
Ten optimization iterations are texture-training updates, not ten rollout
episodes.

Do not retrain from this point onward. Collect states 10-19, then run each
consumer exactly as in the witness smoke:

```bash
"$OPENVLA_PY" scripts/step1_collect_heldout_pairs.py \
  --attack-texture "$FORMAL_ROOT/training/final_attack_texture.png" \
  --output-dir "$FORMAL_ROOT/heldout_pairs" \
  --libero-root "$LIBERO_ROOT_PATH" \
  --state-ids 10-19

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
LIBERO_ROOT="$LIBERO_ROOT_PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/openvla" \
"$OPENVLA_PY" scripts/step1_openvla_analysis.py \
  --pairs-dir "$FORMAL_ROOT/heldout_pairs" \
  --output-dir "$FORMAL_ROOT/openvla" \
  --pretrained-checkpoint "$OPENVLA_CKPT" \
  --libero-root "$LIBERO_ROOT_PATH"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
"$JOINT_PY" scripts/step1_pi05_witness.py \
  --pairs-dir "$FORMAL_ROOT/heldout_pairs" \
  --output-dir "$FORMAL_ROOT/pi05" \
  --openpi-root "$OPENPI_ROOT" \
  --shared-feature-root "$SHARED_ROOT" \
  --checkpoint-dir "$PI05_CKPT"

"$JOINT_PY" scripts/step1_analyze_transfer.py \
  --pairs-dir "$FORMAL_ROOT/heldout_pairs" \
  --openvla-dir "$FORMAL_ROOT/openvla" \
  --pi05-dir "$FORMAL_ROOT/pi05" \
  --training-dir "$FORMAL_ROOT/training" \
  --output-dir "$FORMAL_ROOT/analysis" \
  --formal
```

Finally inspect every required artifact, rerun the CPU regression suite, verify
`asset_restoration.json`, repeat the backup search, and record `git status`.
Only then may `docs/step1-o2-p2-transfer-report.md` state `STEP 1 COMPLETE`.
