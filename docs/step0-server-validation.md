# Step 0 server validation

Run every command from `/home/xmq/src/tex3d` on branch
`fix/openvla-baseline-correctness`. These commands do not run Step 1 or a
formal attack.

## 1. CPU regression suite

```bash
cd /home/xmq/src/tex3d
PYTHONDONTWRITEBYTECODE=1 \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  -m pytest -q tests/unit
```

## 2. Real checkpoint processor equivalence

```bash
cd /home/xmq/src/tex3d
mkdir -p /tmp/tex3d-step0-evidence
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
LIBERO_ROOT=/opt/libero \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/openvla" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  scripts/audit_step0_processor.py \
  --checkpoint /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --unnorm-key libero_spatial_no_noops \
  --output /tmp/tex3d-step0-evidence/processor-equivalence.json
```

The command exits with status 2 if the real Spatial task-0/state-0 clean frame
does not produce 7/7 identical action tokens. Status 3 means that repeating the
official-processor inference was itself not token-deterministic. Do not proceed
by adding BPDA; return the JSON for review.

## 3. Renderer, visibility, and compositor audit

```bash
cd /home/xmq/src/tex3d
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
LIBERO_ROOT=/opt/libero \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/openvla" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  scripts/audit_step0_renderer.py \
  --resolution 512 \
  --output /tmp/tex3d-step0-evidence/renderer-alignment.json
```

The audit requires two Akita instances, zero position offset, union IoU and
visible recall at least 0.95, zero-delta Linf at most `1e-6`, and finite nonzero
image and texture gradients.

## 4. One-step training plus one-episode Active Texture rollout

```bash
cd /home/xmq/src/tex3d
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
LIBERO_ROOT=/opt/libero \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 \
TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/openvla" \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint /data/huangsimin/openvla-7b-finetuned-libero-spatial \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial \
  --object_name akita_black_bowl \
  --task_id 0 \
  --num_trials_per_task 1 \
  --enable_attack True \
  --attack_iters 1 \
  --num_train_init_states 1 \
  --train_frames_per_state 1 \
  --num_frames_to_attack 1 \
  --photometric_calib_frames 1 \
  --live_test_enabled False \
  --replay_resolution 512 \
  --save_attack_artifacts True \
  --use_wandb False \
  --local_log_dir /tmp/tex3d-step0-evidence/one-step \
  --run_id_note step0-smoke
```

Return the complete command output plus these files/directories:

- `/tmp/tex3d-step0-evidence/processor-equivalence.json`
- `/tmp/tex3d-step0-evidence/renderer-alignment.json`
- `/tmp/tex3d-step0-evidence/one-step/*.txt`
- `/tmp/tex3d-step0-evidence/one-step/attack_artifacts/*/Ep0_step_metrics.jsonl`
- `/tmp/tex3d-step0-evidence/one-step/attack_artifacts/*/Ep0_frame_contract.json`
- `/tmp/tex3d-step0-evidence/one-step/attack_artifacts/*/Ep0_gradient_log.txt`
- `/tmp/tex3d-step0-evidence/one-step/attack_artifacts/*/Ep0_Vertex_Noise.pt`
- `/tmp/tex3d-step0-evidence/one-step/attack_artifacts/*/Ep0_UV_Map.png`
- the saved rollout MP4 path printed by the command

After the command exits, also run:

```bash
cd /home/xmq/src/tex3d
git status --short
find /opt/libero/libero/libero/assets \
  -type f \( -name '*clean_backup*' -o -name 'texture_clean_backup_*.png' \)
```

`git status --short` must be empty and the backup search must print nothing.
