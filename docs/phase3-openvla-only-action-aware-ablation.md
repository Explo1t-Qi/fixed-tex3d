# Exploratory Phase 3: OpenVLA-only action-aware ablation

Status: implementation complete; server 10-step smoke and 500-step optimization pending. This is a fixed-seed-7 source-policy ablation, not authoritative Phase 2B v3 or a transfer experiment. The expanded PI0.5 P2 cross-seed stability gate remains unresolved.

## Question and fixed protocol

Does the action-aware O2 objective cause a stronger OpenVLA policy-level effect than native O2 displacement? The new explicit `openvla_action_predictive` runner loads only OpenVLA and the expanded provisional O2 seed-7 probe. It never loads the PI0.5 policy/checkpoint or uses the P2 probe in forward, loss, or gradient computation. It retains task 0, official states 0–9, one initial frame per state, 512-pixel renderer, the deployment-view/preprocessing path, seed 7, sign-PGD step 0.05, 500 updates, and the existing 128/255 texture budget.

For each frame, `Z = mean_token(O2) @ W_O2.T`, with all seven frozen TRAIN-normalized action coordinates. The loss is `-MSE(Z_adv,Z_clean) + lambda_dir * cosine(Z_adv,Z_clean)`; the clean O2 and Z reference are captured under `no_grad`. Before texture updates, `lambda_dir = median_frames(||g_mag_O||_2 / (||g_dir_O||_2 + 1e-12))` is recomputed from all ten frames. Within each update, raw OpenVLA gradients are averaged over the ten frames and applied directly, without model-level normalization or cross-model averaging.

The runner checks expanded provisional artifact identity, split hash, seed-7 probe protocol, O2 W/statistics hashes, OpenVLA checkpoint path, frozen model/probe, finite gradients, and texture budget. Outputs include `training_config.json`, `frame_contract.json`, `lambda_calibration.json`, per-step `step_metrics.jsonl`, final texture/parameter, checkpoint textures/parameters when requested, `training_summary.json`, `metadata.json`, and SHA-256 inventory. A smoke has `SMOKE_COMPLETE`; a full run has `TRAINING_COMPLETE_PENDING_OPENVLA_ROLLOUT`.

## Server sequence

Use a clean checkout at the commit reported with this implementation. Verify all paths with `test -e` and `realpath`; both output directories must be fresh. The server's CUDA/MuJoCo environment should match the previously successful Phase 3 runs. The new runner requires no OpenPI root, PI0.5 checkpoint, or second GPU.

```bash
export REPO=/data/xiaomengqi/src/tex3d-fixed
export PY=/data/xiaomengqi/src/shared-feature-tex3d/.venv-joint/bin/python
export PHASE2_ROOT=/data/xiaomengqi/logs/shared-feature-phase2
export LIBERO_ROOT=/data/xiaomengqi/src/LIBERO-joint
export OPENVLA_CKPT=/data/huangsimin/openvla-7b-finetuned-libero-spatial
export PROVISIONAL_DIR=$PHASE2_ROOT/phase2-expanded-seed7-provisional-action-probes-v1
export O_ONLY_SMOKE=$PHASE2_ROOT/phase3-openvla-only-action-aware-10step-smoke-v1
export O_ONLY_500=$PHASE2_ROOT/phase3-openvla-only-action-aware-500step-v1

for p in "$REPO" "$PY" "$LIBERO_ROOT" "$OPENVLA_CKPT" "$PROVISIONAL_DIR/metadata.json" "$PROVISIONAL_DIR/artifact_inventory.json" "$PROVISIONAL_DIR/openvla/o2/W.pt"; do test -e "$p" && realpath "$p" || exit 1; done
git -C "$REPO" status --short
git -C "$REPO" rev-parse HEAD
test ! -e "$O_ONLY_SMOKE"
```

Run the 10-step engineering smoke first:

```bash
"$PY" -u "$REPO/scripts/phase3_openvla_action_optimization.py" \
  --probe-artifact-dir "$PROVISIONAL_DIR" \
  --openvla-checkpoint "$OPENVLA_CKPT" \
  --libero-root "$LIBERO_ROOT" \
  --openvla-device cuda:0 \
  --iterations 10 --pgd-step 0.05 --smoke \
  --output-dir "$O_ONLY_SMOKE"
```

Stop for review unless the smoke's `metadata.json` says `SMOKE_COMPLETE`, lambda calibration says `PASS`, all ten steps are finite and nonzero-gradient, the probe W is unchanged, and the texture budget is respected. The smoke itself does not establish policy effect.

Only after review, run a fresh 500-step exploratory optimization:

```bash
test ! -e "$O_ONLY_500"
"$PY" -u "$REPO/scripts/phase3_openvla_action_optimization.py" \
  --probe-artifact-dir "$PROVISIONAL_DIR" \
  --openvla-checkpoint "$OPENVLA_CKPT" \
  --libero-root "$LIBERO_ROOT" \
  --openvla-device cuda:0 \
  --iterations 500 --pgd-step 0.05 \
  --checkpoint-steps 50,100,250,500 \
  --output-dir "$O_ONLY_500"
```

This command recalibrates lambda from scratch. A 500-step run stops at `TRAINING_COMPLETE_PENDING_OPENVLA_ROLLOUT`; do not infer policy success from representation losses.

## Baseline and interpretation

The historical O2-only native-displacement code path exists in `attack_openvla.py`: `L_native = -MSE(O2_adv,O2_clean)`. The documented formal Step 1 artifact (`step1-o2-p2-formal-v1`) ran only ten optimization steps with policy rollout disabled. It is not a strict 500-step behavioral baseline. The dual-model native-GE 500-step artifact is also not O2-only.

For a controlled comparison, run the native O2-only path for 500 updates under the same OpenVLA checkpoint, task/states/frame pool, renderer and preprocessing, initialization seed, sign-PGD step, texture budget, and source-policy rollout protocol. The recent dual-model OpenVLA rollout evaluated task-0 states 0–49, with `num_steps_wait=10`, `max_steps=300`, and paired seed `seed + state_id`; use the same 50 state IDs and report states 0–9 (seen during texture training) separately from states 10–49. Compare paired clean→adversarial OpenVLA success/conditional ASR (same trials), then action-coordinate displacement, native O2 MSE, perturbation magnitude, and trajectory-level differences. Archive code commit and texture hashes for both arms. Existing results may be contextual references but should not be promoted to the strict baseline without this protocol match.
