# Step 1B server validation

Status: STEP 1B IMPLEMENTATION READY. Server execution is pending; this document
does not report a scientific result.

Base: `e8d7d5eb97d08cbad50d1f8295651867c43d9be6`.
Branch: `feat/step1b-mature-o2-trajectory`.
The sole specification is [step1b-task.md](step1b-task.md), the deduplicated
user-provided task. The user's latest instruction limits this delivery to
implementation, CPU tests, and commands; the runtime items in specification
section 21 are acceptance checks for the subsequent server run.

## Run directly in the logged-in server shell

The user selected physical GPU **1** for this run, superseding the original
GPU 7 execution setting. The script explicitly sets `CUDA_VISIBLE_DEVICES=1`.

After syncing this revision with the existing local `git push server` workflow,
run the following directly on the server. The final handoff supplies the exact
commit to use for `EXPECTED_HEAD`; the commands below use the checked-out HEAD.

```bash
cd /data/xiaomengqi/src/tex3d-fixed
git switch feat/step1b-mature-o2-trajectory
EXPECTED_HEAD=$(git rev-parse HEAD)
bash scripts/step1b_server_run.sh "$EXPECTED_HEAD" --preflight-only
```

The preflight checks paths, assets, backup files, Git HEAD and worktree, and
whether the run directory already exists. It does not run Python, create the
run directory, create a console log, or start training. Failure output names the
predicate and, for a dirty worktree, prints the affected files.

After `PREFLIGHT PASSED`, start the full pipeline in that server shell:

```bash
bash scripts/step1b_server_run.sh "$EXPECTED_HEAD"
```

Existing run directories remain fail-closed. Each full attempt receives a fresh
`mktemp` console log, so a log from a failed preflight can be preserved without
blocking the next attempt. There is no automatic restart or resume of training.

Local synchronization, when needed, is a separate local-workstation command:

```bash
cd /home/xmq/src/tex3d
git push server HEAD:refs/heads/feat/step1b-mature-o2-trajectory
```

## Full server Bash implementation

The source of this block is `scripts/step1b_server_run.sh`; it defines all
environment variables and uses physical GPU 1. The argument is the expected
committed HEAD.

```bash
#!/usr/bin/env bash
# Server usage: bash scripts/step1b_server_run.sh HEAD [--preflight-only]
set -euo pipefail
trap 'printf "FAILED at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR
EXPECTED_HEAD="${1:?Pass the local committed HEAD as the first argument}"
MODE="${2:-run}"
case "$MODE" in run|--preflight-only) ;; *) printf 'Invalid mode: %s\n' "$MODE" >&2; exit 2 ;; esac
export REPO=/data/xiaomengqi/src/tex3d-fixed
export LOG_ROOT=/data/xiaomengqi/logs/step1b
export OPENVLA_PY=/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python
export OPENVLA_CKPT=/data/huangsimin/openvla-7b-finetuned-libero-spatial
export LIBERO_ROOT_PATH=/data/xiaomengqi/src/LIBERO-joint
export JOINT_PY=/data/xiaomengqi/src/shared-feature-tex3d/.venv-joint/bin/python
export OPENPI_ROOT=/data/xiaomengqi/src/openpi
export SHARED_ROOT=/data/xiaomengqi/src/shared-feature-tex3d
export PI05_CKPT=/data/xiaomengqi/checkpoints/pi05_libero_pytorch
export RUN_ID=step1b-mature-o2-5000-v1
export RUN_ROOT="$LOG_ROOT/$RUN_ID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export LIBERO_ROOT="$LIBERO_ROOT_PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset PYTHONPATH

cd "$REPO"
changes=$(git status --porcelain)
if [[ -n "$changes" ]]; then
    printf 'AUDIT FAILED: dirty worktree before branch switch:\n%s\n' "$changes" >&2
    exit 1
fi
branch_head=$(git rev-parse refs/heads/feat/step1b-mature-o2-trajectory)
if [[ "$branch_head" != "$EXPECTED_HEAD" ]]; then
    printf 'AUDIT FAILED: branch HEAD mismatch: expected=%s actual=%s\n' "$EXPECTED_HEAD" "$branch_head" >&2
    exit 1
fi
git switch feat/step1b-mature-o2-trajectory
for item in "$OPENVLA_PY" "$OPENVLA_CKPT" "$LIBERO_ROOT_PATH" "$JOINT_PY" \
            "$OPENPI_ROOT" "$SHARED_ROOT" "$PI05_CKPT/model.safetensors"; do
    if [[ ! -e "$item" ]]; then printf 'MISSING PATH: %s\n' "$item" >&2; exit 1; fi
done

ASSET_DIR="$LIBERO_ROOT_PATH/libero/libero/assets/stable_scanned_objects/akita_black_bowl"
git_audit() {
    local actual_head changes
    actual_head=$(git rev-parse HEAD) || return 1
    if [[ "$actual_head" != "$EXPECTED_HEAD" ]]; then
        printf 'AUDIT FAILED: HEAD mismatch: expected=%s actual=%s\n' "$EXPECTED_HEAD" "$actual_head" >&2
        return 1
    fi
    changes=$(git status --porcelain) || return 1
    if [[ -n "$changes" ]]; then
        printf 'AUDIT FAILED: dirty worktree:\n%s\n' "$changes" >&2
        return 1
    fi
    git status --short --branch
}
asset_and_git_audit() {
    printf '%s  %s\n' \
      18c1074cfa09baea739bb75928f9bd2bd80e22ac18655f6a27f005dbf77ccfda "$ASSET_DIR/akita_black_bowl.xml" \
      8a646d98b084b7400dd91beed9c83e10b4a4dca572896b3974356b7937a4bd85 "$ASSET_DIR/texture.png" | sha256sum -c - || return 1
    local backups
    backups=$(find "$LIBERO_ROOT_PATH/libero/libero/assets" -type f \
      \( -name '*clean_backup*' -o -name 'texture_clean_backup_*.png' \)) || {
        printf 'AUDIT FAILED: backup search failed\n' >&2; return 1;
    }
    if [[ -n "$backups" ]]; then
        printf 'AUDIT FAILED: backup files remain:\n%s\n' "$backups" >&2
        return 1
    fi
    git_audit
}
on_exit() {
    local status=$?
    trap - EXIT
    asset_and_git_audit || status=1
    if (( status != 0 )); then
        printf 'STOPPED: keep this run and logs; do not retrain or continue analysis.\n' >&2
    fi
    exit "$status"
}
trap on_exit EXIT
asset_and_git_audit
printf 'PRECHECK: GPU=%s expected_HEAD=%s run_root=%s\n' "$CUDA_VISIBLE_DEVICES" "$EXPECTED_HEAD" "$RUN_ROOT"
if [[ -e "$RUN_ROOT" ]]; then
    printf 'STOP: run directory already exists: %s (no restart)\n' "$RUN_ROOT" >&2
    exit 1
fi
if [[ "$MODE" == --preflight-only ]]; then
    printf 'PREFLIGHT PASSED: training not started; no console log created.\n'
    exit 0
fi
mkdir -p "$LOG_ROOT"
# Fresh log per attempt; retain earlier failed-preflight logs without blocking.
CONSOLE_LOG=$(mktemp "$LOG_ROOT/$RUN_ID.console.XXXXXX.log")
exec > >(tee "$CONSOLE_LOG") 2>&1
printf 'CONSOLE_LOG=%s\n' "$CONSOLE_LOG"
git_audit

CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" -m pytest -q tests/unit
"$OPENVLA_PY" -c 'import torch, transformers, tokenizers; assert torch.__version__ == "2.2.0+cu121"; assert transformers.__version__ == "4.40.1"; assert tokenizers.__version__ == "0.19.1"; assert torch.cuda.is_available(); print("source", torch.__version__, transformers.__version__, tokenizers.__version__, torch.cuda.get_device_name(0))'
"$JOINT_PY" -c 'import torch, transformers, scipy; from transformers.models.siglip import check; assert torch.__version__ == "2.6.0+cu124"; assert transformers.__version__ == "4.53.2"; assert check.check_whether_transformers_replace_is_installed_correctly(); assert torch.cuda.is_available(); print("witness", torch.__version__, transformers.__version__, "scipy", scipy.__version__)'
test "$(git -C "$OPENPI_ROOT" rev-parse HEAD)" = 15a9616a00943ada6c20a0f158e3adb39df2ccac
test "$(git -C "$SHARED_ROOT" rev-parse HEAD)" = fffea7571fcde7922b0d0abc1a56d1e88439c011
printf '%s  %s\n' feeedaf6abe1601f8fb24041e21ae8c022b91141ebf1616678cfd2ea8640a09e \
    "$PI05_CKPT/model.safetensors" | sha256sum -c -

# One initialization and one 5000-update source process. No held-out execution here.
PYTHONPATH="$REPO/openvla" "$OPENVLA_PY" openvla/experiments/robot/libero/attack_openvla.py \
  --pretrained_checkpoint "$OPENVLA_CKPT" \
  --unnorm_key libero_spatial_no_noops \
  --task_suite_name libero_spatial --task_id 0 --object_name akita_black_bowl \
  --attack_objective o2_displacement --attack_iters 5000 --attack_lr 0.05 \
  --num_train_init_states 10 --train_frames_per_state 1 --num_frames_to_attack 20 \
  --photometric_calib_frames 5 --frame_collect_with_policy False --collect_grasp_frames False \
  --num_trials_per_task 0 --live_test_enabled False --save_attack_artifacts True \
  --step1_output_dir "$RUN_ROOT" --step1_formal False --step1b_mature_trajectory True \
  --use_wandb False --seed 7 --run_id_note step1b-mature

# The completion marker requires final equivalence AND successful restoration.
CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" scripts/step1b_analyze_trajectory.py audit --run-root "$RUN_ROOT"
asset_and_git_audit
test ! -e "$RUN_ROOT/trajectory"
for iteration in 10 100 500 1000 2000 5000; do
    printf -v tag 'iter_%06d' "$iteration"
    checkpoint="$RUN_ROOT/training/checkpoints/$tag"
    output="$RUN_ROOT/trajectory/$tag"
    test ! -e "$output"
    "$OPENVLA_PY" scripts/step1_collect_heldout_pairs.py \
      --attack-texture "$checkpoint/attack_texture.png" \
      --output-dir "$output/heldout_pairs" --libero-root "$LIBERO_ROOT_PATH" --state-ids 10-19
    PYTHONPATH="$REPO/openvla" "$OPENVLA_PY" scripts/step1_openvla_analysis.py \
      --pairs-dir "$output/heldout_pairs" --output-dir "$output/openvla" \
      --pretrained-checkpoint "$OPENVLA_CKPT" --libero-root "$LIBERO_ROOT_PATH"
    "$JOINT_PY" scripts/step1_pi05_witness.py \
      --pairs-dir "$output/heldout_pairs" --output-dir "$output/pi05" \
      --openpi-root "$OPENPI_ROOT" --shared-feature-root "$SHARED_ROOT" \
      --checkpoint-dir "$PI05_CKPT"
    CUDA_VISIBLE_DEVICES='' "$JOINT_PY" scripts/step1b_analyze_trajectory.py analyze \
      --run-root "$RUN_ROOT" --iteration "$iteration"
    asset_and_git_audit
done
CUDA_VISIBLE_DEVICES='' "$JOINT_PY" scripts/step1b_analyze_trajectory.py summarize --run-root "$RUN_ROOT"
CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" scripts/step1b_analyze_trajectory.py audit --run-root "$RUN_ROOT"
CUDA_VISIBLE_DEVICES='' "$OPENVLA_PY" -m pytest -q tests/unit
asset_and_git_audit
printf 'STEP 1B SERVER RUN FINISHED: %s\n' "$RUN_ROOT"
```

## Artifact and checkpoint contract

A single run writes `/data/xiaomengqi/logs/step1b/step1b-mature-o2-5000-v1/`.
Console logs are fresh sibling files named
`step1b-mature-o2-5000-v1.console.XXXXXX.log` (the suffix is generated by mktemp).

- `training/checkpoints/checkpoint_manifest.json` declares the fixed
  `[10,100,500,1000,2000,5000]` schedule and its completed prefix.
- `training/checkpoints/iter_NNNNNN/` contains `parameter.pt`,
  `attack_texture.png`, and `metadata.json`.
- Iteration N means **after N parameter updates**. The hook uses `i + 1`;
  `pre_update_total_loss` and `pre_update_o2_displacement` describe the loss
  before that update, not a new forward on the checkpoint.
- Metadata binds parameter/texture file SHA256, parameter Linf, perturbation
  bound, epsilon, seed, learning rate, objective, and training Git commit.
- The hook runs under `torch.no_grad()`; it calls the existing baker and final
  PNG conversion. No model forward, RNG call, optimizer update, calibration,
  MuJoCo asset edit, or held-out call is part of checkpoint serialization.
- `training_complete=true` is published only after all 5000 update records,
  six checkpoint hashes, final parameter tensor equality, final texture PNG
  SHA256 equality, and asset restoration have passed. Partial trajectories are
  rejected by the post-hoc gate even if some checkpoint files exist.
- Parameter `.pt` files are compared by tensor content, shape, and dtype for
  final equivalence. File hashes identify each serialized file separately.
- `trajectory/iter_NNNNNN/{heldout_pairs,openvla,pi05,analysis}/` holds post-hoc
  output. It is outside `training/checkpoints/`.
- `trajectory/trajectory_summary.csv` and `.json` contain all six checkpoints'
  displacement means/medians and state/token Spearman summaries.

## Historical behavior and analysis reuse

The original `_validate_step1_formal_config` still requires 10 updates.
Step 1B has a separate gate requiring 5000 and rejecting simultaneous
`step1_formal=True`. The renderer, preprocessing, O2/P2 helpers, collector,
OpenVLA consumer, PyTorch pi0.5 witness, and original Spearman analyzer have no
changes.

`step1b_analyze_trajectory.py analyze` validates the completed trajectory,
selected checkpoint hashes, fixed split, input hashes, finite feature artifacts,
and authoritative OpenVLA environment before calling the existing Step 1
statistics function. It deliberately does not pass the final-5000 texture as
the provenance of an earlier checkpoint or claim the earlier checkpoint was
a separate historical `step1_formal` run.

The provided order is: sync and clean check; CPU regression and environment
preflight; one source training process; checkpoint/completion audit; all six
paired captures and source/witness/transfer analyses; trajectory summary;
checkpoint, asset and Git audits; final CPU regression.

## Validation evidence and pending work

CPU validation at implementation handoff:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/xmq/.virtualenvs/modified-tex3d/bin/python -m pytest -q tests/unit
73 passed in 11.54s (52 existing + 14 trajectory + 7 server-script cases)
bash -n scripts/step1b_server_run.sh                         PASS
scripts/step1b_analyze_trajectory.py --help                  PASS
git diff --check                                           PASS
protected Step 1 pipeline files vs e8d7d5e                  unchanged
historical formal validator and renderer AST regression    PASS
```

The first server attempt stopped at the initial audit (old script line 62),
before CPU tests or training. Both asset hashes passed. The old audit did not
print which subsequent predicate failed, so the actual server failure remains
unresolved until diagnostic preflight output is available. Tests reproduce the
missing diagnostics for HEAD and worktree mismatches; the updated audit prints
both commit IDs or the dirty file list and retains the same failure gates.
Additional shell tests verify GPU 1, preflight without model/training calls,
preservation of old logs, and refusal to restart an existing run directory.

GPU 1 training, checkpoint baking under nvdiffrast, all post-hoc model runs, and
server restoration are **not executed by Codex** for this delivery. The
synthetic CPU snapshot tests verify unchanged RNG, parameters, gradients,
optimizer state, calibration and training mode; real GPU execution remains a
runtime acceptance check.

No scientific conclusion is made. Token results retain the existing
`token-index displacement trend` label.
