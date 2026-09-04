# Step 1 O2-to-P2 transfer report

## Status

```text
STEP 1 COMPLETE
```

The implementation and CPU contracts are complete. The source GPU smoke and
one-pair OpenVLA O2/PyTorch pi0.5 P2 witness smoke passed on the server. The
single frozen formal training run over states 0-9 and the held-out analysis over
states 10-19 completed. All required artifacts, identities, metrics, runtime
restoration checks, and final regressions passed.

## Git provenance

```text
repository          /home/xmq/src/tex3d
base commit         6ee46c18c78981cfa86b45edaab20c11db030483
branch              feat/step1-o2-p2-transfer-mvp
pre-formal code HEAD dc4ac355e5fc61bbb8182fc0bec5f1e06454fa45
formal training code HEAD 6ec34f1c20e18aa688080de24cc2df2894fee713
held-out consumer code HEAD 6d961c001d2d079483ab071ed4b55b566f6779a8
Step 0 code ancestor ff9fcbc8f84debb7693027a58490a0d856c8cdde
```

The branch was created only after the pre-existing Step 1 `task.md` change was
temporarily stashed, the Step 0 worktree was verified clean, and the task file
was restored on the new branch. `main` was not modified.

Changed files and responsibilities:

- `openvla/experiments/robot/libero/attack_openvla.py`: opt-in O2-only
  objective, formal configuration gate, training artifacts and restoration
  evidence, and correct zero-rollout reporting;
- `openvla/experiments/robot/step1_o2_p2.py`: frozen O2/P2 shape, objective,
  gradient, initialization, token RMS, and identity helpers;
- `scripts/step1_collect_heldout_pairs.py`: paired MuJoCo Active Texture capture
  and scene/hash contracts;
- `scripts/step1_openvla_analysis.py`: authoritative-environment O2 consumer;
- `scripts/step1_pi05_witness.py`: frozen PyTorch pi0.5 P2 consumer;
- `scripts/step1_analyze_transfer.py`: fail-closed artifact validation and
  state/token-index Spearman metrics;
- `tests/unit/test_step1_o2_p2.py`, `tests/unit/test_step1_artifacts.py`, and
  `tests/unit/test_openvla_entrypoint.py`: required objective, gradient, shape,
  no-grad, identity, continuation, legacy, and reporting regressions;
- `docs/step1-server-validation.md`: exact staged server commands;
- `docs/step1-o2-p2-transfer-report.md`: engineering/scientific final report;
- `task.md`: frozen Step 1 specification;
- `.gitignore`: excludes synchronized experiment inbox data from Git.

## Engineering validity

Implemented contracts:

- opt-in `attack_objective=o2_displacement`; default remains `legacy`;
- native OpenVLA O2 extraction at the validated multimodal-projector output,
  with shape `[B,256,4096]`;
- detached clean O2 and differentiable adversarial O2;
- exact `loss = -mean((adv_o2-clean_o2)^2)` without action CE, final-hidden,
  SigLIP, spectral, or pi0.5 terms;
- OpenVLA parameter freezing while retaining image-to-texture gradients;
- an OpenPI import guard in the source training process;
- deterministic nonzero O2-mode initialization, needed because squared
  displacement has an exactly zero derivative at an exactly clean start;
- raw MuJoCo clean/adversarial pair collection with no policy action and exact
  robot/scene/camera equality checks;
- explicit reuse and SHA256 verification of the clean-capture wrist image for
  both pi0.5 inputs; the separately captured adversarial wrist may legitimately
  show the changed texture and is not treated as scene-state drift;
- independent OpenVLA and pi0.5 consumers that recompute and verify the same
  sample IDs and raw-image SHA256 values;
- the validated pi05 adapter mapping `agentview_image -> base_0_rgb`, with wrist,
  state, language, and the padded camera fixed;
- O2/P2 residual, state displacement, token RMS, state Spearman, and per-state
  token-index Spearman artifact generation;
- end-to-end frozen texture SHA256 checks and byte-exact XML/clean-texture
  restoration evidence;
- a strict formal gate for train states 0-9 and held-out states 10-19.

The deterministic O2 initialization uses the existing `attack_lr=0.05` as its
parameter-space bound and seed 7. It does not change the renderer, perturbation
budget, SignSGD update, or objective. This single configuration is recorded in
`config.json`; it is not a sweep. Legacy mode retains zero initialization.

The formal witness uses the converted pi0.5-LIBERO PyTorch checkpoint at
`/data/xiaomengqi/checkpoints/pi05_libero_pytorch/model.safetensors`. It runs
the official OpenPI policy transforms and PyTorch observation preprocessing,
then extracts P2 with `paligemma_with_expert.embed_image`. Every extracted
image representation is required to equal its corresponding token slice from
the official `embed_prefix()` continuation. The model is frozen, set to
`eval()`, and executed inside `torch.no_grad()`.

### CPU regression evidence

Pre-change baseline:

```text
PYTHONDONTWRITEBYTECODE=1 \
/home/xmq/.virtualenvs/modified-tex3d/bin/python -m pytest -q tests/unit
30 passed in 7.75s
```

Latest implementation suite after the zero-episode reporting regression fix:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/xmq/.virtualenvs/modified-tex3d/bin/python -m pytest -q tests/unit
52 passed in 6.09s
```

Final server regression at held-out consumer HEAD
`6d961c001d2d079483ab071ed4b55b566f6779a8`:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/xiaomengqi/miniconda3/envs/tex3d-openvla/bin/python \
  -m pytest -q tests/unit
52 passed in 3.54s
```

The explicit empty `CUDA_VISIBLE_DEVICES` is required for a CPU regression on
this workstation. One intermediate invocation let TensorFlow discover the
local GPU and failed because local CUDA lacks `libdevice.10.bc`; rerunning as an
actual CPU test passed the suite. No code was changed to conceal that
environmental failure.

The tests cover all required unit gates:

```text
O2 shape and node contract                         PASS
negative-MSE sign                                 PASS
clean O2 detached                                 PASS
adversarial O2 -> image -> texture gradient       PASS (synthetic CPU)
legacy objective formula/default                  PASS
O2/P2 token RMS reduction                         PASS
P2 [B,256,2048]                                   PASS (synthetic CPU)
pi0.5 Torch eval/no-grad and embed-prefix node    PASS (synthetic CPU)
sample/raw-image hash identity and fail-fast      PASS
pair scene equality and camera-field adapter      PASS
```

All four new CLI entrypoints also load their `--help` paths successfully, and
all edited Python files pass `py_compile`. These are CPU engineering results;
they are not substitutes for real OpenVLA, nvdiffrast, LIBERO, or pi0.5 runs.

### Runtime validation evidence

This workstation has neither:

```text
/data/xiaomengqi/src/LIBERO-joint
/data/huangsimin/openvla-7b-finetuned-libero-spatial
the tex3d-openvla server environment
a PyTorch-visible CUDA device
```

The user executed server validation using
`LIBERO_ROOT=/data/xiaomengqi/src/LIBERO-joint/` and stored run artifacts under
`/data/xiaomengqi/logs/`. The authoritative OpenVLA environment used Torch
2.2.0+cu121, Transformers 4.40.1, and Tokenizers 0.19.1. The confirmed joint
OpenVLA/OpenPI witness Python is
`/data/xiaomengqi/src/shared-feature-tex3d/.venv-joint/bin/python`; this
environment provides Torch 2.6.0+cu124, Transformers 4.53.2, and the required
OpenPI Transformers patch. Formal statistics used SciPy 1.17.1. The PyTorch
checkpoint root is `/data/xiaomengqi/checkpoints/pi05_libero_pytorch`; its
`model.safetensors` SHA256 is
`feeedaf6abe1601f8fb24041e21ae8c022b91141ebf1616678cfd2ea8640a09e`.
The OpenPI and validated witness-adapter commits are respectively
`15a9616a00943ada6c20a0f158e3adb39df2ccac` and
`fffea7571fcde7922b0d0abc1a56d1e88439c011`.
The source GPU smoke passed with a finite loss, finite/nonzero O2 displacement
and texture gradient, a changed parameter, a respected texture budget, pi0.5
absent from training, and successful XML/texture restoration. The first witness
collection initially stopped before feature extraction because the attacked
texture was also visible in the separately rendered wrist image. This is
expected image content change, not a scene-state change; the corrected
collection and OpenVLA O2 extraction then passed. The first PyTorch pi0.5 run
stopped because the witness treated the official five-value preprocessing
tuple as an object with image attributes. A regression test now reproduces the
official tuple contract and passes with explicit unpacking. The corrected
PyTorch rerun then passed all shape, provenance, frozen-model, no-grad, and
official-prefix-slice checks.

The formal training run wrote
`/data/xiaomengqi/logs/step1/step1-o2-p2-formal-v1` from code HEAD
`6ec34f1c20e18aa688080de24cc2df2894fee713`. Its frozen configuration has ten
training states (0-9), one frame per state, ten O2-only optimization iterations,
and `num_trials_per_task=0`. The latter deliberately disables policy rollout;
therefore no task success or attack success rate was measured. The legacy
entrypoint misleadingly printed `Attack success rate: 0.00%` for the zero
episode case. This was diagnosed as a presentation bug, not a failed task, and
the post-run code now reports `Task success rate: NOT EVALUATED (rollout
disabled)` when the episode count is zero. The formal texture was not retrained.

The synchronized formal training artifacts pass these checks:

```text
iterations / training frames                 10 / 10
finite scalar metrics                        PASS
finite, nonzero texture gradients            PASS (all 10 iterations)
finite, nonzero image gradients              PASS (all 10 iterations)
loss == -O2 displacement                     PASS (all 10 iterations)
action loss / legacy feature loss            0 / 0 (all 10 iterations)
final O2 displacement                        0.008688865043222905
final texture gradient norm                  0.00012498378055170178
maximum texture perturbation                 0.25123265385627747
renderer epsilon                             0.5019607843137255
texture budget                               PASS
pi0.5 loaded during training                 false
texture SHA256                               sha256:8856483b0c93789082c93b0d88a0d6cda37c3e5126d98fc1f4f192ae184aca75
XML / clean texture restoration              PASS
```

The frame contract contains exactly ten frames, each with both shared-texture
instances visible and the frozen 512-to-224 deployment-view path. Formal pair
collection and both consumers then passed:

```text
MuJoCo pair states                          exactly 10-19
raw pair shape / dtype                     [512,512,3] / uint8
scene, robot, camera equality              PASS for all 10 pairs
no policy action between pair              PASS for all 10 pairs
camera mapping                             agentview_image -> base_0_rgb
fixed pi0.5 wrist input hash               PASS for all 10 pairs
OpenVLA O2                                 [10,256,4096], finite
pi0.5 P2                                   [10,256,2048], finite
OpenVLA / pi0.5 sample and RGB hashes      exact match
pi0.5 backend                              PyTorch PI0Pytorch
pi0.5 eval / no-grad / parameters frozen   PASS
P2 equals official embed_prefix slice      PASS
required formal artifact files             23/23
```

Final server checks also passed. The current XML did not reference the formal
run directory, the backup search was empty, and the restored asset hashes were:

```text
akita_black_bowl.xml  18c1074cfa09baea739bb75928f9bd2bd80e22ac18655f6a27f005dbf77ccfda
texture.png            8a646d98b084b7400dd91beed9c83e10b4a4dca572896b3974356b7937a4bd85
```

The exact fail-closed commands are recorded in
`docs/step1-server-validation.md`. The executed stages were CPU regression,
source GPU smoke, one-pair witness smoke, formal O2-only training, pair
collection, authoritative OpenVLA O2 extraction, frozen PyTorch pi0.5 P2
extraction, formal correlation analysis, artifact audit, and final restoration
audit, in that order. Formal training was executed exactly once and was not
repeated after held-out results were observed.

### Smoke measurements

The one-pair smoke used state 10 only and the source-smoke texture
`sha256:e77ee3d7b1f6e2351d2f517d0690521a557f00ce7014b95782c2b2477375fcde`.
Both consumers verified the same raw pair:

```text
clean RGB sha256:d17110ecbb1c3c5d4b74a0f518137953f51b5f81b7a202f38a7a7cae23a30722
adv RGB   sha256:1f40a02f8e9c481ea6ad02fd5d119d1b545b073cc43e1f1502009a78b1c9c025
O2 shape  [1,256,4096]
P2 shape  [1,256,2048]
d_O2      0.0005087100718570168
d_P2      0.0036383388960389983
```

These values validate the one-pair runtime path only. State-level or
token-index correlation is undefined for one state and was not computed.

## Scientific observation

The formal run answers the three Step 1 questions as follows:

1. **Yes.** The O2-only texture induced nonzero held-out OpenVLA O2
   displacement in all 10 states.
2. **Yes.** The same frozen texture induced nonzero pi0.5 P2 displacement in
   all 10 states.
3. **Mixed.** State-level displacement did not show a positive monotonic trend
   (`rho=-0.28484848484848485`, `p=0.4250381548921454`, `N=10`). The
   token-index displacement trend was positive in every state, with rho between
   0.4249942778667887 and 0.5202484836346989.

Formal displacement summaries:

```text
                    mean                  median                min                   max
d_O2  0.008078825434393309  0.008304925850220052  0.007228606895869299  0.008632323576347101
d_P2   0.09625726137687679   0.09234936414121768   0.05279574134699715   0.13756997334533874
```

The complete state distribution is:

| State | d_O2 | d_P2 | token-index rho | token p-value |
|---:|---:|---:|---:|---:|
| 10 | 0.008263202015538111 | 0.12799719329122 | 0.45764548523689624 | 1.17529923665133e-14 |
| 11 | 0.007247646776671244 | 0.10796981092662038 | 0.4323043030441748 | 4.419540124014724e-13 |
| 12 | 0.008363933207736348 | 0.1316051374127125 | 0.4655084115358205 | 3.582978677291192e-15 |
| 13 | 0.008201699032451241 | 0.06888648640705074 | 0.4379069867246509 | 2.033690551655352e-13 |
| 14 | 0.008458455416954343 | 0.09249915798294617 | 0.5145013160906385 | 1.0549863996370323e-18 |
| 15 | 0.007672901676152591 | 0.13756997334533874 | 0.5202484836346989 | 3.720966240720909e-19 |
| 16 | 0.008372836061310807 | 0.09219957029948919 | 0.4249942778667887 | 1.1911322881415102e-12 |
| 17 | 0.007228606895869299 | 0.0857726493588114 | 0.4677636472877088 | 2.534035312309896e-15 |
| 18 | 0.008632323576347101 | 0.05279574134699715 | 0.43278496223392077 | 4.137140827528544e-13 |
| 19 | 0.008346649684901994 | 0.06527689339758155 | 0.45614914740215146 | 1.4683678930967824e-14 |

Token-index rho summary:

```text
mean    0.460980702105745
median  0.45689731631952385
min     0.4249942778667887
max     0.5202484836346989
N       256 token indices per state
```

## Interpretation boundary

The formal result supports only:

> An OpenVLA-only optimized texture also induces representation response in
> pi0.5, with the observed cross-state/token trend reported quantitatively.

Without random/matched perturbation baselines and action relevance, it must not
be described as proof of a shared transferable vulnerability. O2 and P2 both
have 256 tokens, but compatible spatial transforms and raster ordering have not
been established here; the implemented metric is therefore labelled only as a
`token-index displacement trend`, not spatial alignment. The token p-values are
descriptive outputs over 256 token indices and do not establish independent
spatial samples. The non-significant state-level result must not be reframed as
cross-state agreement. No rollout success/ASR, action relevance, random or
matched perturbation baseline, causal vulnerability, or Step 2 method is
claimed.

## Artifact state

The server smoke artifacts were synchronized read-only under:

```text
experiment_inbox/step1/source-smoke-20260904-173550/
```

The complete formal artifacts are stored on the server at
`/data/xiaomengqi/logs/step1/step1-o2-p2-formal-v1/` and synchronized read-only
under:

```text
experiment_inbox/step1/step1-o2-p2-formal-v1/
  config.json
  asset_restoration.json
  training/
  heldout_pairs/
  openvla/
  pi05/
  analysis/
```

Important artifacts include:

```text
training/final_attack_texture.png
training/parameter.pt
training/loss_history.npy
heldout_pairs/manifest.json
openvla/o2_clean.npz
openvla/o2_adv.npz
openvla/o2_residuals.npz
openvla/o2_state_metrics.csv
openvla/o2_token_rms.npz
pi05/p2_clean.npz
pi05/p2_adv.npz
pi05/p2_residuals.npz
pi05/p2_state_metrics.csv
pi05/p2_token_rms.npz
analysis/displacement_summary.csv
analysis/token_spearman.csv
analysis/summary.json
```

The formal artifact audit found all 23 required files, verified the frozen
texture hash across training, pairs, both consumers, and analysis, and
recomputed every displacement and correlation. The server worktree was clean at
`6d961c001d2d079483ab071ed4b55b566f6779a8` before this final report-only
update.
