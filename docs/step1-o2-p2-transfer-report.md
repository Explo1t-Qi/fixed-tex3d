# Step 1 O2-to-P2 transfer report

## Status

```text
STEP 1 FORMAL TRAINING COMPLETE — HELD-OUT ANALYSIS PENDING
```

The implementation and CPU contracts are complete. The source GPU smoke and
one-pair OpenVLA O2/PyTorch pi0.5 P2 witness smoke passed on the server. The
single frozen formal training run over states 0-9 also completed and its
artifacts passed the local read-only audit. Held-out states 10-19 have not yet
been analyzed, so no formal scientific transfer result is claimed.

## Git provenance

```text
repository          /home/xmq/src/tex3d
base commit         6ee46c18c78981cfa86b45edaab20c11db030483
branch              feat/step1-o2-p2-transfer-mvp
pre-formal code HEAD dc4ac355e5fc61bbb8182fc0bec5f1e06454fa45
formal training code HEAD 6ec34f1c20e18aa688080de24cc2df2894fee713
Step 0 code ancestor ff9fcbc8f84debb7693027a58490a0d856c8cdde
```

The branch was created only after the pre-existing Step 1 `task.md` change was
temporarily stashed, the Step 0 worktree was verified clean, and the task file
was restored on the new branch. `main` was not modified.

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
52 passed in 4.89s
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

### Runtime validation handoff

This workstation has neither:

```text
/data/xiaomengqi/src/LIBERO-joint
/data/huangsimin/openvla-7b-finetuned-libero-spatial
the tex3d-openvla server environment
a PyTorch-visible CUDA device
```

The user is executing server validation using
`LIBERO_ROOT=/data/xiaomengqi/src/LIBERO-joint/` and store run artifacts under
`/data/xiaomengqi/logs/`. The confirmed joint OpenVLA/OpenPI witness Python is
`/data/xiaomengqi/src/shared-feature-tex3d/.venv-joint/bin/python`; this
environment provides Torch 2.6.0+cu124, Transformers 4.53.2, and the required
OpenPI Transformers patch. The PyTorch checkpoint root is
`/data/xiaomengqi/checkpoints/pi05_libero_pytorch`.
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
instances visible and the frozen 512-to-224 deployment-view path. Consequently,
the following required gates remain **NOT EXECUTED** at this report revision:

```text
formal analyze 10-19
formal artifact completeness audit
```

The exact fail-closed commands are recorded in
`docs/step1-server-validation.md`.

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

The three formal Step 1 questions remain unanswered until the frozen 10-state
run is complete:

1. Held-out OpenVLA O2 displacement: **NOT MEASURED**.
2. Frozen-texture pi0.5 P2 displacement: **NOT MEASURED**.
3. State-level and token-index O2/P2 trends: **NOT MEASURED**.

There are no formal `rho`, p-value, or per-state token correlations to report.
No smoke directory is claimed as formal.

## Interpretation boundary

The unit tests establish interface, objective, gradient, provenance, and metric
calculation behavior with synthetic tensors. They do not establish an attack or
transfer effect.

Even after a future real run, nonzero P2 displacement alone will only support:

> An OpenVLA-only optimized texture also induces representation response in
> pi0.5, with the observed cross-state/token trend reported quantitatively.

Without random/matched perturbation baselines and action relevance, it must not
be described as proof of a shared transferable vulnerability. O2 and P2 both
have 256 tokens, but compatible spatial transforms and raster ordering have not
been established here; the implemented metric is therefore labelled only as a
`token-index displacement trend`.

## Artifact state

The server smoke artifacts were synchronized read-only under:

```text
experiment_inbox/step1/source-smoke-20260904-173550/
```

The formal training artifacts are synchronized read-only under:

```text
experiment_inbox/step1/step1-o2-p2-formal-v1/
  config.json
  asset_restoration.json
  training/
```

The remaining server analysis will add:

```text
/data/xiaomengqi/logs/step1/step1-o2-p2-formal-v1/
  heldout_pairs/
  openvla/
  pi05/
  analysis/
```

Formal completion requires updating this report with runtime commands, hashes,
artifact paths, all displacement metrics, both Spearman analyses, final asset
checks, and a clean committed HEAD.
