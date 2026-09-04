# Step 1 O2-to-P2 transfer report

## Status

```text
STEP 1 NOT COMPLETE — RUNTIME VALIDATION BLOCKED
```

The implementation and CPU contracts are complete, but the required source GPU
smoke, one-pair witness smoke, and formal experiment have not run. Therefore no
scientific transfer result is claimed in this report.

## Git provenance

```text
repository          /home/xmq/src/tex3d
base commit         6ee46c18c78981cfa86b45edaab20c11db030483
branch              feat/step1-o2-p2-transfer-mvp
implementation HEAD 4e4dbe5 (before this report commit)
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
  robot/scene/camera/wrist equality checks;
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

The validated released pi0.5 checkpoint is JAX/NNX (`params/`, no
`model.safetensors`). JAX ordinary forward execution does not record an
autograd tape and `PaliGemma.img` is explicitly called with `train=False`. The
witness also runs inside `torch.no_grad()` and calls `eval()` when a backend
provides it. The formal JAX/NNX model has no PyTorch-style global `eval()` API;
this backend fact is emitted explicitly rather than misreported.

### CPU regression evidence

Pre-change baseline:

```text
PYTHONDONTWRITEBYTECODE=1 \
/home/xmq/.virtualenvs/modified-tex3d/bin/python -m pytest -q tests/unit
30 passed in 7.75s
```

Final implementation suite:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/xmq/.virtualenvs/modified-tex3d/bin/python -m pytest -q tests/unit
47 passed in 5.52s
```

The explicit empty `CUDA_VISIBLE_DEVICES` is required for a CPU regression on
this workstation. One intermediate invocation let TensorFlow discover the
local GPU and failed because local CUDA lacks `libdevice.10.bc`; rerunning as an
actual CPU test passed all 47 tests. No code was changed to conceal that
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
pi0.5 eval/no-grad guard                          PASS (synthetic CPU)
sample/raw-image hash identity and fail-fast      PASS
pair scene equality and camera-field adapter      PASS
```

All four new CLI entrypoints also load their `--help` paths successfully, and
all edited Python files pass `py_compile`. These are CPU engineering results;
they are not substitutes for real OpenVLA, nvdiffrast, LIBERO, or pi0.5 runs.

### Runtime validation blocker

This workstation has neither:

```text
/opt/libero
/data/huangsimin/openvla-7b-finetuned-libero-spatial
the tex3d-openvla server environment
a PyTorch-visible CUDA device
```

The configured validation server
`xiaomengqi@59.78.189.196` was contacted twice with a 10/15-second connection
timeout. Both attempts failed before authentication with:

```text
ssh: connect to host 59.78.189.196 port 22: Connection timed out
```

Consequently, the following required gates are **NOT EXECUTED**:

```text
source GPU smoke
one held-out-pair OpenVLA/P2 witness smoke
formal train 0-9
formal analyze 10-19
server XML/texture restoration audit
formal artifact completeness audit
```

The exact fail-closed commands are recorded in
`docs/step1-server-validation.md`.

## Scientific observation

No real-model measurements are available. The three Step 1 questions remain
unanswered:

1. Held-out OpenVLA O2 displacement: **NOT MEASURED**.
2. Frozen-texture pi0.5 P2 displacement: **NOT MEASURED**.
3. State-level and token-index O2/P2 trends: **NOT MEASURED**.

There are no formal `rho`, p-value, per-state token correlations, or O2/P2
displacement values to report. No experiment directory is claimed as formal.

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

No smoke or formal runtime artifacts were generated locally. Once the server is
reachable, the validated pipeline will write:

```text
experiments/step1/<run-id>/
  config.json
  asset_restoration.json
  training/
  heldout_pairs/
  openvla/
  pi05/
  analysis/
```

Formal completion requires updating this report with runtime commands, hashes,
artifact paths, all displacement metrics, both Spearman analyses, final asset
checks, and a clean committed HEAD.
