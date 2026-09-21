# Phase 3 — Action-Predictive Texture Objective

## Status

**EXPLORATORY PIPELINE-FEASIBILITY AUTHORIZED with a fixed expanded seed-7
provisional probe artifact. Authoritative Phase 2B v3 remains NOT FROZEN /
BLOCKED.**

The first historical 1-step/10-step smoke used a PI0.5 probe fitted on
`P2 / sqrt(2048)` while the differentiable runtime supplied native `embed_image()`
P2. Those smoke outputs and `phase2b-primary-action-probes-v2` are retained for
provenance but are invalid for corrected Phase 3 scientific interpretation.

Expanded Pilot v0.3 has now completed corrected O2/P2 extraction, seed-7 probe
fitting, and corrected-P2 six-seed stability. The seed-7 probes have strong
held-out action predictivity, but stability failed (`CV=0.05564`, minimum
principal cosine `0.65546`, maximum projection distance `0.61612`). This is a
full-subspace reproducibility limitation, not an artifact/provenance failure.
Therefore `phase2b-primary-action-probes-v3` has not been created.

The next phase is explicitly **fixed-seed-7 exploratory Phase 3 pipeline
feasibility** using only
`phase2-expanded-seed7-provisional-action-probes-v1`. Its question is whether a
fixed action-predictive probe can drive the complete representation-objective →
gradient aggregation → texture optimization → policy-evaluation pipeline to a
meaningful behavioral effect. It does not establish general probe stability,
unique action-subspace identification, held-out VLA transfer, or general seed
robustness. These unresolved Phase 2 limits remain attached even if an
exploratory Phase 3 run succeeds.

This phase implements a source-model action-predictive texture objective. It does not establish held-out transfer, causal action relevance, decoded-action change, or policy degradation until the corresponding server runs are complete.

## Phase 2 provenance

The current exploratory Phase 3 handoff uses only the exact seed-7 primary
projected nodes from Expanded Pilot v0.3 and
`pilot-v0.3-expanded-split-v1`:

| Model | Node | Shape | Phase 2A held-out MSE | Mean baseline MSE | W TRAIN-nullspace fraction |
|---|---|---:|---:|---:|---:|
| OpenVLA | O2 | `[256,4096]` | 0.29370 | 0.99444 | 0.38858 |
| PI0.5 | corrected P2 | `[256,2048]` | 0.32206 | 1.00244 | 0.22191 |

The provisional promotion validates the expanded Phase 2A inventory, exact
seed-7 configuration, split identity, collection and representation-manifest
hashes, corrected P2 identity, stability provenance, source artifact invariance,
and byte-identical O2/P2 copies. It performs no probe refitting. The output
inventory records SHA-256 for every copied file.

The stability result remains a failed/unresolved qualification. Corrected P2
held-out predictivity is strong, but all three frozen stability checks fail.
The Phase 3 task explicitly selects the fixed seed-7 O2/P2 artifact for
pipeline feasibility only; it does not relabel the complete Phase 2 diagnostic
as stable or create authoritative Phase 2B v3.

Frozen probe protocol:

```text
Linear(D, 7, bias=False)
seed = 7
learning_rate = 1e-3
weight_decay = 1e-4
steps = 2000
action_std_epsilon = 1e-6
probe_reg = 1e-4
TRAIN / HELD-OUT = 1914 / 456 observations
```

The superseded historical Phase 2B artifact is retained only for provenance:

```text
experiment_inbox/shared-feature-phase3/phase2b-primary-action-probes-v2/
```

The current exploratory artifact is:

```text
phase2-expanded-seed7-provisional-action-probes-v1/
```

## Objective

For each model-specific native projected representation (F\in\mathbb{R}^{B\times256\times D}) and frozen probe (W\in\mathbb{R}^{7\times D}):

\[
Z(F)=\operatorname{mean}_{token}(F)W^T.
\]

The two probes remain separate: `W_O2` only maps OpenVLA O2 and `W_P2` only maps PI0.5 P2. The runtime does not use PCA, CCA, feature averaging, or a shared probe.

For a clean/adversarial pair:

\[
L_{mag}=-\operatorname{MSE}(Z_a,Z_c),
\]

\[
L_{dir}=\cos(Z_a,Z_c),
\]

\[
L=L_{mag}+\lambda_{dir}L_{dir}.
\]

All seven normalized deployed-action coordinates are equally weighted. The probe weights and TRAIN-only action-normalization statistics are registered as frozen buffers. Clean O2/P2 and clean action coordinates are computed once per frame under `torch.no_grad()`, detached, cloned, and reused.

## Lambda calibration

Before any texture update, the runner uses all ten frozen training frames at the deterministic native-GE texture initialization. It obtains independent raw texture gradients for magnitude and direction in each model, then computes:

\[
r_m=\operatorname{median}_{frames}
\frac{\|g_{mag,m}\|_2}{\|g_{dir,m}\|_2+10^{-12}},
\qquad
\lambda_{dir}=\sqrt{r_O r_P}.
\]

Calibration is read-only and verifies exact texture-parameter equality before and after. Missing, disconnected, zero, or non-finite component gradients block the run. The selected value and all frame-level loss, norm, mean-absolute-gradient, and component-cosine diagnostics are written to `lambda_calibration.json`.

No corrected empirical `lambda_dir` is reported. The historical value from the
invalid v2-probe smoke must not be reused. Calibration will be rerun against the
validated provisional seed-7 artifact before the first corrected smoke.

## Training

The new objective is an opt-in fourth mode, `action_predictive_gradient_ensemble`. The original `shared_cca`, `native_gradient_ensemble`, and `multilevel_native_gradient_ensemble` paths remain unchanged.

The model-gradient hierarchy reuses the existing native GE implementation:

```text
per-frame OpenVLA / PI0.5 action-predictive gradient
→ mean over the complete frame batch within each model
→ separate mean-absolute model normalization
→ average normalized model gradients
→ one sign-PGD texture update
```

The frozen pilot remains task 0, Akita black bowl, states 0–9, one frame per state, effective batch 10, 500 iterations, step 0.05, seed 7, and the existing renderer epsilon and texture parameterization.

Each step records native O2/P2 MSE, action-coordinate MSE, direction cosine, clean/adv/delta coordinate norms, per-coordinate signed/absolute/squared displacement, raw/normalized model-gradient relationship, ensemble norm, texture update, and budget status. Component-gradient decomposition is added only at iterations 0, 50, 100, 250, and 499.

Historical native O2/P2 displacement and magnitude-only objectives are diagnostics in the first pilot. They are not independently trained ablation conditions.

## Runtime validation plan

The dedicated entry point is:

```text
scripts/phase3_action_aware_optimization.py
```

The first corrected server run will use one selected training frame for one update while still materializing the frozen ten-frame substrate and calibrating on all ten frames. The second corrected run will use all ten frames for ten updates. These outputs are explicitly marked non-authoritative. Neither corrected run has completed. A 500-step run starts only after both and the paired-noise consistency diagnostic pass.

Expected evidence includes frozen-probe hashes, clean-reference detachment, finite/nonzero per-model and ensemble gradients, one update per iteration, unchanged W buffers, texture-budget compliance, and no retained-graph memory growth.

## Source rollout

The existing paired source evaluator remains unchanged. After a successful 500-step run, the same final baked texture will be evaluated independently against OpenVLA and PI0.5 using matched clean/adversarial initial-state trials.

The required comparison will be filled only after rollout completion:

| Objective | OpenVLA conditional effect | PI0.5 conditional effect |
|---|---:|---:|
| Historical Native O2/P2 GE | 43.48% conditional ASR | 0% conditional ASR |
| Phase 3 action-aware GE | pending | pending |

## Interpretation boundary

The implemented coordinates encode linearly readable action-predictive information under the frozen Phase 2 protocol. A large action-coordinate displacement alone does not establish decoded-action change, texture-accessible causal relevance, policy degradation, or transferability. The final state remains `IMPLEMENTATION_COMPLETE` until the CUDA smoke, 500-step training, paired OpenVLA rollout, paired PI0.5 rollout, and artifact checks are complete.
