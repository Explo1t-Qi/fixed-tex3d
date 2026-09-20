# Phase 3 — Action-Predictive Texture Objective

## Status

**IMPLEMENTATION_COMPLETE; authoritative CUDA smoke and source pilot pending.**

This phase implements a source-model action-predictive texture objective. It does not establish held-out transfer, causal action relevance, decoded-action change, or policy degradation until the corresponding server runs are complete.

## Phase 2 provenance

Phase 3 uses only the seed-7 primary projected nodes from the frozen Pilot v0.2 dataset and `pilot-v0.2-c5-split-v1` split:

| Model | Node | Shape | Phase 2A held-out MSE | Frozen W SHA-256 |
|---|---|---:|---:|---|
| OpenVLA | O2 | `[256,4096]` | 0.619732 | `0377a285b180aaf80e7a038e6b036860389892b818368448c10966328eaed25c` |
| PI0.5 | P2 | `[256,2048]` | 0.666358 | `ba2351f836bb8c58de28f7aa125b0f7b901c8b08bec2346c21ef2c9107200369` |

The Phase 2B promotion validates the original Phase 2A inventory, exact seed-7 probe configuration, split identity, stability provenance, and seed-7 weight equality before copying O2/P2 artifacts into a fresh authority directory. The Phase 2A source inventory was rehashed after promotion with zero mismatches.

The stability result remains qualified. P2 passed all registered prediction/subspace checks. O2 passed prediction-CV and principal-angle checks, while its maximum ridge-projection relative distance (0.3342) exceeded the diagnostic threshold of 0.25. The Phase 3 task explicitly selects seed-7 O2/P2; this report preserves the O2 caveat rather than relabeling the complete Phase 2 diagnostic as stable.

Frozen probe protocol:

```text
Linear(D, 7, bias=False)
seed = 7
learning_rate = 1e-3
weight_decay = 1e-4
steps = 2000
action_std_epsilon = 1e-6
probe_reg = 1e-4
TRAIN / HELD-OUT = 160 / 40 observations
```

The versioned local Phase 2B artifact is:

```text
experiment_inbox/shared-feature-phase3/phase2b-primary-action-probes-v2/
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

No empirical `lambda_dir` is reported yet because the real dual-GPU calibration has not run.

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

The first server run uses one selected training frame for one update while still materializing the frozen ten-frame substrate and calibrating on all ten frames. The second run uses all ten frames for ten updates. These outputs are explicitly marked non-authoritative. A 500-step run is started only after both pass.

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
