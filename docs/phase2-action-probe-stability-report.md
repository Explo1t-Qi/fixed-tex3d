# Phase 2 — Action-Probe Stability Diagnostic

## Scope

This CPU-only diagnostic measures prediction stability and candidate action-predictive subspace stability across random probe initializations. It does not establish causal action relevance, action controllability, texture effectiveness, or transferability, and it does not modify the Phase 2A/2B protocol.

> **Historical PI0.5 P2 diagnostic.** The P2 rows in this report were computed
> from the superseded `P2 / sqrt(2048)` Phase 2A archive. They remain provenance
> for that fitting condition and are not evidence that the corrected native P2
> probe is stable. The corrected 200-observation P2 representation archive now
> passes identity validation, but its six-seed diagnostic is still pending.

## Frozen protocol

- Dataset: frozen Pilot v0.2, 200 observations.
- Split: `pilot-v0.2-c5-split-v1`, 160 TRAIN and 40 HELD-OUT observations.
- Probe: `Linear(D,7,bias=False)`.
- AdamW: learning rate `1e-3`, weight decay `1e-4`, 2,000 steps.
- Action normalization: TRAIN statistics only, epsilon `1e-6`.
- Projection ridge: `1e-4`.
- Seeds: [1, 2, 3, 4, 5, 7].

## Aggregate stability

| Node | Held-out MSE mean | std | min | max | Mean principal cosine | Minimum principal cosine | Projection distance mean | max | Phase 2A seed-7 W max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA O2 | 0.618122 | 0.001885 | 0.615699 | 0.620112 | 0.977263 | 0.928177 | 0.297742 | 0.334208 | 0 |
| OpenVLA O-deep | 0.526010 | 0.002565 | 0.522108 | 0.530185 | 0.880648 | 0.695771 | 0.659440 | 0.693893 | 0 |
| PI0.5 P2 | 0.664591 | 0.001900 | 0.661821 | 0.666516 | 0.991469 | 0.949208 | 0.181865 | 0.226768 | 0 |
| PI0.5 P-deep | 0.554127 | 0.063120 | 0.493122 | 0.685885 | 0.357041 | 0.002323 | 1.298007 | 1.308061 | 0 |

Detailed per-seed metrics, signed action-row cosines, all seven principal-angle cosines for every pair, low-rank projection distances, and comparisons against seed 7 are stored in `per_seed_metrics.json` and `subspace_similarity.json`.

The newly fitted seed-7 `W` is element-wise identical to the Phase 2A seed-7 artifact for all four nodes (`max_abs_difference = 0.0`). This confirms that the diagnostic reproduces the frozen fitting protocol before comparing alternative initializations.

## Prediction stability

OpenVLA O2, OpenVLA O-deep, and PI0.5 P2 have held-out MSE coefficients of variation of `0.00305`, `0.00488`, and `0.00286`, respectively. Their held-out prediction performance is effectively stable across the six initializations.

PI0.5 P-deep has coefficient of variation `0.11391`, with held-out MSE ranging from `0.49312` to `0.68588`. Seed 3 retains TRAIN normalized MSE `0.12139` and final optimization loss `0.09797`, whereas the other seeds finish between approximately `2.5e-6` and `2.8e-3`. Under the frozen 2,000-step schedule, P-deep probe optimization is therefore initialization-sensitive.

Per-action R² is stable for O2 and P2. O-deep shows only small variation. P-deep is more variable, especially for `y` where the R² standard deviation is `0.29991`; its other per-action R² standard deviations range from `0.04319` to `0.09265`.

## Direction and subspace stability

| Node | Range of mean signed action-row cosine by action | Lowest signed row cosine over all pairs/actions | Mean pairwise principal cosine | Lowest principal cosine | Max projection distance |
|---|---:|---:|---:|---:|---:|
| OpenVLA O2 | 0.9567–0.9943 | 0.9368 | 0.9773 | 0.9282 | 0.3342 |
| OpenVLA O-deep | 0.7702–0.9599 | 0.7497 | 0.8806 | 0.6958 | 0.6939 |
| PI0.5 P2 | 0.9909–0.9963 | 0.9736 | 0.9915 | 0.9492 | 0.2268 |
| PI0.5 P-deep | 0.1050–0.4881 | -0.1429 | 0.3570 | 0.0023 | 1.3081 |

Historical scaled PI0.5 P2 is stable under every predeclared criterion for that superseded fitting condition. This result must be rerun on corrected native P2 before Phase 2B v3. OpenVLA O2 has highly consistent action rows and principal angles, but its maximum ridge-projection distance of `0.3342` exceeds the `0.25` threshold. OpenVLA O-deep produces stable predictions from only moderately stable row spaces. Its gripper direction is the least consistent action row, and its minimum principal cosine is `0.6958`.

PI0.5 P-deep is strongly under-determined. Its gripper row cosine has pairwise mean `0.1050` and reaches `-0.1429`; the minimum principal cosine is nearly zero and the projection distance is close to the maximum expected for distinct equal-rank subspaces. This is not solely a consequence of seed 3 failing to converge: seeds 1, 2, 4, and 5 each have mean principal cosine only about `0.34–0.35` relative to seed 7.

## Seed-7 reference comparisons

| Node | Other seed | Held-out MSE difference | Mean principal cosine | Minimum principal cosine | Projection distance |
|---|---:|---:|---:|---:|---:|
| OpenVLA O2 | 1 | -0.002873 | 0.9756 | 0.9374 | 0.3091 |
| OpenVLA O2 | 2 | 0.000380 | 0.9827 | 0.9558 | 0.2615 |
| OpenVLA O2 | 3 | -0.004033 | 0.9775 | 0.9548 | 0.2975 |
| OpenVLA O2 | 4 | -0.003474 | 0.9846 | 0.9564 | 0.2466 |
| OpenVLA O2 | 5 | 0.000343 | 0.9761 | 0.9406 | 0.3060 |
| OpenVLA O-deep | 1 | -0.003756 | 0.8876 | 0.7307 | 0.6420 |
| OpenVLA O-deep | 2 | -0.000986 | 0.8787 | 0.7123 | 0.6644 |
| OpenVLA O-deep | 3 | 0.002233 | 0.8781 | 0.7054 | 0.6651 |
| OpenVLA O-deep | 4 | 0.004320 | 0.8841 | 0.7317 | 0.6512 |
| OpenVLA O-deep | 5 | -0.000938 | 0.8864 | 0.7332 | 0.6454 |
| PI0.5 P2 | 1 | -0.000269 | 0.9963 | 0.9860 | 0.1208 |
| PI0.5 P2 | 2 | -0.003974 | 0.9930 | 0.9835 | 0.1673 |
| PI0.5 P2 | 3 | -0.001980 | 0.9917 | 0.9716 | 0.1810 |
| PI0.5 P2 | 4 | -0.004537 | 0.9921 | 0.9811 | 0.1769 |
| PI0.5 P2 | 5 | 0.000158 | 0.9906 | 0.9725 | 0.1933 |
| PI0.5 P-deep | 1 | 0.024857 | 0.3373 | 0.0501 | 1.3081 |
| PI0.5 P-deep | 2 | 0.053491 | 0.3478 | 0.1265 | 1.3066 |
| PI0.5 P-deep | 3 | 0.178962 | 0.3442 | 0.1100 | 1.3031 |
| PI0.5 P-deep | 4 | -0.013801 | 0.3483 | 0.1105 | 1.3057 |
| PI0.5 P-deep | 5 | 0.039719 | 0.3521 | 0.0606 | 1.2982 |

## Decision rule

The predeclared `STABLE` rule requires every node to have held-out MSE coefficient of variation at most 0.05, minimum pairwise principal cosine at least 0.90, and maximum projection relative Frobenius distance at most 0.25. These thresholds classify reproducibility; they do not tune the probe or select a node from HELD-OUT performance.

## Result

Status: `NEEDS_REVIEW`. Nodes requiring review: ['openvla/o2', 'openvla/deep', 'pi05/deep'].

The historical Phase 2A action-predictability conclusion remains supported for its recorded fitting condition. The concrete candidate direction/subspace is not equally reproducible across nodes. The historical scaled PI0.5 P2 result passed this diagnostic, but corrected native P2 is not yet classified. OpenVLA O2 is close but misses the projection-distance criterion; O-deep and especially P-deep are under-determined in the current high-dimensional, small-sample fitting setup.

Phase 2B freeze should therefore remain paused for method review. The diagnostic does not select or apply a remedy. Stronger regularization, a closed-form ridge/minimum-norm probe, or more observations are follow-up candidates that require an explicit protocol decision and a new controlled experiment.

The source artifact trees were hash-snapshotted before and after the diagnostic and remained unchanged.
