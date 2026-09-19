# Phase 2 — Action-Predictive Representation Report

## 1. Scope and status

This experiment evaluates action predictability only. It does not establish causal action relevance, texture effectiveness, controllability, policy degradation, or transferability.

This document records the Phase 2A dataset-sufficiency experiment. Its artifacts are candidates pending Phase 2B formal materialization; they are not yet authoritative Phase 3 inputs.

Phase 2A result: `COMPLETE — Case A`. The existing 200-observation dataset has adequate action coverage and all four candidate representations show non-trivial held-out linear predictability. Dataset densification is not required before Phase 2B.

## 2. Dataset and split

The input is the frozen Pilot v0.2 collection:

- 10 LIBERO-Spatial tasks;
- 5 trajectory groups per task;
- 4 observations per trajectory at target progress 0.10, 0.40, 0.70, and 0.90;
- 50 trajectory groups and 200 observations in total;
- trajectory identity `(task_id, initial_state_id)`.

Both model extractors produced 200 unique, identically ordered samples from collection manifest SHA-256 `9f23c78500840d812c1b3dbe02e79a735d4dc3a8e8e4c2b38713f1b10a5b777b`. Every archive hash, sample identity, tensor shape, and finite-value check passed.

The frozen `pilot-v0.2-c5-split-v1` group-aware split contains 40 TRAIN groups (160 observations) and 10 HELD-OUT groups (40 observations), with one held-out trajectory group per task. TRAIN and HELD-OUT group sets are disjoint. Exact group and sample identities are stored in `split.json`.

## 3. Action targets and coverage

OpenVLA representations are paired with the model's own decoded 7-D LIBERO action after the existing gripper binarization and inversion. PI0.5 representations are paired with the first deployed 7-D action from the model's own action chunk under deterministic per-sample diffusion noise.

Each task contributes 20 actions and each target progress contributes 50. No action dimension is near-constant overall or within any task for either model. Approximate overall standard deviations are:

| Model | x | y | z | rot_x | rot_y | rot_z | gripper |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA | 0.4158 | 0.3616 | 0.5232 | 0.0293 | 0.0616 | 0.0441 | 0.9928 |
| PI0.5 | 0.4368 | 0.3685 | 0.5335 | 0.0317 | 0.0666 | 0.0480 | 0.9911 |

OpenVLA has 88 negative and 112 positive gripper actions. PI0.5 has 87 negative and 113 positive gripper actions; its raw deployed values vary slightly around -1 and +1. For both models, all 50 observations at progress 0.10 use the negative gripper state, while later progress positions contain both states. This stage-specific structure is recorded but does not make the overall gripper target near-constant.

Full mean, standard deviation, extrema, percentiles, near-zero fractions, task breakdowns, progress breakdowns, and raw gripper frequencies are stored under `action_audit/`.

## 4. Representation nodes

| Model | Node | Definition | Shape |
|---|---|---|---:|
| OpenVLA | O2 | Multimodal projector output before Llama | `[256,4096]` |
| OpenVLA | O-deep | `language_model.model.layers[15]` output after the first 16 of 32 blocks; visual slice `[:,1:257,:]` | `[256,4096]` |
| PI0.5 | P2 | Base-camera PaliGemma projector output with official `1/sqrt(hidden_size)` scaling | `[256,2048]` |
| PI0.5 | P-deep | `paligemma_with_expert.paligemma.language_model.layers[8]` output after the first 9 of 18 prefix blocks; base-camera slice `[:,0:256,:]` | `[256,2048]` |

The OpenVLA and PI0.5 repeat-forward checks produced maximum absolute differences of `0.0` for the projected node, deeper node, and decoded action. Extraction used `torch.inference_mode()` and populated no VLA parameter gradients.

## 5. Probe protocol

Each representation is mean-pooled over all 256 visual tokens and detached. Each model/node pair uses an independent `Linear(D,7,bias=False)` probe trained against model-specific action targets normalized with TRAIN-only statistics.

The fixed Phase 2A configuration is AdamW, learning rate `1e-3`, weight decay `1e-4`, 2,000 steps, seed 7, action standard-deviation epsilon `1e-6`, and action-subspace ridge `1e-4`. HELD-OUT results were not used to alter this configuration.

## 6. Aggregate results

All values below use normalized action coordinates.

| Model | Node | D | Train MSE | Held-out MSE | Mean baseline MSE | MSE reduction vs baseline |
|---|---|---:|---:|---:|---:|---:|
| OpenVLA | O2 | 4096 | 0.000309 | 0.619732 | 1.109212 | 44.13% |
| OpenVLA | O-deep | 4096 | 0.000045 | 0.525864 | 1.109212 | 52.59% |
| PI0.5 | P2 | 2048 | 0.174961 | 0.666358 | 1.069136 | 37.67% |
| PI0.5 | P-deep | 2048 | 0.000146 | 0.506923 | 1.069136 | 52.59% |

The large TRAIN/HELD-OUT gaps, especially for the deeper nodes, are expected caution signals in the high-dimensional, low-sample regime. The scientific evidence comes from consistent improvement over the TRAIN-mean predictor on held-out trajectory groups, not from near-zero TRAIN error.

## 7. Held-out per-action results

These metrics also use normalized action coordinates.

| Model | Node | Dimension | MSE | MAE | Pearson | R² |
|---|---|---|---:|---:|---:|---:|
| OpenVLA | O2 | gripper | 0.2048 | 0.3061 | 0.8963 | 0.7932 |
| OpenVLA | O2 | rot_x | 1.3788 | 0.9299 | 0.5290 | 0.2522 |
| OpenVLA | O2 | rot_y | 0.9945 | 0.7680 | 0.4980 | 0.1186 |
| OpenVLA | O2 | rot_z | 0.6828 | 0.6302 | 0.4631 | -0.0709 |
| OpenVLA | O2 | x | 0.4107 | 0.4760 | 0.8225 | 0.6691 |
| OpenVLA | O2 | y | 0.3624 | 0.4855 | 0.8405 | 0.6448 |
| OpenVLA | O2 | z | 0.3041 | 0.4499 | 0.8280 | 0.6589 |
| OpenVLA | O-deep | gripper | 0.1897 | 0.2650 | 0.9043 | 0.8084 |
| OpenVLA | O-deep | rot_x | 1.1195 | 0.7672 | 0.6374 | 0.3929 |
| OpenVLA | O-deep | rot_y | 0.8058 | 0.7046 | 0.5858 | 0.2858 |
| OpenVLA | O-deep | rot_z | 0.7100 | 0.6271 | 0.4697 | -0.1135 |
| OpenVLA | O-deep | x | 0.3451 | 0.4542 | 0.8529 | 0.7219 |
| OpenVLA | O-deep | y | 0.2968 | 0.4126 | 0.8550 | 0.7091 |
| OpenVLA | O-deep | z | 0.2141 | 0.3647 | 0.8738 | 0.7598 |
| PI0.5 | P2 | gripper | 0.4851 | 0.4806 | 0.7441 | 0.5110 |
| PI0.5 | P2 | rot_x | 1.4802 | 0.9343 | 0.4424 | 0.0690 |
| PI0.5 | P2 | rot_y | 0.7241 | 0.6942 | 0.6421 | 0.3171 |
| PI0.5 | P2 | rot_z | 0.4208 | 0.5303 | 0.6921 | 0.4782 |
| PI0.5 | P2 | x | 0.5270 | 0.5765 | 0.7654 | 0.5832 |
| PI0.5 | P2 | y | 0.3943 | 0.4968 | 0.7804 | 0.5960 |
| PI0.5 | P2 | z | 0.6330 | 0.6372 | 0.5497 | 0.1850 |
| PI0.5 | P-deep | gripper | 0.2113 | 0.3208 | 0.8942 | 0.7870 |
| PI0.5 | P-deep | rot_x | 1.2135 | 0.8979 | 0.5784 | 0.2367 |
| PI0.5 | P-deep | rot_y | 0.4382 | 0.5572 | 0.8140 | 0.5868 |
| PI0.5 | P-deep | rot_z | 0.5348 | 0.5829 | 0.6799 | 0.3368 |
| PI0.5 | P-deep | x | 0.4393 | 0.4869 | 0.8217 | 0.6525 |
| PI0.5 | P-deep | y | 0.4086 | 0.5103 | 0.7874 | 0.5813 |
| PI0.5 | P-deep | z | 0.3028 | 0.4196 | 0.8503 | 0.6102 |

The deeper node has the lowest aggregate held-out MSE for each model. Translational dimensions and gripper are generally more predictable than rotations. OpenVLA `rot_z` remains below the mean baseline by R² for both nodes, and PI0.5 P2 has weak `rot_x` and `z` evidence. These are retained as negative or weak per-dimension findings.

## 8. Artifact validation

All four probe weights have rank 7. `W` shapes are `[7,4096]` for OpenVLA and `[7,2048]` for PI0.5; corresponding `P_action` shapes are `[4096,4096]` and `[2048,2048]`. All tensors and TRAIN normalization arrays are finite, and no action dimension uses the near-constant fallback.

Probe predictions are bit-identical after serialization reload. Projection symmetry residuals are at most `3.62e-7`; ridge-idempotence residuals range from `2.90e-6` to `5.00e-4`. The artifact inventory contains hashes for 29 files.

Candidate evidence is stored under `experiment_inbox/shared-feature-phase2/phase2-action-probe-2a-20260919-203707/`. The model representations were extracted on the server; the dependency-light Phase 2A probe materialization was run locally on the synchronized, hash-validated archives.

## 9. Interpretation and Phase 2A decision

Under this protocol, all four representations contain linearly readable action-predictive information that generalizes across held-out trajectory groups. O-deep and P-deep contain the strongest aggregate signal among the tested nodes. This comparison establishes predictability only; it does not establish that these directions are causally action-relevant or controllable, or that optimizing a texture against them will alter policy behavior.

The Phase 2A evidence supports retaining the existing 200-observation dataset for action-predictability evaluation without immediate densification. A subsequent six-seed diagnostic found stable held-out prediction for O2, O-deep, and P2, but materially unstable fitted subspaces for O-deep and especially P-deep; P-deep prediction was also initialization-sensitive. The current status is therefore `NEEDS_REVIEW`, and Phase 2B freeze is paused pending an explicit probe-identification decision. See `docs/phase2-action-probe-stability-report.md`.
