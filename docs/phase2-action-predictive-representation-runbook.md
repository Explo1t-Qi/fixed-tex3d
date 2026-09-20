# Phase 2 — Action-Predictive Representation Runbook

This stage measures whether model-specific deployed 7-D actions are linearly readable from four frozen representation nodes. It does not test causal action relevance, texture effectiveness, policy degradation, or transferability.

## Frozen representation contract

| Model | Node | Definition | Shape per observation |
|---|---|---|---:|
| OpenVLA | O2 | Multimodal projector output before Llama | `[256,4096]` |
| OpenVLA | O-deep | Output of the last block in the first half of the 32-layer Llama tower; zero-based layer 15; visual slice `[1:257]` | `[256,4096]` |
| PI0.5 | P2 | `base_0_rgb` PaliGemma-ready projected visual tokens, including the official post-projector `1/sqrt(hidden_size)` scaling | `[256,2048]` |
| PI0.5 | P-deep | Output of the last block in the first half of the 18-layer PaliGemma prefix tower; zero-based layer 8; base-camera slice `[0:256]` | `[256,2048]` |

The runtime derives the midpoint from the actual decoder depth and also checks the expected authoritative depths of 32 and 18. O2/O-deep are captured during one OpenVLA action generation. P2/P-deep are captured during one PI0Pytorch action inference.

OpenVLA targets are the decoded action after the existing LIBERO gripper binarization and inversion. PI0.5 targets are the first deployed 7-D action from its action chunk. PI0.5 uses deterministic per-sample diffusion noise derived from the global seed and sample identity.

## Pipeline

1. `phase2_action_representation_extract.py --model openvla` creates OpenVLA feature/action archives.
2. `phase2_action_representation_extract.py --model pi05` creates PI0.5 feature/action archives.
3. `phase2_action_probe_materialize.py` validates pairing, applies the historical group-aware split, performs action audits, mean-pools tokens, fits four independent probes, evaluates HELD-OUT once, and freezes `W` and `P_action`.

The extraction CLI accepts `--max-observations 1` for a real-checkpoint hook and action-target smoke. Such a manifest is marked `SMOKE_COMPLETE` and is rejected by the formal materializer. Omitting the flag uses all 200 observations.

The materializer requires `--stage phase2a` or `--stage phase2b`. Phase 2A outputs are marked as candidates pending the dataset-sufficiency decision. Phase 2B uses the frozen chosen dataset and marks the resulting reusable artifacts as authoritative Phase 2 inputs. Probe hyperparameters stay fixed; HELD-OUT results are not used for tuning.

The split rule is `pilot-v0.2-c5-split-v1`: one deterministic held-out trajectory group per task, yielding 160 TRAIN and 40 HELD-OUT observations. Action normalization statistics use TRAIN only.

Each probe is `Linear(D,7,bias=False)` and uses a fixed AdamW schedule. The action-subspace operator is

\[
P_{action}=W^T(WW^T+\eta I)^{-1}W.
\]

The materializer records symmetry and ridge-idempotence residuals and verifies probe, weight, and projection serialization.

## Real-checkpoint extraction smoke status

Both model-specific extraction paths have passed a one-observation server smoke on the same frozen Pilot v0.2 sample, `libero_spatial__task00__state00__step0008`.

| Model | Code commit | Projected node | Deep node | Action | Repeat-forward maximum absolute differences | Result |
|---|---|---|---|---|---|---|
| OpenVLA | `3d362a8f24033b3dbec150b23660d2b0315ba95f` | O2 `[256,4096]` | O-deep `[256,4096]` | `[7]` | O2 `0.0`; O-deep `0.0`; action `0.0` | `SMOKE_COMPLETE` |
| PI0.5 | `9b174369eee36a4715dd9b716d5213bebec27bae` | P2 `[256,2048]` | P-deep `[256,2048]` | `[7]` | P2 `0.0`; P-deep `0.0`; action `0.0` | `SMOKE_COMPLETE` |

Both archives are float32, finite, and match the SHA-256 recorded in their manifests. Extraction ran under `torch.inference_mode()`, and neither smoke populated gradients on VLA parameters.

The PI0Pytorch path required an extraction ownership fix. Its compiled CUDA Graph runtime may reuse decoder-layer output storage on the next `infer()` call. P-deep is therefore cloned after `policy.infer()` returns and before another model invocation. This preserves the same layer-8 tensor values and base-camera token slice while preventing a later inference from overwriting the captured representation. A regression test models this reusable-buffer behavior and verifies that the first extraction remains unchanged after a repeated inference.

The synchronized evidence is stored under:

- `experiment_inbox/shared-feature-phase2/phase2-action-probe-smoke-20260919-201027/openvla/`
- `experiment_inbox/shared-feature-phase2/phase2-action-probe-pi05-smoke-fix-20260919-202246/`

This evidence closes the real-checkpoint hook, tensor-shape, action-target, repeatability, and serialization smoke gates.

## Phase 2A result

Both models subsequently completed extraction on all 200 paired observations at code commit `dabe0d4da006a2d9074d4db7d933f4f0d8fc31fa`. All 400 feature archives passed SHA-256, identity, shape, ordering, and finite-value validation. Repeat-forward maximum absolute differences remained `0.0` for both representations and actions.

The group-aware Phase 2A materialization completed with 160 TRAIN and 40 HELD-OUT observations. All four candidate probes beat their model-specific mean-action baseline on aggregate held-out normalized MSE:

| Model | Node | Held-out MSE | Mean baseline MSE | Reduction |
|---|---|---:|---:|---:|
| OpenVLA | O2 | 0.619732 | 1.109212 | 44.13% |
| OpenVLA | O-deep | 0.525864 | 1.109212 | 52.59% |
| PI0.5 | P2 | 0.666358 | 1.069136 | 37.67% |
| PI0.5 | P-deep | 0.506923 | 1.069136 | 52.59% |

No action dimension is near-constant overall or within a task. Probe optimization, serialization, projection construction, and artifact inventory checks passed. Phase 2A therefore retains the existing 200-observation dataset without immediate densification. Detailed action coverage, per-dimension metrics, limitations, and candidate artifact validation are recorded in `docs/phase2-action-predictive-representation-report.md`.

A subsequent CPU-only diagnostic refitted every node from seeds 1, 2, 3, 4, 5, and 7 under the unchanged protocol. O2, O-deep, and P2 predictions were stable, while P-deep held-out MSE had coefficient of variation `0.1139`. P2 produced a stable row space; O2 narrowly missed the projection-distance threshold; O-deep and P-deep produced materially initialization-dependent subspaces. The diagnostic status remains `NEEDS_REVIEW`; full results are recorded in `docs/phase2-action-probe-stability-report.md`.

Phase 3 subsequently made an explicit primary-node decision: freeze the exact seed-7 O2/P2 probes under the unchanged protocol, retain the O2 projection-distance caveat, and leave both deeper nodes outside the primary texture objective. The Phase 2B promotion validates and copies only O2/P2 into `phase2b-primary-action-probes-v2`; it does not relabel the complete four-node stability diagnostic as `STABLE`.

## Server-only validation boundary

Complete extraction requires the authoritative OpenVLA and PI0Pytorch checkpoints, CUDA, and the complete OpenPI runtime. That extraction and Phase 2A candidate materialization have completed. Local tests cover capture semantics with synthetic towers, split integrity, audits, normalization, probe fitting, metrics, serialization, projection construction, and multi-seed subspace diagnostics. The primary O2/P2 Phase 2B promotion is complete locally with source hashes unchanged; Phase 3 CUDA calibration and gradient closure remain server-only.
