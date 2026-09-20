# Phase 2 — Action-Predictive Representation Runbook

This stage measures whether model-specific deployed 7-D actions are linearly readable from frozen representation nodes. Historical Pilot v0.2 evaluated four nodes; the expanded primary decision evaluates only OpenVLA O2 and corrected PI0.5 P2. It does not test causal action relevance, texture effectiveness, policy degradation, or transferability.

> **PI0.5 P2 identity correction (2026-09-20).** The historical Phase 2A PI0.5 extractor divided the projector hook output by `sqrt(2048)` even though the current PI0Pytorch `embed_image()` returns `get_image_features()` directly. Historical PI0.5 P2 probe/stability artifacts and `phase2b-primary-action-probes-v2` are preserved for provenance but are superseded and invalid as Phase 3 runtime inputs. The corrected v2 representation manifest requires exact extractor/direct-`embed_image`/prefix identity and no manual scaling.

The corrected 200-observation representation extraction completed and passed
exact P2 identity, action identity, determinism, archive-shape, and finite-value
checks. Its manifest is
`experiment_inbox/shared-feature-phase2/phase2-pi05-p2-identity-correction-v1/pi05-representations/representation_manifest.json`
(SHA-256 `40c2b489b537f24b06c098398f82e89791cb6054797f5e56f48573a78cd2002f`).
Its seed-7 corrected P2 probe reached held-out normalized MSE `0.99361` versus
the `1.06914` mean baseline. Six-seed prediction CV passed at `0.01425`, but the
minimum principal cosine (`0.74476`) and maximum projection distance (`0.60848`)
failed their unchanged stability thresholds. The status is
`CORRECTED_P2_NEEDS_REVIEW`. Read-only diagnosis localized most fitted-weight
variation to the 160-sample TRAIN feature matrix nullspace. Those 200-observation
probe/stability results remain provenance, not Phase 2B v3 inputs.

## Current Pilot v0.3 expanded protocol

Phase 2 now tests whether genuine independent-data coverage resolves the P2
subspace-identification problem without changing the probe solver or thresholds.
Pilot v0.3 is a new protocol and schema; it does not relabel or overwrite Pilot
v0.2 artifacts.

- Collection schema: `pilot_v0_3_expanded_collection_v1`.
- Protocol identity: `pilot-v0.3-expanded-v1`.
- Split rule: `pilot-v0.3-expanded-split-v1`.
- Group identity: `(task_id, initial_state_id)`.
- Candidate states: every official initial state in ascending order; no repeated
  state may count as another group.
- Acceptance: successful clean OpenVLA on-policy trajectories only.
- Observation progress: `0.10, 0.25, 0.40, 0.55, 0.70, 0.90`, using
  `floor(q*(T-1)+0.5)`. A trajectory is rejected from the accepted dataset if
  these targets collide on a timestep.

The local authoritative LIBERO checkout exposes 50 fixed initial states for
each of the ten LIBERO-Spatial tasks. The server capacity audit must repeat the
same `get_task_init_states` call against `/data/xiaomengqi/src/LIBERO-joint`
before collection. Available capacity is therefore at most 500 unique groups
and 3,000 observations; actual capacity is the number of successful,
collision-free trajectories and is never padded by duplicated states.

The collection manifest and independently cross-validated
`feasibility_report.json` record per task: available and attempted states,
successful clean trajectories, policy failures, rejected states, accepted
unique groups, and exact samples. Each observation archive has a recorded
SHA-256, and the manifest records the exact Tex3D, shared-feature, and LIBERO
commits. Collection attempts every available state.

Collection execution is state-checkpointed in `collection_progress.json`.
Without `--resume`, the output directory must be fresh. With explicit
`--resume`, the collector first verifies the protocol, commits, checkpoint,
LIBERO revision, canonical state prefix, sample identities, observation files,
and SHA-256 values. Only verified completed states are skipped; an interrupted
group is removed and rerun as a unit. The state-granularity progress display
uses the capacity discovered from the live suite and reports task/state,
accepted groups, policy failures, sampling rejects, elapsed time, rate, and
ETA. Resume counters live under `execution` and do not change the scientific
dataset identity or protocol.

Formal fitting stops if any task has fewer than two accepted groups. Otherwise,
the split deterministically assigns `floor(0.20 * accepted_groups)` groups per
task to HELD-OUT, with a minimum of one and at least one TRAIN group. With 50
accepted groups per task this is exactly 40 TRAIN / 10 HELD-OUT groups per task,
or 2,400 / 600 observations.

## Frozen representation contract

| Model | Node | Definition | Shape per observation |
|---|---|---|---:|
| OpenVLA | O2 | Multimodal projector output before Llama | `[256,4096]` |
| OpenVLA | O-deep | Output of the last block in the first half of the 32-layer Llama tower; zero-based layer 15; visual slice `[1:257]` | `[256,4096]` |
| PI0.5 | P2 | Current PI0Pytorch `paligemma_with_expert.embed_image(base_0_rgb)` output; no additional manual scaling | `[256,2048]` |
| PI0.5 | P-deep | Output of the last block in the first half of the 18-layer PaliGemma prefix tower; zero-based layer 8; base-camera slice `[0:256]` | `[256,2048]` |

The runtime derives the midpoint from the actual decoder depth and also checks the expected authoritative depths of 32 and 18. O2/O-deep are captured during one OpenVLA action generation. P2/P-deep are captured during one PI0Pytorch action inference.

OpenVLA targets are the decoded action after the existing LIBERO gripper binarization and inversion. PI0.5 targets are the first deployed 7-D action from its action chunk. PI0.5 uses deterministic per-sample diffusion noise derived from the global seed and sample identity.

## Expanded Phase 2 pipeline

1. `phase2_expanded_collection.py audit` records the actual per-task initial-state capacity.
2. `phase2_expanded_collection.py collect` attempts all states and creates the expanded paired raw-observation dataset plus feasibility report.
3. `phase2_action_representation_extract.py --model openvla` creates O2/action archives from the frozen observations.
4. `phase2_action_representation_extract.py --model pi05` creates corrected P2/action archives from exactly the same observations and deterministic per-sample diffusion-noise contract.
5. `phase2_action_probe_materialize.py --stage phase2a --nodes openvla/o2 pi05/p2` validates hashes, ordering, raw-observation pairing, P2 identity, action identity, and determinism before freezing the split, auditing actions, and fitting only the two primary probes.
6. `phase2_action_probe_stability.py --nodes pi05/p2 --seeds 1 2 3 4 5 7` reruns the unchanged corrected-P2 stability gate.

The extraction CLI accepts a positive `--max-observations N` only for a prefix
smoke. Such a manifest is marked `SMOKE_COMPLETE` and is rejected by the formal
materializer. Omitting the flag consumes the complete collection count recorded
by the manifest; it is no longer fixed at 200.

The materializer requires `--stage phase2a` or `--stage phase2b`. Phase 2A outputs are marked as candidates pending the dataset-sufficiency decision. Phase 2B uses the frozen chosen dataset and marks the resulting reusable artifacts as authoritative Phase 2 inputs. Probe hyperparameters stay fixed; HELD-OUT results are not used for tuning.

Pilot v0.2 continues to use `pilot-v0.2-c5-split-v1` and reproduces its 160/40
observation split. Pilot v0.3 uses the expanded rule above. Both are
trajectory-group aware, deterministic, per-task stratified, and compute action
normalization from TRAIN only. The split is materialized before any probe metric
is computed; HELD-OUT is not used for hyperparameter tuning.

Each probe is `Linear(D,7,bias=False)` and uses a fixed AdamW schedule. The action-subspace operator is

\[
P_{action}=W^T(WW^T+\eta I)^{-1}W.
\]

The materializer records symmetry and ridge-idempotence residuals and verifies probe, weight, and projection serialization.
It also records TRAIN/HELD-OUT observation and group counts, per-task group
counts, feature-matrix rank and effective condition number, nullspace dimension,
train/held-out MSE and gap, baseline MSE, per-action R², W rank, and the fraction
of W norm in the TRAIN feature nullspace. Stability outputs retain all signed
corresponding action-row cosines, all principal-angle cosines, projection
distances, six-seed MSE CV, and per-seed nullspace fractions.

Phase 2B v3 freeze and all Phase 3 commands are blocked until the user reviews
the expanded corrected-P2 stability result. Stability thresholds and frozen
probe hyperparameters remain unchanged.

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

## Historical Pilot v0.2 Phase 2A result

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

Phase 3 subsequently made an explicit primary-node decision: freeze the exact seed-7 O2/P2 probes under the unchanged protocol, retain the O2 projection-distance caveat, and leave both deeper nodes outside the primary texture objective. That historical promotion created `phase2b-primary-action-probes-v2`. Its OpenVLA O2 artifact remains valid for historical provenance; its PI0.5 P2 artifact is superseded by the identity correction above. Neither is an expanded-dataset result. A new v3 authority may be frozen only after user review of the expanded seed-7 probes and expanded corrected-P2 stability result.

## Server-only validation boundary

Complete extraction requires the authoritative OpenVLA and PI0Pytorch checkpoints, CUDA, and the complete OpenPI runtime. Historical Phase 2A extraction/materialization and corrected PI0.5 representation extraction have completed. Local tests cover collection-schema validation, configurable sampling, variable group counts, split integrity, audits, normalization, probe fitting, metrics, serialization, projection construction, and multi-seed subspace diagnostics. Historical Phase 2B v2 is preserved but is invalid as a corrected Phase 3 input. Expanded collection, expanded O2/P2 extraction, expanded seed-7 probes, and expanded corrected-P2 stability remain server-only work. Phase 2B v3 and every Phase 3 operation remain paused pending review of those results.
