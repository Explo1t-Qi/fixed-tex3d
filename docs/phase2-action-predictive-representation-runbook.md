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

## Server-only validation boundary

Real extraction requires the authoritative OpenVLA and PI0Pytorch checkpoints, CUDA, and the complete OpenPI runtime. Local tests cover capture semantics with synthetic towers, split integrity, audits, normalization, probe fitting, metrics, serialization, and projection construction. The generated scientific report is authoritative only after both 200-observation extraction manifests and formal materialization complete on the server.
