# Expanded Phase 2 → Exploratory Phase 3 Handoff

## Decision

Expanded Pilot v0.3 is the current Phase 2 dataset authority:

```text
395 accepted clean OpenVLA trajectory groups
2370 paired frozen observations
319 TRAIN groups / 1914 observations
76 HELD-OUT groups / 456 observations
split = pilot-v0.3-expanded-split-v1
```

The primary seed-7 probes are action-predictive:

| Model | Node | Shape | TRAIN MSE | HELD-OUT MSE | Baseline MSE |
|---|---|---:|---:|---:|---:|
| OpenVLA | O2 | `[256,4096]` | 0.13429 | 0.29370 | 0.99444 |
| PI0.5 | corrected P2 | `[256,2048]` | 0.23880 | 0.32206 | 1.00244 |

Corrected P2 is exactly
`paligemma_with_expert.embed_image(base_0_rgb)` with no manual
`1/sqrt(2048)` scaling. It has positive held-out R² in every action coordinate,
rank-7 W, and exact serialization/reload.

The six-seed corrected-P2 gate is nevertheless
`CORRECTED_P2_NEEDS_REVIEW`:

```text
held-out MSE CV = 0.05564      (threshold <= 0.05)
minimum principal cosine = 0.65546  (threshold >= 0.90)
maximum projection distance = 0.61612 (threshold <= 0.25)
```

## Authority separation

```text
Phase 2 action-predictive signal:            PASS / strongly supported
Seed-7 probe usability:                      PASS as a fixed candidate artifact
Cross-seed full-subspace reproducibility:    FAIL / unresolved
Authoritative Phase 2B v3:                   NOT FROZEN / BLOCKED
Exploratory Phase 3 pipeline feasibility:    AUTHORIZED, fixed seed-7 only
```

The provisional authority is:

```text
phase2-expanded-seed7-provisional-action-probes-v1
```

It contains byte-identical copies of the expanded seed-7 O2/P2 `W`, probe,
projection, TRAIN-only action statistics, metrics, and training histories,
together with the exact split, representation-manifest hashes, collection
provenance, source inventories, and stability provenance. Its metadata must
state `PHASE2_EXPANDED_SEED7_PROVISIONAL_FROZEN`; it is neither Phase 2B v3 nor
evidence that the full action-predictive subspace is seed-reproducible.

## Permitted Phase 3 sequence

```text
fixed provisional seed-7 O2/P2 probes
→ fresh lambda_dir recalibration
→ corrected 1-step smoke
→ review
→ corrected 10-step smoke
→ probe-vs-actual-action consistency diagnostic
→ exploratory full optimization pilot
→ source-policy rollout evaluation
```

Historical lambda values from the superseded scaled-P2 artifact are invalid. No
step may skip calibration or the preceding smoke review. A successful exploratory
run is evidence about pipeline feasibility only, not held-out transfer, causal
action relevance, unique subspace identification, or general seed robustness.
