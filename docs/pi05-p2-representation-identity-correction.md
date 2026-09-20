# PI0.5 P2 Representation Identity Correction

## Scope

This correction aligns the Phase 2 action-probe representation with current authoritative PI0Pytorch runtime semantics. It does not change OpenVLA, P-deep, the Phase 3 loss, renderer, gradient ensemble, or texture update.

## Corrected definition

```text
P2 = model.paligemma_with_expert.embed_image(base_0_rgb)
shape = [B,256,2048]
manual scaling = none
definition_id = pi05_p2_embed_image_no_manual_scaling_v2
```

Formal extraction saves the direct official `embed_image(base_0_rgb)` output. It validates that independently extracted tensor against a second direct `embed_image()` call and the base-camera slice of the official prefix path. It also compares all 200 deployed first-step actions against the historical extraction under the same deterministic per-sample diffusion noise and blocks on a difference above `1e-6`.

The first corrected CUDA smoke showed that direct `embed_image()` and the official prefix slice were bit-identical, while the projector hook inside compiled `policy.infer()` differed by relative L2 `0.006723` (maximum absolute difference `3.0`). This was not a fixed scaling: all three norms were approximately `4033`. The corrected extractor therefore uses the direct official path as its representation source instead of relaxing the identity tolerance for the compiled hook.

## Historical status

The historical Phase 2A PI0.5 P2 archives, stability run, Phase 2B v2 artifact, and initial Phase 3 smoke are preserved unchanged. Their P2 definition used an erroneous extra `1/sqrt(2048)` factor and they are superseded for Phase 3 use. This fixed scaling does not itself change theoretical information content, but it changes finite-step AdamW fitting and made the frozen probe incompatible with native runtime P2.

## Required corrected pipeline

1. Re-extract only PI0.5 P2/P-deep/actions into a fresh v2 representation materialization.
2. Fit only corrected PI0.5 P2 with the frozen seed-7 protocol.
3. Run only corrected PI0.5 P2 seeds `1,2,3,4,5,7` through the frozen stability diagnostic.
4. If status is `CORRECTED_P2_STABLE`, create Phase 2B v3 by copying OpenVLA O2 byte-identically from Phase 2B v2 and corrected PI0.5 P2 from the new materialization.
5. Re-run lambda calibration, 1-step smoke, 10-step smoke, and paired-noise probe/action consistency.

The 500-step pilot remains blocked until these gates pass. Runtime results belong in a follow-up result commit; this document does not predeclare them.
