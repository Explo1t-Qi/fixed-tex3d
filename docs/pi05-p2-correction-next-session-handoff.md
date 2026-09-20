# PI0.5 P2 Correction — Next-Session Handoff

## Verified state

Repository state at documentation update:

```text
branch: feat/fix-pi05-p2-representation-identity
HEAD:   8f1b51cc86a38f9565daa7c7e3dbb0b7f7c13869
```

The corrected PI0.5 representation extraction is complete for all 200 Pilot
v0.2 observations. The authoritative synchronized manifest is:

```text
experiment_inbox/shared-feature-phase2/
  phase2-pi05-p2-identity-correction-v1/
    pi05-representations/representation_manifest.json
```

Verified facts:

- schema: `phase2_action_representation_manifest_v2`;
- materialization: `pi05_p2_runtime_identity_v2`;
- manifest SHA-256: `40c2b489b537f24b06c098398f82e89791cb6054797f5e56f48573a78cd2002f`;
- 200 unique, ordered, finite feature/action archives;
- corrected P2 definition: native `embed_image(base_0_rgb)`, no manual scaling;
- corrected extractor, direct `embed_image()`, and official prefix slice are bit-identical on the identity check;
- all three reference norms: `4033.727294921875`;
- 200/200 deployed actions match the historical targets under paired deterministic noise;
- maximum action difference: `0.0`;
- repeated projected, deep, and action outputs have maximum difference `0.0`.

The synchronized artifact tree and 31 relevant unit tests were revalidated
locally after synchronization.

## Outstanding gates

Continue in this order and stop at the first failed gate:

1. Materialize the corrected PI0.5 P2 seed-7 probe with the frozen protocol.
2. Run corrected PI0.5 P2 stability for seeds `1,2,3,4,5,7`.
3. Require status `CORRECTED_P2_STABLE` without changing hyperparameters.
4. Create `phase2b-primary-action-probes-v3`, preserving OpenVLA O2 bytes from v2 and using corrected PI0.5 P2.
5. Verify v3 hashes and provenance.
6. Re-run Phase 3 lambda calibration.
7. Run corrected 1-step smoke.
8. Run corrected 10-step smoke.
9. Run corrected paired-noise probe-vs-actual-action consistency diagnostic.
10. Decide `GO_FOR_500_STEP` or `BLOCKED` from those results.

The 500-step experiment remains blocked. No corrected Phase 3 result should use
`phase2b-primary-action-probes-v2`.

## Server paths confirmed by the operator

```text
repository:
  /data/xiaomengqi/src/tex3d-fixed

OpenPI:
  /data/xiaomengqi/src/openpi

shared-feature repository:
  /data/xiaomengqi/src/shared-feature-tex3d

joint Python:
  /data/xiaomengqi/src/shared-feature-tex3d/.venv-joint/bin/python

OpenVLA checkpoint:
  /data/huangsimin/openvla-7b-finetuned-libero-spatial

PI0.5 PyTorch checkpoint:
  /data/xiaomengqi/checkpoints/pi05_libero_pytorch

LIBERO:
  /data/xiaomengqi/src/LIBERO-joint

Phase 2 log root:
  /data/xiaomengqi/logs/shared-feature-phase2

Phase 3 log root:
  /data/xiaomengqi/logs/shared-feature-phase3
```

The following path was an incorrect assumption and must not be reused:

```text
/data/xiaomengqi/checkpoints/openvla-7b-finetuned-libero-spatial
```

The server's `openvla-7b-oft-finetuned-libero-spatial` checkpoint is a separate
OFT model, not the authoritative OpenVLA checkpoint for this protocol.

At handoff time, the server Phase 3 log root contains historical Phase 2B v2 and
historical/attempted smoke directories, but no
`phase2b-primary-action-probes-v3` directory. Before every future command, check
each referenced input with `test -e` or `realpath`; do not infer paths from local
`experiment_inbox` layout.

## Artifact field contracts

Corrected Phase 2 representation manifest:

```text
pi05_p2_identity
action_identity.compared_observations
```

Corrected Phase 3 training config, once produced:

```text
action_predictive_gradient_ensemble.pi05_p2_runtime_identity
```

`representation_identity` and `action_identity.compared_samples` are not valid
keys. A previous inline validation snippet used those names and raised a
`KeyError`; that error did not invalidate the corrected representation archive.

## Scientific boundary

The completed correction establishes representation identity and unchanged
action targets. It does not yet establish corrected P2 probe predictability,
corrected subspace stability, a reusable v3 probe authority, Phase 3 gradient
closure, decoded-action consistency, texture effectiveness, or policy impact.
