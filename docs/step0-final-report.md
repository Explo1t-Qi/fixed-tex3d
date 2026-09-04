# Step 0 Baseline Correctness Final Report

## Scope and Git provenance

```text
repository  /home/xmq/src/tex3d
base        bfd0726cad6af4514251af3a717736cb8746ecf6
branch      fix/openvla-baseline-correctness
audited code HEAD ff9fcbc8f84debb7693027a58490a0d856c8cdde
```

The only non-baseline gradient approximation introduced is the explicitly
user-authorized OpenVLA image-preprocessing boundary described in the Phase A
evidence. The attack action/feature objective, loss weights, renderer objective,
and Step 1 remain unchanged. No O2/P2, Pi 0.5 loading, spectral method,
dual-view objective, gradient protection, or renderer BPDA was introduced.

## Thirteen-bug audit

| Bug | Baseline audit/reproduction evidence | Minimal repair and regression evidence | Status |
| --- | --- | --- | --- |
| 1. Generation input/mask alignment | Baseline delegated inputs to `predict_action` without jointly extending an existing attention mask when the trailing empty token was absent. A mismatched synthetic input reproduced unequal sequence lengths. | `ensure_trailing_empty_token` extends IDs and mask together and rejects pre-existing misalignment. Three model-input tests pass. | `REPRODUCED_AND_FIXED` |
| 2. Fused vision branch order | Baseline concatenated SigLIP before DINOv2, while the real checkpoint reports DINOv2 then SigLIP. | The processor specification follows `model.config.timm_model_ids` and checkpoint processor branch position. Branch-order tests and real checkpoint audit pass. | `REPRODUCED_AND_FIXED` |
| 3. Resize semantics | Baseline attack preprocessing used bilinear `F.interpolate`; the checkpoint processor uses bicubic resize with antialiasing. | Resize mode and antialias settings are derived from the processor. Resize regression and oracle audit pass. | `REPRODUCED_AND_FIXED` |
| 4. Hard-coded preprocessing | Baseline hard-coded branch order, sizes, and SigLIP/DINO mean/std values in the training function. | One checkpoint-derived preprocessing specification supplies order, size, interpolation, antialias, mean, and std. Configuration tests pass. | `REPRODUCED_AND_FIXED` |
| 5. Different clean/attack visual paths | Baseline generated clean labels with the official processor but computed clean hidden states and adversarial logits with the hand-built six-channel tensor. | Clean reference and adversarial forward now share one preprocessing boundary; the official processor remains an independent oracle. The first pure differentiable attempt produced a real-frame `3/7` mismatch and was rejected. The authorized exact-forward/surrogate-backward fallback gives exact pixels, `0/7` mismatch, identical decoded action, and finite nonzero gradient. | `REPRODUCED_AND_FIXED` |
| 6. Policy input coupled to replay resolution | Baseline rollout derived policy input from the configurable video-resolution frame. | A fixed 512 policy source and 224 pre-crop canvas are independent of replay resolution. Policy-view tests and runtime frame contract pass. | `REPRODUCED_AND_FIXED` |
| 7. Missing deployment center crop | Baseline differentiable attack path resized directly to 224 while rollout applied the deployment crop separately. | Training and rollout share the same crop-area-0.9 deployment specification. Unit tests and runtime frame contract pass. | `REPRODUCED_AND_FIXED` |
| 8. Renderer position offset | Baseline defaulted to `[0.02, 0.01, 0.025]`. | Default offset is exactly zero. Unit test passes; real renderer union IoU/recall are both `0.9991582491582491`. | `REPRODUCED_AND_FIXED` |
| 9. Evaluation re-rendered with nvdiffrast | Baseline live test baked/installed a texture but still hid the MuJoCo target and substituted an nvdiffrast foreground. | Differentiable rendering is restricted to training. Evaluation installs the active texture and reads the MuJoCo camera observation directly. One-episode smoke completes. | `REPRODUCED_AND_FIXED` |
| 10. Policy/video provenance mismatch | Baseline live path could feed a composited renderer image to policy while recording a different MuJoCo image. | Policy and replay views derive from the same MuJoCo observation, with resolution changes only after the common source. Unit test and 512-resolution rollout video pass. | `REPRODUCED_AND_FIXED` |
| 11. Shared-texture multi-instance | Baseline target lookup stopped at the first body, although Spatial task 0 contains two bowls using the same texture asset. | All bodies connected through compiled texture -> material -> geom relations are returned and share one parameter. Native MuJoCo, robosuite-wrapper, prefixed-asset, and two-instance tests pass; runtime finds two instances. | `REPRODUCED_AND_FIXED` |
| 12. Missing front-most MuJoCo visibility | Baseline removed the MuJoCo foreground and replaced it using only the renderer mask, which lacks scene occlusion. | Composition applies only the renderer color delta under mutually exclusive front-most MuJoCo instance masks. Zero delta is bit-exact, order invariant, and differentiable in tests; runtime zero-delta Linf is 0. | `REPRODUCED_AND_FIXED` |
| 13. Runtime asset name guessing | Baseline selected texture/material nodes using guessed object-name substrings and could overwrite the real texture file. | Binding resolves `texture.file` and material-to-texture relations, with name fallback only last; activation points XML to a separate generated PNG and restores XML bytes. Three asset tests and post-run cleanup pass. | `REPRODUCED_AND_FIXED` |

## Numerical runtime evidence

### Processor equivalence

Authoritative environment versions:

```text
torch        2.2.0+cu121
torchvision  0.17.0+cu121
transformers 4.40.1
tokenizers   0.19.1
```

The initial pure differentiable preprocessing was not accepted:

```text
real clean-frame action-token mismatch 3/7
decoded action L2                      0.1252588910065189
decoded action Linf                    0.10997899139628675
```

After the user-authorized preprocessing-only exact-forward/surrogate-backward
fallback:

```text
official repeat token Hamming      0/7
candidate-exact token Hamming      0/7
decoded action L2/Linf             0 / 0
pixel MAE/Linf                     0 / 0
BF16 unequal count                 0
backward gradient finite           true
backward gradient norm             0.011945548467338085
processor audit exit code          0
```

### Deployment view

```text
policy source              512 x 512
pre-crop canvas            224 x 224
deployment crop area       0.9
replay resolution coupling absent
runtime shared instances   2
```

### Renderer, visibility, and compositor

```text
position offset            [0, 0, 0]
instance count             2
union IoU                  0.9991582491582491
union visible recall       0.9991582491582491
zero-delta Linf            0
image gradient norm        0.001147657516412437
texture gradient norm      5.399637666414492e-05
```

### One-step and rollout

The authoritative pure OpenVLA run reported:

```text
action loss                30.012054443359375
feature loss               0
total loss                 30.012054443359375
texture gradient norm      0.21750734746456146
parameter before/after     0 / 0.05000000074505806 Linf
parameter change Linf      0.05000000074505806
maximum perturbation       0.025077147409319878
rollout                    complete (1 episode)
video                      77 frames, 512 x 512, 30 FPS
```

The joint OpenPI/OpenVLA environment also completed the same smoke, but is kept
as compatibility-only evidence because it reports mismatched OpenVLA dependency
versions. Its different action loss (`28.125`) is not substituted for the
authoritative result.

After the final runtime smoke, server `git status --short` and the backup-file
search were both empty at source HEAD
`ff9fcbc8f84debb7693027a58490a0d856c8cdde`.

## Tests actually executed

```text
/home/xmq/.virtualenvs/modified-tex3d/bin/pytest -q tests/unit
30 passed

scripts/audit_step0_processor.py (real checkpoint, OpenVLA environment)
exit code 0; all exact equivalence and gradient gates passed

scripts/audit_step0_renderer.py (LIBERO task 0/state 0, CUDA/EGL)
artifact satisfies every scripted pass predicate

attack_openvla.py (one update + one rollout, OpenVLA environment)
completed normally; artifacts, update, rollout, and final restoration present

attack_openvla.py (one update + one rollout, joint environment)
completed normally as supplementary compatibility smoke
```

The complete commands are maintained in `docs/step0-server-validation.md`.

## Changed files and responsibilities

| Files | Responsibility |
| --- | --- |
| `AGENTS.md`, `task.md` | Persistent repository rules and the complete Step 0 specification. |
| `.gitignore` | Excludes the server-result synchronization inbox. |
| `openvla_model_inputs.py`, `openvla_utils.py` | Generation token/mask alignment and shared deployment-view use. |
| `openvla_image_transform.py` | Checkpoint-derived differentiable processor and authorized exact-forward/surrogate-backward boundary. |
| `openvla_policy_view.py` | Fixed policy source, pre-crop canvas, center crop, and replay derivation contracts. |
| `openvla_renderer_contracts.py` | Multi-instance discovery, native/wrapped MuJoCo compatibility, visibility masks, and delta compositor. |
| `openvla_runtime_assets.py` | Relationship-based texture activation and recoverable XML restoration. |
| `libero/attack_openvla.py` | Integrates the corrected contracts into training and evaluation, emits audit artifacts, and restores assets; also fixes direct-entrypoint import ordering. |
| `scripts/audit_step0_processor.py` | Synthetic and real-checkpoint processor/action/gradient oracle audit. |
| `scripts/audit_step0_renderer.py` | Real LIBERO renderer, visibility, zero-delta, and gradient audit. |
| `tests/unit/` | Thirty CPU unit/regression tests covering all repaired contracts and entrypoint bootstrap. |
| `docs/step0-*.md` | Reproducible server commands and tracked numerical evidence. |

## Artifact locations

The synchronized, ignored copies are under:

```text
experiment_inbox/tex3d-step0-evidence/processor-equivalence-exact-forward.json
experiment_inbox/tex3d-step0-evidence/renderer-alignment-v3.json
experiment_inbox/tex3d-step0-evidence/one-step/
experiment_inbox/tex3d-step0-evidence/post-run-cleanup.txt
```

Tracked summaries and hashes are in:

```text
docs/step0-phase-a-evidence.md
docs/step0-phase-c-evidence.md
docs/step0-phase-de-evidence.md
```

## Remaining boundaries

No formal attack-effectiveness experiment or broad multi-task evaluation was
performed. Step 1 was not run. These are intentionally outside Step 0 and do
not block baseline readiness.

## Final verdict

```text
STEP 0 BASELINE READY
```
