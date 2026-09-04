# Step 0 Phase C Renderer Evidence

## Runtime audit

- Branch: `fix/openvla-baseline-correctness`
- Audited HEAD: `1b1b8cb5cf5ac43fb962c44df3abf85f8e32e12b`
- Task/state: `libero_spatial` task 0, state 0
- Resolution: 512 x 512
- Synced artifact: `renderer-alignment-v3.json`
- Artifact SHA-256: `f7b04906680b70d0475a678c0aaac9921d8f6b56aea38ea4980d9c06c0455d0f`

The artifact independently satisfies every pass condition implemented by
`scripts/audit_step0_renderer.py`:

| Contract | Required | Observed |
| --- | ---: | ---: |
| Shared instance count | 2 | 2 |
| Position offset | `[0, 0, 0]` | `[0, 0, 0]` |
| Union IoU | >= 0.95 | 0.9991582491582491 |
| Union visible recall | >= 0.95 | 0.9991582491582491 |
| Zero-delta Linf | <= 1e-6 | 0.0 |
| Loss finite | true | true |
| Image gradient norm | finite and > 0 | 0.001147657516412437 |
| Texture gradient norm | finite and > 0 | 5.399637666414492e-05 |

The resolved instances and their individual renderer-mask agreement were:

| Body | IoU | Visible recall | Reference pixels | Renderer pixels |
| --- | ---: | ---: | ---: | ---: |
| `akita_black_bowl_1_main` | 0.9993919124353907 | 0.9993919124353907 | 3289 | 3287 |
| `akita_black_bowl_2_main` | 0.998868351565447 | 0.998868351565447 | 2651 | 2648 |

## Verdict

`Phase C PASS`

This establishes the task-0/state-0 runtime contracts for multi-instance
discovery, zero renderer position offset, MuJoCo visibility, renderer-mask
alignment, exact zero-delta composition, and a finite nonzero texture-gradient
path. It is a smoke-validation result, not a scientific attack metric.
