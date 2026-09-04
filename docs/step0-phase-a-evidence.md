# Step 0 Phase A processor evidence

## Environment

The authoritative server run used the OpenVLA dependency versions required by
the reference repository:

```text
torch        2.2.0+cu121
torchvision  0.17.0+cu121
transformers 4.40.1
tokenizers   0.19.1
checkpoint   /data/huangsimin/openvla-7b-finetuned-libero-spatial
```

Resolved preprocessing configuration:

```text
branch 0  vit_large_patch14_reg4_dinov2.lvd142m
branch 1  vit_so400m_patch14_siglip_224
resize    bicubic, bicubic
antialias true, true
shape     [1, 6, 224, 224]
```

## Initial pure differentiable result

The initial checkpoint-derived differentiable processor did not satisfy the
real clean-equivalence gate. On `libero_spatial` task 0, state 0:

```text
official repeat token Hamming       0/7
surrogate-vs-official token Hamming 3/7
float32 MAE                         3.6798898150891546e-08
float32 Linf                        4.76837158203125e-07
bfloat16 unequal elements           1022
decoded action L2                   0.1252588910065189
decoded action Linf                 0.10997899139628675
```

Evidence SHA-256:

```text
2d842febeb52d7b561e125e780e6616c39e5e3608ca5b0f58f7fb5ee757c39e8
```

This failure triggered the user-authorized preprocessing-only fallback. It was
not treated as an acceptable numerical discrepancy.

## Exact-forward/surrogate-backward result

The fallback evaluates the checkpoint's official image processor on a detached
deployment uint8/PIL image for forward and combines it with the unchanged
differentiable processor as:

```text
exact + (surrogate - surrogate.detach())
```

The exact branch is evaluated under `torch.no_grad()`. No attack objective,
renderer objective, or other gradient boundary was changed.

All four processor audit cases produced exact candidate forward values:

| Case | Candidate MAE | Candidate Linf | BF16 unequal | Token Hamming |
|---|---:|---:|---:|---:|
| gradient | 0 | 0 | 0 | 0/7 |
| checkerboard | 0 | 0 | 0 | 0/7 |
| seeded noise | 0 | 0 | 0 | 0/7 |
| LIBERO Spatial task 0/state 0 | 0 | 0 | 0 | 0/7 |

Real-frame gates:

```text
official repeat token Hamming  0/7
decoded action L2              0
decoded action Linf            0
backward gradient finite       true
backward gradient norm         0.011945548467338085
audit exit code                0
```

The JSON retains the original pure-surrogate errors separately from the exact
candidate fields. Evidence SHA-256:

```text
f4c609367b8b0db063453bd2b60b109823d8e418b51f570b3fed9146371e5caa
```

## Verdict

```text
PHASE A PASS
```

This verdict authorizes proceeding to the existing Phase B/C runtime gates. It
does not by itself make the complete Step 0 baseline ready.
