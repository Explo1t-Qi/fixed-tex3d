# Shared-Feature Tex3D Phase 2 Source-Attack Report

**Status:** source-model pilots and final-texture gradient decomposition complete;
held-out transfer not evaluated

**Date:** 2026-09-18

**Task:** `libero_spatial`, task `0`, object `akita_black_bowl`

## Scope

This report records the completed Phase 2 differentiable mapping, loss, live
gradient closure, source-model texture optimization, paired source rollout,
and multi-level gradient-decomposition results. OpenVLA and PI0.5 both
participated in every Phase 2.4 optimization objective reported here. They are
source models, not held-out transfer targets.

The frozen scientific boundary remains:

```text
shared != vulnerable != policy-relevant != transferable
```

No PI0 held-out rollout or transfer experiment has been run.

## Implementation and provenance

| Stage | Repository commit | Result |
| --- | --- | --- |
| Phase 2.1 differentiable frozen PCA+CCA mapping | `shared-feature-tex3d@f22858a14287a7d70679817e53286d6b9a4aa42b` | PASS |
| Phase 2.2 shared canonical displacement loss | `shared-feature-tex3d@176c46c2b13e8a647a7f513ea7583cefe0ee44ba` | PASS |
| Phase 2.3 dual-VLA end-to-end gradient closure | `fixed-tex3d@09c196138de86ca30c2358ac041152c6209448af` | PASS |
| Phase 2.4 shared-CCA pilot implementation | `fixed-tex3d@830284b7da70e4c018ab58a1d9b57066e9bfa197` | PASS |
| Source evaluator checkpoint-path fix | `fixed-tex3d@f8925fb63c74cb19fb81c8e5f22172050ec8c894` | PASS |
| Schedule diagnostics | `fixed-tex3d@27ba339a8af001c8fbfa192bfd39edecbf06fc03` | PASS |
| Independently adjustable iteration/step configuration | `fixed-tex3d@00f7f49099bb6e22df737d5114e34212d357d164` | PASS |
| Native model-gradient ensemble | `fixed-tex3d@5e13b0950703ad1243b5b970f8d065eef50da4c4` | PASS |
| Multi-level native model-gradient ensemble | `fixed-tex3d@2e9daefda2679320a8c793fa202bcf0934186740` | PASS |
| Read-only multi-level gradient decomposition | `fixed-tex3d@785bae172cb8c61d51659745a9530d5f57fb4f64` | PASS |

The shared-CCA runs use mapping materialization
`phase1_o2_p2_pi05_torch_v1`, whose `mapping.npz` SHA-256 is:

```text
572d4772432025f130ecf0403562bab20a20d4bec008c778985b2b9aee28caec
```

The native-gradient-ensemble mode neither loads nor validates that mapping.

## Frozen Phase 2.4 substrate

The comparable pilot configuration is:

```text
task suite:              libero_spatial
task id:                 0
training state IDs:      0..9
frames per state:        1
effective frame pool:    10
selected frames/step:    all 10
seed:                    7
texture update:          sign-PGD
renderer epsilon:        0.5019607843137255
texture parameter:       tanh(adv_noise) * epsilon
OpenVLA device:          cuda:0
PI0.5 device:            cuda:1
```

Each outer iteration aggregates all selected frame losses or gradients and
updates the shared texture exactly once. Clean references are computed once and
detached. The renderer, visibility mask, lighting calibration, state selection,
camera semantics, and texture baking are shared across objectives.

## Phase 2.1 and 2.2 frozen objective

The frozen mapping is implemented as differentiable PyTorch buffers:

\[
H_O=((O2-\mu_O)B_O)W_O,
\qquad
H_P=((P2-\mu_P)B_P)W_P.
\]

`W_O` and `W_P` already contain the CCA whitening contribution. Runtime forward
does not multiply the separately serialized whitening arrays again.

The Phase 2.2 loss is:

\[
H_{shared}^{clean}=\frac{H_O^{clean}+H_P^{clean}}{2},
\qquad
H_{shared}^{adv}=\frac{H_O^{adv}+H_P^{adv}}{2},
\]

\[
L_{shared}
=-
\operatorname{MSE}
\left(H_{shared}^{adv},H_{shared}^{clean}\right).
\]

It uses every element of the `[B,256,262]` canonical tensors and performs no
token pooling, top-k selection, model reweighting, KL term, or cosine
regularization.

## Phase 2.3 end-to-end closure

The authoritative real dual-GPU smoke is synchronized under:

```text
experiment_inbox/shared-feature-phase2/phase2-3-smoke-04/
```

It established the same-image live graph:

```text
renderer.adv_noise
-> differentiable renderer
-> one adversarial base image
-> OpenVLA O2 on cuda:0 -> H_O
-> PI0Pytorch P2 on cuda:1 -> H_P -> cuda:0
-> shared loss
-> image gradient
-> texture gradient
-> one optimizer update
```

O2 was `[1,256,4096]`, P2 was `[1,256,2048]`, and both mapped outputs were
`[1,256,262]`. OpenVLA, PI0.5, joint-image, and renderer-texture gradients were
finite and nonzero. Clean gradients were absent. Phase 2.3 therefore passed as
an engineering gradient-closure result; it did not make an attack-effectiveness
claim.

## Shared-CCA optimization runs

All four schedule configurations completed without nonfinite loss, missing
gradient, zero gradient, or texture-budget violation.

| Run | Iterations | Step | Final shared MSE | Final O2 canonical MSE | Final P2 canonical MSE | Final displacement cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase2-4-pilot-01` | 500 | 0.05 | 0.607750 | 0.502958 | 1.755447 | 0.092534 |
| `phase2-4-step-size-001-500` | 500 | 0.01 | 0.548126 | 0.337590 | 1.742210 | 0.077404 |
| `phase2-4-iterations-5000` | 5000 | 0.05 | 0.610378 | 0.503223 | 1.765391 | 0.092473 |
| `phase2-4-step-001-iterations-5000` | 5000 | 0.01 | 0.610866 | 0.342240 | 1.986051 | 0.075436 |

The `.05/5000` run changed the final shared MSE only from `0.607750` to
`0.610378` relative to `.05/500`. Extending duration alone did not improve the
corresponding source rollout result.

The correctly identified `.01/5000` texture has SHA-256
`fafe05da427f34c171370e348c1e0472bbd49db08c384ede73edb7038407cbef`.
Only a 20-state screen (`10..29`) was run against that artifact: OpenVLA
conditional ASR was `6/17 = 35.29%`, while PI0.5 conditional ASR was `0/20`.
The directory `phase2-4-step-001-iterations-5000-openvla-eval-01` points to the
`.05/5000` texture SHA rather than the `.01/5000` texture and must not be cited
as a `.01/5000` evaluation.

## Native O2/P2 model-gradient ensemble

The native baseline does not use PCA, CCA, or a shared-feature artifact. For
each frame it uses:

\[
L_O=-\operatorname{mean}[(O2_{adv}-O2_{clean})^2],
\qquad
L_P=-\operatorname{mean}[(P2_{adv}-P2_{clean})^2].
\]

The two texture gradients remain separate. Each model first averages its own
gradients across the complete frame batch:

\[
g_O=\frac{1}{B}\sum_b \nabla_\theta L_O^{(b)},
\qquad
g_P=\frac{1}{B}\sum_b \nabla_\theta L_P^{(b)}.
\]

Model-wise normalization and ensemble are then applied:

\[
\hat g_O=\frac{g_O}{\operatorname{mean}(|g_O|)+10^{-12}},
\quad
\hat g_P=\frac{g_P}{\operatorname{mean}(|g_P|)+10^{-12}},
\quad
g_{GE}=\frac{\hat g_O+\hat g_P}{2}.
\]

The texture is updated once with `-0.05 * sign(g_GE)`. This is not equivalent
to `(L_O + L_P).backward()` and does not normalize per frame.

The 500-step run `phase2-native-ge-500-v1` completed all 500 texture updates.
O2 native MSE increased from `3.8259e-05` to `0.0584087`; P2 native MSE
increased from `0.000681664` to `0.459409`. Raw model gradients remained finite
and nonzero. Their final cosine was `-0.252427`, and the final normalized
ensemble norm was `904.934`. The texture stayed within the renderer budget.

The final texture SHA-256 is:

```text
45a2dc510b464af1dd7f63700b165fad2bcaa63b4421ab917f65aaff624f2f1d
```

## Multi-level O1/O2 and P1/P2 native ensemble

The multi-level baseline keeps the native model-gradient ensemble unchanged
and adds one earlier representation to each model. The frozen nodes are:

```text
OpenVLA O1-S: SigLIP branch output before fusion/projector  [B,256,1152]
OpenVLA O2:   multimodal projector output                  [B,256,4096]
PI0.5 P1:     final normalized vision output before head   [B,256,1152]
PI0.5 P2:     PaliGemma-ready projected visual tokens      [B,256,2048]
```

Within each model, the two level losses are added without scale normalization
or reweighting:

\[
L_O=-\left(\operatorname{MSE}_{O1}+\operatorname{MSE}_{O2}\right),
\qquad
L_P=-\left(\operatorname{MSE}_{P1}+\operatorname{MSE}_{P2}\right).
\]

The method then retains the native-GE hierarchy: per-frame model gradients,
mean across all ten frames within each model, separate model-wise mean-absolute
normalization, cross-model mean, and one sign-PGD update. It performs no
level-wise normalization and does not use PCA, CCA, or a shared-feature
artifact.

The formal run `phase2-multilevel-native-ge-500-v1` completed all 500 updates
with finite, nonzero gradients and no texture-budget violation. Its last
training row was:

| Metric | Value |
| --- | ---: |
| O1 MSE | 0.407841 |
| O2 MSE | 0.020929 |
| O1/O2 MSE ratio | 19.4866 |
| P1 MSE | 0.262342 |
| P2 MSE | 0.417361 |
| P1/P2 MSE ratio | 0.628574 |
| Raw model-gradient cosine | -0.765875 |
| Normalized ensemble L2 | 683.643 |

The final vertex-noise and baked-texture SHA-256 values are:

```text
vertex noise:  99c1d465948084c4afaa2ad841f2823db287972463cde3db39c2a8ad6a65efe2
baked texture: 2e0c9bda96c310071c35645ebff6fbb42e2ee959c3c92fdc447709d50a6df3c4
```

## Paired source-model rollout results

Every full evaluation used state IDs `0..49`, seed rule `7 + state_id` reset
before clean and adversarial rollout, `num_steps_wait=10`, and identical model
checkpoints. OpenVLA used a 300-step horizon; PI0.5 used a 400-step horizon and
`replan_steps=1`. Each summary recorded `asset_xml_restored=true`.

| Objective | Model | Clean success | Adversarial success | Adversarial failure rate | Clean-success/adv-failure | Conditional ASR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shared CCA, `.05/500` | OpenVLA | 46/50 (92%) | 34/50 (68%) | 32% | 14 | 14/46 = **30.43%** |
| Shared CCA, `.05/500` | PI0.5 | 50/50 (100%) | 49/50 (98%) | 2% | 1 | 1/50 = **2.00%** |
| Shared CCA, `.05/5000` | OpenVLA | 46/50 (92%) | 37/50 (74%) | 26% | 12 | 12/46 = **26.09%** |
| Shared CCA, `.05/5000` | PI0.5 | 50/50 (100%) | 49/50 (98%) | 2% | 1 | 1/50 = **2.00%** |
| Native GE, `.05/500` | OpenVLA | 46/50 (92%) | 28/50 (56%) | 44% | 20 | 20/46 = **43.48%** |
| Native GE, `.05/500` | PI0.5 | 50/50 (100%) | 50/50 (100%) | 0% | 0 | 0/50 = **0%** |
| Multi-level native GE, `.05/500` | OpenVLA | 46/50 (92%) | 36/50 (72%) | 28% | 13 | 13/46 = **28.26%** |
| Multi-level native GE, `.05/500` | PI0.5 | 50/50 (100%) | 50/50 (100%) | 0% | 0 | 0/50 = **0%** |

For the native texture, OpenVLA clean/adv discordance was 20 clean-success to
adversarial-failure cases versus 2 clean-failure to adversarial-success cases;
the two-sided exact McNemar value is approximately `p=0.000121`. Native GE
improved the OpenVLA conditional-ASR point estimate by `13.05` percentage points
over shared `.05/500`. In a direct paired comparison of the two adversarial
textures over the common clean-success states, the discordant counts were 9 in
favor of native GE and 3 in favor of shared CCA (`p=0.145996`). The current
50-trial sample therefore does not establish a statistically reliable
native-over-shared advantage.

The multi-level texture reduced the OpenVLA conditional-ASR point estimate
from `20/46 = 43.48%` for O2/P2-only native GE to `13/46 = 28.26%`. Across the
46 common clean-success trials, both textures caused failure on 8 states,
O2/P2-only caused failure alone on 12, multi-level caused failure alone on 5,
and neither caused failure on 21. The two-sided exact paired McNemar value is
approximately `p=0.143`. The observed direction therefore disfavors the simple
multi-level objective, but 50 trials do not establish a statistically reliable
difference between the two textures. PI0.5 succeeded in all 50 adversarial
rollouts under both native objectives.

## Final-texture gradient decomposition

The read-only diagnostic used the exact final multi-level vertex-noise artifact
and the same task-0 states `0..9`. It computed four negative-MSE gradients per
frame with `torch.autograd.grad`, averaged each raw component across all ten
frames, and performed no per-frame, level, or model normalization. The texture
parameter was bit-identical before and after the diagnostic.

| Component | Batch MSE | Gradient L2 | Gradient mean abs | Gradient Linf |
| --- | ---: | ---: | ---: | ---: |
| O1 | 0.408152 | 4.85605e-4 | 2.51331e-7 | 7.75556e-5 |
| O2 | 0.020963 | 1.12073e-4 | 5.40995e-8 | 2.01663e-5 |
| P1 | 0.262677 | 1.41632e-3 | 7.06230e-7 | 2.45447e-4 |
| P2 | 0.418050 | 2.66068e-3 | 1.29437e-6 | 5.25830e-4 |

O1/O2 MSE was `19.4697`, while O1/O2 gradient L2 and mean-absolute
ratios were `4.3329` and `4.6457`. O1 therefore genuinely dominated the
OpenVLA texture gradient, although the MSE ratio overstated the gradient-scale
imbalance. P2/P1 gradient L2 was `1.8786`.

The raw batch-gradient cosine matrix was:

| | O1 | O2 | P1 | P2 |
| --- | ---: | ---: | ---: | ---: |
| O1 | 1.000 | -0.408 | -0.757 | -0.752 |
| O2 | -0.408 | 1.000 | 0.118 | 0.104 |
| P1 | -0.757 | 0.118 | 1.000 | 0.959 |
| P2 | -0.752 | 0.104 | 0.959 | 1.000 |

Within OpenVLA, the cancellation ratio
`||g_O1+g_O2||/(||g_O1||+||g_O2||)` was `0.7557`. The reconstructed OpenVLA
gradient had cosine `0.9740` with O1 and `-0.1902` with O2; the equal-coefficient
loss was therefore O1-dominated in texture-gradient space. Within PI0.5, the
cancellation ratio was `0.9906`, so P1 and P2 reinforced rather than opposed
each other.

The reconstructed OpenVLA/PI0.5 model-gradient cosine was `-0.7909`. Its raw
cross-model dot product decomposed as:

```text
O1-P1: -5.208e-7
O1-P2: -9.715e-7
O2-P1: +1.875e-8
O2-P2: +3.089e-8
```

Thus the negative model relationship came almost entirely from O1 against both
PI0.5 levels, especially O1-P2. O2 weakly offset about `3.3%` of that negative
dot product. The diagnostic does not show P1/P2 internal conflict and therefore
does not explain PI0.5's zero conditional ASR through multi-level cancellation.

The reconstructed PI0.5 raw-gradient L2 was about `8.94` times the OpenVLA
value, but the actual trainer normalizes each complete model gradient
separately before ensembling. Raw magnitude therefore does not make PI0.5
dominate the cross-model update. Likewise, the negative cosine indicates
contested coordinate directions rather than a reduced PGD step: the subsequent
`sign` operation still applies the frozen step to every nonzero ensemble
coordinate.

The diagnostic cosine differs slightly from the last training-row cosine
`-0.7659` because the training metric is evaluated before the final PGD update,
whereas the diagnostic evaluates the persisted post-update texture. The close
values and the nearly identical reconstructed OpenVLA gradient L2 support the
runtime consistency of the offline measurement.

## Interpretation and current decision boundary

The implementation, real-model gradient closure, training execution, artifact
publication, and paired source evaluation all passed their engineering gates.

The source-model evidence is asymmetric:

- shared CCA and native GE both degrade OpenVLA policy success;
- neither produces meaningful PI0.5 policy degradation;
- larger P2 feature displacement does not imply PI0.5 policy degradation;
- model-wise gradient magnitude normalization does not make the native feature
  directions action relevant;
- adding unweighted O1/P1 supervision reduced the OpenVLA conditional-ASR point
  estimate and left PI0.5 at zero conditional ASR;
- O1 dominates the multi-level OpenVLA texture gradient and conflicts with O2;
- P1 and P2 are strongly aligned, so PI0.5 failure is not attributable to
  within-model multi-level gradient cancellation;
- the strong negative cross-model gradient relationship is driven primarily by
  O1 against P1/P2, while O2/P2 are only weakly positively aligned;
- increasing shared-CCA training from 500 to 5000 iterations did not improve
  source-model effectiveness.

Accordingly, the current dual-source attack hypothesis is not supported. The
native-gradient experiment is a useful partial positive result for OpenVLA and a
negative result for PI0.5. It does not show transferability because both evaluated
models participated in optimization. No final claim about the broader
shared-feature transfer hypothesis can be made until a model excluded from
optimization is evaluated under a separately frozen protocol.

The simple equal-coefficient multi-level objective is not supported as an
improvement over O2/P2-only native GE under this frozen protocol. This is a
method-level negative result, not evidence that all multi-level objectives must
fail. The decomposition is local to the final texture and uses batch-mean
gradients, so it does not establish when the conflict emerged or whether every
state has the same relationship. The next diagnostic can apply the same
read-only decomposition to the saved step-100 through step-500 artifacts. Any
replacement objective still requires a separately frozen design and must not be
inferred directly from cosine values alone.

## Synchronized evidence

Repository-relative to `fixed-tex3d`:

```text
experiment_inbox/shared-feature-phase2/phase2-3-smoke-04/
experiment_inbox/shared-feature-phase2/phase2-4-pilot-01/
experiment_inbox/shared-feature-phase2/phase2-4-openvla-source-eval-02/
experiment_inbox/shared-feature-phase2/phase2-4-pi05-source-eval-01/
experiment_inbox/shared-feature-phase2/phase2-4-iterations-5000/
experiment_inbox/shared-feature-phase2/phase2-4-iterations-5000-openvla-eval-01/
experiment_inbox/shared-feature-phase2/phase2-4-iterations-5000-pi05-eval-01/
experiment_inbox/shared-feature-phase2/phase2-4-step-001-iterations-5000/
experiment_inbox/shared-feature-phase2/phase2-4-step-001-iterations-5000-openvla-screen20/
experiment_inbox/shared-feature-phase2/phase2-4-step-001-iterations-5000-pi05-screen20/
experiment_inbox/shared-feature-phase2/phase2-native-ge-500-v1/
experiment_inbox/shared-feature-phase2/phase2-native-ge-500-v1-openvla-eval-01/
experiment_inbox/shared-feature-phase2/phase2-native-ge-500-v1-pi05-eval-01/
experiment_inbox/shared-feature-phase2/phase2-multilevel-native-ge-500-v1/
experiment_inbox/shared-feature-phase2/phase2-multilevel-native-ge-500-v1-openvla-eval-01/
experiment_inbox/shared-feature-phase2/phase2-multilevel-native-ge-500-v1-pi05-eval-01/
experiment_inbox/shared-feature-phase2/phase2-multilevel-gradient-diagnostic-final-20260918-143739/
```
