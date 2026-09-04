# Step 1 — OpenVLA O2 单代理纹理优化与 π0.5 P2 跨模型响应 MVP

## 0. 任务定位

本任务是 Shared-Feature Tex3D 研究路线中的第一个正式跨模型实验。

核心科学问题：

> 仅使用 OpenVLA 的 O2 representation displacement 优化 Tex3D 纹理后，该纹理在未参与优化的 π0.5 上，是否会引起 P2 representation displacement，并表现出跨状态、跨 token 的一致响应趋势？

本任务只验证最基础的：

```text
source-model representation vulnerability
→ held-out model representation response
```

暂时不证明：

```text
shared vulnerable direction
action relevance
transferable vulnerability mechanism
better rollout attack success
```

必须始终遵守：

```text
shared ≠ vulnerable ≠ action-relevant ≠ transferable
```

---

# 1. Git 与代码基线

## 1.1 工作仓库

正式实现仓库：

```text
Explo1t-Qi/fixed-tex3d
```

服务器工作目录：

```text
/home/xmq/src/tex3d
```

Step 0 分支：

```text
fix/openvla-baseline-correctness
```

Step 0 最终文档 HEAD：

```text
6ee46c18c78981cfa86b45edaab20c11db030483
```

Step 0 实际 runtime audited code HEAD：

```text
ff9fcbc8f84debb7693027a58490a0d856c8cdde
```

新建 Step 1 分支，例如：

```text
feat/step1-o2-p2-transfer-mvp
```

必须从已经完成 Step 0 的分支 HEAD 创建。

不得重新从原始：

```text
bfd0726cad6af4514251af3a717736cb8746ecf6
```

开发。

---

## 1.2 Step 0 是冻结前置条件

以下 Step 0 修复均视为当前实验基础设施，不允许在 Step 1 中回退：

- OpenVLA generation token / attention-mask 对齐
- DINOv2 → SigLIP checkpoint branch order
- checkpoint-derived preprocessing
- bicubic + antialias resize
- OpenVLA exact-forward / surrogate-backward preprocessing boundary
- clean / adversarial 统一视觉输入路径
- 512×512 Policy Source
- 224×224 Pre-Crop Canvas
- crop area = 0.9 的 Deployment Effective View
- renderer position offset = 0
- shared-texture multi-instance
- MuJoCo front-most visibility
- visibility-masked renderer delta compositor
- Active Texture 正式 MuJoCo evaluation
- XML / texture asset transaction 与恢复

特别注意：

第一次纯 PyTorch differentiable preprocessing 已经在真实 LIBERO frame 上出现：

```text
3 / 7 action-token mismatch
```

因此当前 Step 0 已正式采用：

```text
official exact forward
+
differentiable surrogate backward
```

Step 1 不重新讨论或删除这一 boundary。

---

# 2. 外部只读参考

以下仓库/文档仅允许作为接口和语义参考：

```text
/home/xmq/src/shared-feature-tex3d
```

重点查找已经验证过的：

```text
OpenVLA O2 extractor
π0.5 P2 extractor
OpenVLA continuation interface
π0.5 observation / preprocessing adapter
```

不得重新定义 O2/P2。

如果需要迁移少量 extractor 代码到 `fixed-tex3d`，应：

1. 保持节点语义完全一致；
2. 增加 shape regression test；
3. 明确记录代码来源和迁移原因；
4. 不把 `shared-feature-tex3d` 改造成 Step 1 的攻击训练仓库。

`modified-tex3d` 仅允许作为历史参考，不复制其 spectral、dual-view、gradient protection、TAAO 等复杂实现。

---

# 3. 冻结 Representation Node

## OpenVLA

主节点：

```text
O2
```

定义：

```text
multimodal projector output
before deeper Llama processing
```

预期单图 shape：

```text
[256, 4096]
```

不得用：

```text
final Llama hidden state
O1-S
O1-F
```

替代 O2。

---

## π0.5

主节点：

```text
P2
```

定义：

```text
PaliGemma-ready projected visual representation
```

预期主视觉输入 shape：

```text
[256, 2048]
```

不得以其它 SigLIP hidden、Gemma hidden 或 action representation 替代。

---

# 4. Threat Model

## 4.1 Surrogate

唯一攻击代理：

```text
OpenVLA
```

纹理训练期间：

```text
π0.5 MUST NOT be loaded
π0.5 MUST NOT provide loss
π0.5 MUST NOT provide gradient
π0.5 MUST NOT influence model selection
```

禁止：

\[
L =
L_{\mathrm{OpenVLA}}
+
L_{\pi0.5}
\]

禁止任何形式的：

```text
dual-surrogate optimization
ensemble loss
cross-model gradient averaging
π0.5-guided hyperparameter selection
```

---

## 4.2 Witness

π0.5 只允许在最终纹理被冻结后进行：

```text
post-training
no-grad
held-out witness analysis
```

因此实验结构必须是：

```text
OpenVLA
   │
   │ gradient
   ▼
Tex3D texture optimization
   │
   │ freeze texture
   ▼
held-out raw clean / adversarial RGB pairs
   ├── OpenVLA → O2
   └── π0.5   → P2
```

---

# 5. Step 1 实验划分

第一版固定：

```text
LIBERO suite: libero_spatial
task_id: 0
object: akita_black_bowl
```

训练状态：

```text
0–9
```

held-out analysis：

```text
10–19
```

实验不得根据 held-out 结果重新修改超参数。

---

# 6. Phase A — O2 Extraction 接入

为 corrected OpenVLA baseline 增加 O2 extraction interface。

要求：

```text
input:
    corrected OpenVLA model forward

output:
    O2 [B, 256, 4096]
```

必须证明：

```text
shape correct
dtype finite
same node as validated shared-feature extractor
gradient can propagate from O2 to image
```

不要修改 OpenVLA model weights。

建议所有模型参数：

```python
requires_grad_(False)
```

只保留：

```text
image / rendered texture graph
```

所需梯度。

---

# 7. Phase B — 新增 O2 Displacement Objective

保留原 Tex3D objective 行为。

新增 opt-in objective，例如：

```text
legacy
o2_displacement
```

默认必须仍为：

```text
legacy
```

Step 1 正式实验使用：

```text
o2_displacement
```

---

## 7.1 Clean Reference

对于训练 frame \(x_i\)：

\[
O_i = O2(x_i)
\]

clean reference 必须：

```text
cache
detach
requires_grad=False
```

不得每轮构建 clean gradient graph。

---

## 7.2 Adversarial Forward

攻击 texture 参数为：

\[
\theta
\]

经过现有 Tex3D：

```text
texture parameter
→ nvdiffrast renderer
→ front-most visibility compositor
→ Deployment View
→ exact-forward/surrogate-backward OpenVLA preprocessing
→ OpenVLA
→ O2
```

得到：

\[
O'_i(\theta)=O2(x'_i(\theta))
\]

---

## 7.3 Loss

定义：

\[
D_{O2}^{(i)}
=
\operatorname{mean}
\left[
(O'_i-O_i)^2
\right]
\]

训练 loss：

\[
L_{O2}
=
-\frac{1}{N}
\sum_iD_{O2}^{(i)}
\]

因此：

```text
minimize L_O2
=
maximize O2 displacement
```

必须增加单元测试防止符号写反。

---

## 7.4 Feature-only

在：

```text
o2_displacement
```

模式下，正式 optimization objective 必须只有：

```text
O2 negative MSE
```

不要混入：

```text
action CE
legacy final-hidden MSE
SigLIP loss
O1 loss
spectral loss
```

这些值如果为了 debug 被计算，只能作为：

```text
diagnostic
```

不得进入 total gradient。

---

# 8. Phase C — 保留 Tex3D 攻击机制

除 objective 外，不修改 Step 0 已经验证的攻击路径。

保持：

```text
renderer
vertex texture parameterization
epsilon / perturbation budget
update rule
optimizer / SignSGD semantics
lighting calibration
multi-instance handling
front-most visibility
deployment transform
asset lifecycle
```

不得为了提高 O2 displacement：

```text
换 optimizer
调 epsilon
调 learning rate
增加 EoT
增加 wrist view
增加 spectral parameterization
增加 support mask
增加 gradient protection
```

Formal run 使用 repository 当前 baseline 的 resolved Tex3D defaults。

所有实际 resolved 参数必须写入：

```text
config.json
```

---

# 9. Phase D — Held-out Raw Observation Pair

这是 Step 1 很关键的 provenance contract。

正式 held-out comparison 不应直接拿训练时 nvdiffrast composite 当最终跨模型观测。

最终 texture 冻结后，应通过 Step 0 已验证的：

```text
Active Texture
+
MuJoCo camera rendering
```

生成真正用于分析的 clean / adversarial observation。

---

## 9.1 Pair Construction

每个：

```text
state_id = 10 ... 19
```

构造一个确定性 pair：

```text
same LIBERO task
same init state
same camera
same physical object pose
same robot state
same lighting
same observation timing

clean:
    original texture

adversarial:
    frozen attack texture
```

pair 采集期间不要执行策略 action 导致轨迹分叉。

也就是说比较对象应尽量满足：

\[
x_i^{adv}
=
\text{same scene as }x_i^{clean}
\text{ except texture}
\]

---

## 9.2 保存 Common Raw Input

共同跨模型输入应保存为：

```text
Policy Source RGB
uint8
512 × 512 × 3
```

不要把已经经过 OpenVLA：

```text
224 resize
center crop
DINO/SigLIP normalize
```

的 tensor 当作跨模型共同输入。

OpenVLA 与 π0.5 都从同一份原始 RGB pair 开始，再分别执行自己的 preprocessing。

---

## 9.3 Pair Identity

每个 sample 至少记录：

```text
sample_id
task_suite
task_id
state_id
clean_rgb_sha256
adv_rgb_sha256
texture_sha256
camera
source_resolution
```

OpenVLA 和 π0.5 analysis 必须根据：

```text
sample_id + SHA256
```

确认读取的是完全相同的 raw image。

如果 hash 不一致：

```text
FAIL FAST
```

不得继续计算 correlation。

---

# 10. Phase E — OpenVLA O2 Held-out Analysis

OpenVLA-side formal measurement 使用 Step 0 的权威 OpenVLA 环境。

不要用 dependency 不一致的 joint environment 代替 authoritative OpenVLA result。

对于每个 held-out pair：

\[
\Delta O_i
=
O2(x_i^{adv})-O2(x_i^{clean})
\]

保存完整 residual：

```text
[10, 256, 4096]
```

---

## 10.1 State-level displacement

\[
d_O^{(i)}
=
\operatorname{mean}
\left[
(\Delta O_i)^2
\right]
\]

记录：

```text
state_id
d_O
```

---

## 10.2 Token-level displacement map

对于 token \(j\)：

\[
r_O^{(i)}[j]
=
\sqrt{
\frac{1}{4096}
\sum_k
\Delta O_i[j,k]^2
}
\]

shape：

```text
[256]
```

保存全部 held-out state map：

```text
[10,256]
```

---

# 11. Phase F — π0.5 P2 Witness Analysis

π0.5 仅在 texture 冻结以及 observation pair 固定后加载。

整个分析：

```text
torch.no_grad()
model.eval()
```

不得产生 attack gradient。

---

## 11.1 Observation Adapter

π0.5 必须使用其自己正确的 observation/preprocessing pipeline。

禁止：

```text
OpenVLA-preprocessed tensor → π0.5
```

共同输入只有：

```text
raw RGB observation
```

若 π0.5 还需要：

```text
robot state
proprioception
other camera image
prompt / language
```

全部保持 clean、固定。

只替换与 OpenVLA 被攻击主视图对应的 RGB field。

必须在 metadata 中明确记录：

```text
OpenVLA attacked camera field
↔
π0.5 corresponding image field
```

不要攻击 π0.5 的其它 image input。

---

## 11.2 P2 Residual

\[
\Delta P_i
=
P2(x_i^{adv})-P2(x_i^{clean})
\]

保存：

```text
[10,256,2048]
```

---

## 11.3 State-level displacement

\[
d_P^{(i)}
=
\operatorname{mean}
\left[
(\Delta P_i)^2
\right]
\]

---

## 11.4 Token-level map

\[
r_P^{(i)}[j]
=
\sqrt{
\frac{1}{2048}
\sum_k
\Delta P_i[j,k]^2
}
\]

shape：

```text
[256]
```

---

# 12. Cross-model Metrics

## 12.1 State-level trend

在 held-out states 10–19 上计算：

\[
\rho_{\mathrm{state}}
=
\operatorname{Spearman}
(
[d_O^{(i)}],
[d_P^{(i)}]
)
\]

必须同时报告：

```text
rho
p-value
N
individual d_O
individual d_P
```

本实验不预设：

```text
rho > 0.5
rho > 0.7
```

之类的人为通过阈值。

---

# 12.2 Token-level trend

对每个 state：

\[
\rho_{\mathrm{token}}^{(i)}
=
\operatorname{Spearman}
(
r_O^{(i)},
r_P^{(i)}
)
\]

输出 10 个值，以及：

```text
mean
median
min
max
```

---

# 12.3 Spatial Claim Guardrail

O2 与 P2 都有：

```text
256 visual tokens
```

但这不自动代表它们的：

```text
token j
```

就是同一图像 patch。

只有在代码层证明：

```text
both = 16 × 16 patch grid
same raster ordering
compatible crop geometry
```

之后，才允许使用：

```text
same spatial patch
spatial alignment
```

等措辞。

否则只写：

```text
token-index displacement trend
```

---

# 13. 可选但免费时可记录的指标

可以额外记录 normalized displacement：

\[
\tilde d_F
=
\frac{
\|F(x^{adv})-F(x^{clean})\|_F
}{
\|F(x^{clean})\|_F+\epsilon
}
\]

但该指标：

```text
不是主指标
不得用于调参
不得阻塞 MVP
```

---

# 14. 明确禁止的 Scope Creep

Step 1 v0.1 不实现：

```text
CCA
SVCCA
CKA
RSA
PCA
shared subspace
cross-model residual cosine
low-rank fusion
shared vulnerable direction
action relevance
FFN steering
FIA
NAA
ILA
TAP
action gradient
action loss redesign
random perturbation baseline
matched perturbation baseline
rollout ASR comparison
task-success transfer
reverse π0.5 → OpenVLA
dual-surrogate optimization
spectral attack
TAAO
EoT
multi-view optimization
hyperparameter sweep
loss sweep
threshold sweep
```

---

# 15. Environment Contract

## Source optimization / O2

优先使用 Step 0 authoritative OpenVLA environment。

Step 0 已记录的权威 dependency：

```text
torch        2.2.0+cu121
torchvision  0.17.0+cu121
transformers 4.40.1
tokenizers   0.19.1
```

不能因为 joint environment 能加载 OpenVLA 就把 formal O2 source result 换过去。

---

## π0.5

使用已经验证能够运行 OpenPI/π0.5 的 joint environment。

该环境只负责：

```text
π0.5 P2 witness extraction
```

如果需要跨环境传递数据：

```text
uint8 RGB
NPZ
JSON
SHA256
```

是允许且推荐的。

跨环境不需要共享 autograd graph，因为 π0.5 本来就不能参与攻击梯度。

---

# 16. Unit Tests

至少增加以下 regression tests。

### O2 extractor

```text
O2 shape == [B,256,4096]
finite
correct node
```

### Loss semantics

人工构造：

```text
adv farther from clean
→ displacement larger
→ L_O2 smaller
```

验证 negative-MSE sign。

### Clean detach

```text
clean O2 requires_grad == False
```

### Attack gradient

```text
adv O2
→ image
→ texture parameter
```

存在 finite nonzero gradient。

### Legacy preservation

不指定 O2 objective 时：

```text
original Tex3D objective behavior unchanged
```

### Token RMS

测试 reduction：

```text
[D=4096] → scalar
[D=2048] → scalar
```

以及：

```text
[B,256,D] → [B,256]
```

### π0.5 witness

```text
model eval
no grad
P2 [B,256,2048]
```

### Pair identity

OpenVLA / π0.5 consumer 对同一 sample：

```text
sample_id equal
clean SHA equal
adv SHA equal
```

---

# 17. GPU Smoke

正式训练前运行最小 smoke：

```text
libero_spatial
task 0
state 0
1 frame
1 optimization iteration
```

必须满足：

```text
O2 clean finite
O2 adv finite
O2 loss finite
texture gradient finite
texture gradient != 0
attack parameter changed
perturbation budget respected
XML restored
texture restored
no backup left behind
```

同时确认：

```text
π0.5 is not loaded
```

---

# 18. Witness Smoke

冻结 smoke texture 或零成本测试 texture 后：

```text
1 held-out pair
```

验证：

```text
raw clean/adv pair successfully saved
OpenVLA consumer hash correct
π0.5 consumer hash correct
O2 shape correct
P2 shape correct
P2 analysis no-grad
metrics finite
```

这个 smoke 只验证 pipeline。

不得解释成 scientific transfer result。

---

# 19. Formal Run

只有全部 smoke 通过后才运行。

固定：

```text
train states   0–9
analysis       10–19
one frozen configuration
```

不得在看见 state 10–19 结果后重新训练。

---

# 20. 必须输出的 Artifacts

建议：

```text
experiments/step1/<run-id>/
```

至少保存：

```text
config.json

training/
    loss_history.npy
    texture parameter artifact
    final attack texture PNG
    texture_sha256.txt
    training_summary.json

heldout_pairs/
    state_10/
        clean.png
        adversarial.png
        metadata.json
    ...
    state_19/

openvla/
    o2_clean.npz
    o2_adv.npz
    o2_residuals.npz
    o2_state_metrics.csv
    o2_token_rms.npz

pi05/
    p2_clean.npz
    p2_adv.npz
    p2_residuals.npz
    p2_state_metrics.csv
    p2_token_rms.npz

analysis/
    displacement_summary.csv
    token_spearman.csv
    summary.json
```

---

# 21. `displacement_summary.csv`

至少：

```text
sample_id
state_id
clean_rgb_sha256
adv_rgb_sha256
d_O2
d_P2
normalized_d_O2(optional)
normalized_d_P2(optional)
```

---

# 22. `summary.json`

至少：

```text
source_model
witness_model
source_checkpoint
witness_checkpoint
source_git_commit
witness_adapter_commit
texture_sha256
train_state_ids
heldout_state_ids

state_spearman_rho
state_spearman_p
num_states

token_spearman_per_state
token_spearman_mean
token_spearman_median

o2_shape
p2_shape

token_spatial_order_verified

scientific_scope
```

---

# 23. 最终报告

新增：

```text
docs/step1-o2-p2-transfer-report.md
```

报告必须明确区分：

## Engineering result

例如：

```text
O2 optimization pipeline works
held-out pairs are hash-identical across model consumers
P2 witness extraction works
```

## Scientific observation

例如：

```text
OpenVLA-only texture caused X O2 displacement
same texture caused Y P2 displacement
state-level Spearman = ...
token-level distribution = ...
```

## Interpretation limit

必须明确写：

> P2 的非零响应只说明该 OpenVLA-only texture 在另一 VLA 的 representation 上产生了响应。由于 v0.1 尚未加入 matched/random perturbation baseline，也尚未验证 action relevance，因此不能将该结果直接解释为“shared transferable vulnerability”。

---

# 24. Stop Conditions

遇到以下任一情况必须停止 formal experiment，并报告，不得自行加入复杂 workaround：

```text
Step 0 regression fails

OpenVLA exact-forward equivalence breaks

O2 node cannot reproduce validated [256,4096]

P2 node cannot reproduce validated [256,2048]

π0.5 appears in training graph

OpenVLA and π0.5 consume different raw pair hashes

clean/adv held-out pair changes robot/scene state

texture gradient is zero or non-finite

asset restoration fails

held-out states accidentally influence training

legacy objective behavior is changed
```

特别禁止遇到问题后自行加入：

```text
new BPDA
new surrogate loss
new renderer approximation
π0.5 gradient
multi-model loss
hyperparameter sweep
```

---

# 25. Step 1 最终问题

本轮只回答三个问题：

### Q1

OpenVLA-only O2 objective 是否能够稳定优化 texture，并在 held-out states 上产生 O2 displacement？

### Q2

同一冻结纹理是否在未参与训练的 π0.5 P2 上产生 measurable representation displacement？

### Q3

held-out states 以及 256 token-index displacement maps 上，O2/P2 是否存在一致趋势？

如果这三点已有清晰结果，Step 1 即完成。

下一阶段再决定是否进入：

```text
raw residual
→ action relevance
→ vulnerable structure extraction
→ cross-model vulnerable alignment
→ shared vulnerable representation
```

不要在本任务中提前实现。

---

# 26. 完成标准

只有同时满足：

```text
all CPU tests pass
source GPU smoke pass
witness smoke pass
formal run uses frozen configuration
10 held-out pairs provenance complete
O2/P2 artifacts complete
all correlations reproducible from saved artifacts
repository clean
assets restored
final report committed
```

才能标记：

```text
STEP 1 COMPLETE
```
