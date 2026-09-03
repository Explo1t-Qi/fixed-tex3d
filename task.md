# Codex 前置任务书 v2  

# Step 0 — Tex3D/OpenVLA Baseline Correctness Repair & Smoke Validation

Before implementing any Step-0 fix, create and commit the repository-root `AGENTS.md` exactly for long-term engineering rules, then read it together with this task specification. Task-specific scientific requirements belong to the task specification, not to `AGENTS.md`.

## 1. 任务目标

在实现后续 Step 1：

```text
O2 adversarial displacement
→ π0.5 P2 response analysis
```

之前，先修复并验证原始 Tex3D OpenVLA pipeline 中已经由历史 `modified-tex3d` 研究确认的 correctness 问题。

本任务的目标不是改进攻击算法，而是建立一个：

```text
scientifically trustworthy
+
deployment-consistent
+
differentiable
```

的 Tex3D/OpenVLA baseline。

只有 Step 0 验收通过后，才能开始 Step 1。

---

# 2. 本地仓库位置

Workspace 根目录：

```text
/home/xmq/src
```

当前相关仓库：

```text
/home/xmq/src/
├── mechanistic-steering-vlas
├── modified-tex3d
├── openpi
├── openvla
├── shared-feature-tex3d
└── tex3d
```

---

# 3. 各仓库职责

## 3.1 原始 Tex3D

主要修改对象：

```text
/home/xmq/src/tex3d
```

当前已知 baseline：

```text
branch:
main

HEAD:
bfd0726cad6af4514251af3a717736cb8746ecf6

working tree:
clean
```

开始工作前重新执行：

```bash
cd /home/xmq/src/tex3d

git status
git rev-parse HEAD
git remote -v
```

如果 HEAD 或工作区状态与上述信息不一致：

```text
STOP
```

不要静默继续。

---

## 3.2 OpenVLA 官方仓库

```text
/home/xmq/src/openvla
```

用途：

```text
authoritative reference for
OpenVLA model / processor / preprocessing semantics
```

涉及以下内容时，以这里的真实代码和 checkpoint configuration 为准：

```text
vision branch ordering
processor configuration
image resize semantics
normalization
center crop
prompt/model input semantics
action decoding
```

---

## 3.3 modified-tex3d

```text
/home/xmq/src/modified-tex3d
```

用途：

```text
historical bug diagnosis
reference implementation
regression-test reference
```

不得整体复制该仓库。

只提取本任务确认需要的 baseline correctness fixes。

---

## 3.4 shared-feature-tex3d

```text
/home/xmq/src/shared-feature-tex3d
```

本任务原则上：

```text
READ ONLY
```

Step 1 的 O2/P2 研究将在 Step 0 完成后另外实施。

---

# 4. Git 工作方式

不得直接修改：

```text
main
```

从：

```text
bfd0726cad6af4514251af3a717736cb8746ecf6
```

创建独立分支，例如：

```bash
git switch -c fix/openvla-baseline-correctness
```

最终要求：

```text
main remains untouched
all Step-0 changes live on dedicated branch
```

---

# 5. 重要原则：先审计，再修复

下面列出的 bug 均来自 `modified-tex3d` 的历史诊断。

但是 Codex 不应机械复制历史 patch。

对每个 bug 都必须先在：

```text
/home/xmq/src/tex3d
HEAD bfd0726...
```

验证其是否仍然存在。

最终每个 bug 必须得到以下三种状态之一：

```text
REPRODUCED_AND_FIXED

NOT_PRESENT_IN_CURRENT_BASELINE

NOT_REPRODUCIBLE
```

后两种情况必须给出证据。

---

# 6. 已确认需要审计的 Baseline Bug Inventory

---

## BUG 1 — OpenVLA generation input / attention-mask alignment

### 历史问题

OpenVLA `predict_action()` 在 prompt 尾部缺少 LLaMA empty token 时可能自动补：

```text
input_ids
```

但原调用路径已经存在的：

```text
attention_mask
```

不会同步扩展。

结果可能导致：

```text
input_ids length
!=
attention_mask length
```

最终在 generation / attention 中产生 shape mismatch。

`modified-tex3d` 后来增加了同步补齐逻辑。

### 当前任务

检查：

```text
/home/xmq/src/tex3d/openvla/experiments/robot/openvla_utils.py
```

当前 `get_vla_action()` 是否仍存在该问题。

如果存在：

```text
fix input_ids and attention_mask together
```

并增加 regression test。

---

# 7. BUG 2 — OpenVLA fused vision branch ordering 错误

原 Tex3D attack path 手工构造：

```text
SigLIP
→ DINOv2
```

但真实 checkpoint configuration 为：

```text
DINOv2
→ SigLIP
```

原代码因此将两种已经按不同 mean/std normalization 的图像送入了错误的 vision encoder。

历史审计确认这不是小数值误差，而是实际 branch semantics 错误。

### 修复要求

禁止继续：

```text
hard-code
SigLIP first
DINO second
```

顺序必须从：

```text
model.config.timm_model_ids
+
processor.image_processor
```

等真实 checkpoint configuration 获取。

---

# 8. BUG 3 — Differentiable resize semantics 错误

原 attack path 使用：

```python
F.interpolate(..., mode="bilinear")
```

而真实 checkpoint processor 对应：

```text
bicubic
+
antialias
```

这会导致 attack-time model input 与真实 processor input 不一致。

### 修复要求

实现 checkpoint-derived resize semantics：

```text
RGB tensor
→ correct resize
→ correct branch normalization
→ fused pixel_values
```

不要在不同模块手写多个 resize implementation。

---

# 9. BUG 4 — Preprocessing configuration 被手工硬编码

原 Tex3D attack path手工定义：

```text
SigLIP mean/std
DINO mean/std
branch position
input size
```

即使其中部分数值当前恰好正确，这种实现仍与真实 checkpoint processor 脱节。

历史修复最终改为从：

```text
model
+
processor
```

构造唯一 preprocessing specification。

### 修复要求

建立唯一 preprocessing interface。

语义上类似：

```text
DifferentiableOpenVLAImageProcessor
```

具体名称由当前仓库风格决定。

至少读取：

```text
branch ordering
image size
interpolation
antialias
mean
std
```

不要根据“已知 OpenVLA 应该是什么”重新硬编码。

---

# 10. BUG 5 — Clean label / clean hidden / adversarial logits 使用不同视觉路径

这是 BUG 2–4 导致的一个独立科学问题。

原始 pipeline 中：

```text
clean_output_ids
```

由：

```text
official checkpoint processor
```

产生。

但：

```text
clean_hidden
```

和 attack optimization 中的：

```text
adv logits
adv hidden
```

来自手工构造的 6-channel tensor。

因此实际形成：

```text
clean labels from visual path A

vs.

attack logits from visual path B
```

历史审计发现 zero texture 情况下两条 clean path 已经平均出现明显 action-token disagreement。

### 修复要求

正式训练中的：

```text
clean reference
adv forward
action loss
hidden feature loss
```

必须共享同一个 image preprocessing contract。

官方 processor 仍然保留为：

```text
independent oracle
```

用于 equivalence test。

禁止形成循环自证：

```text
new preprocessing
vs.
new preprocessing
```

---

# 11. BUG 6 — Policy input 被录像分辨率隐式决定

历史 rollout 曾从：

```text
video-resolution camera image
```

直接 resize 得到 policy input。

这意味着：

```text
changing replay/video resolution
```

可能同时改变：

```text
OpenVLA policy input
```

这是错误的职责耦合。

后续修复明确建立：

```text
MuJoCo observation
      ↓
fixed 512×512 Policy Source
      ↓
224×224 Policy Pre-Crop Canvas
```

录像分辨率不再影响 policy input。

### 当前任务

确认原 Tex3D 是否仍存在：

```text
video resolution
→ policy image
```

的隐式耦合。

如果存在，修复为固定 policy-source contract。

对于当前 LIBERO/OpenVLA baseline：

```text
Policy Source:
512 × 512

Pre-Crop Canvas:
224 × 224
```

除非当前官方 OpenVLA/LIBERO implementation 明确要求其他语义。

---

# 12. BUG 7 — Attack Training 缺少真实 Deployment Center Crop

OpenVLA deployment path 在：

```text
224×224 Pre-Crop Canvas
```

之后还执行训练时约定的 center crop，然后 resize 回模型输入大小。

历史实现中 rollout 有该逻辑，但 attack differentiable path 并没有正确消费同一个变换。

后续仓库将其显式定义成：

```text
Deployment Effective View
```

并让 rollout 与 attack training 共用同一 crop specification。

当前历史配置对应：

```text
crop area = 0.9
```

### 修复要求

必须确认：

```text
official rollout path
```

真正使用的 center-crop semantics。

然后确保：

```text
attack training
```

看到的有效视野与 deployment 一致。

即：

```text
MuJoCo Policy Source
        ↓
Pre-Crop Canvas
        ↓
Deployment Center Crop
        ↓
OpenVLA processor
```

不能只做：

```text
renderer
→ resize 224
→ processor
```

---

# 13. BUG 8 — Renderer 默认 position offset 导致几何错位

原 Tex3D renderer 历史上使用：

```text
[0.02, 0.01, 0.025]
```

的默认 position offset。

实际 MuJoCo segmentation / nvdiffrast alignment audit 显示该 offset 会导致严重投影错位。

Spatial task 0：

```text
historical offset:
IoU ≈ 0.3852
center error ≈ (-13.79, -15.74) px

zero offset:
IoU ≈ 0.9990
center error ≈ (-0.01, +0.02) px
```

Object task 0 也有相同问题。

### 修复要求

默认：

```text
position offset = exact zero
```

只有某个具体资产经过显式 alignment measurement 后，才允许提供非零 calibration offset。

禁止历史 magic constant 继续作为默认值。

---

# 14. BUG 9 — 正式 Evaluation / Live Test 错误地再次使用 nvdiffrast 重画物体

原 Tex3D 在最终 texture 已经安装到 MuJoCo XML 后，evaluation/live-test 仍会：

```text
hide target in MuJoCo
+
nvdiffrast render adversarial foreground
+
composite
```

这意味着 evaluation 并不是实际部署语义。

后续审计确认 nvdiffrast foreground：

```text
没有 MuJoCo scene depth buffer
```

因此可能画出：

```text
被桌面或其他物体遮挡的表面
```

正式 policy 应直接读取：

```text
MuJoCo camera frame
with Active Texture installed
```

而不是再次进行 differentiable compositing。

### 修复要求

明确边界：

```text
Differentiable renderer:
Attack Training only

MuJoCo Active Texture:
Live Test / Formal Rollout
```

---

# 15. BUG 10 — Policy 看到的 frame 与保存的视频 frame 不一致

这是 BUG 9 的伴随问题。

历史 live/evaluation path 中可能：

```text
policy sees nvdiffrast-composited image

but

video stores MuJoCo camera image
```

因此录像不能作为 policy 实际输入的证据。

后续修复要求：

```text
policy image
and
replay/video image
```

从同一个 MuJoCo observation 派生。

### 修复要求

录像与 policy input 必须有明确共同 provenance。

录像可以使用不同 resolution：

```text
but must not represent a different scene rendering
```

---

# 16. BUG 11 — Shared Texture Multi-Instance 错误

这一项与本项目 Step 1 **直接相关，必须修复**。

LIBERO Spatial task 0 中存在：

```text
akita_black_bowl_1
akita_black_bowl_2
```

两个实例。

它们引用：

```text
the same texture PNG
```

因此最终安装 adversarial PNG 时：

```text
both bowls change
```

但原 renderer / target-body search：

```text
only finds/renders first matched body
```

历史 GPU audit 中因此出现：

```text
primary-view observed recall ≈ 52.45%
```

而不是完整 texture influence。

后续修复改为：

```text
find all instances sharing target texture

for each instance:
    own model matrix
    own MVP
    own rasterization

all instances:
    share same texture parameters

gradients:
    summed into same texture parameter
```



### 修复要求

对于一个 texture asset：

```text
texture parameter
```

是研究对象，而不是单个 semantic body。

必须识别该纹理实际影响的全部 scene instances。

特别验证：

```text
libero_spatial
task 0
akita bowl
```

---

# 17. BUG 12 — Training Compositor 没有使用 MuJoCo front-most visibility

即便所有 shared-texture instances 都被 renderer 重建，nvdiffrast mask 仍只有：

```text
renderer projection visibility
```

并不知道真实 MuJoCo scene 中：

```text
which instance/surface is front-most
```

因此多个 renderer instance：

```text
可能重叠
```

或者在真实场景中已经被桌子/其他物体遮挡。

后续正确方案不是：

```text
replace MuJoCo foreground entirely
```

而是使用 renderer 提供的**纹理颜色变化量**：

\[
F_{\mathrm{adv}}-F_{\mathrm{clean}}
\]

再由 MuJoCo front-most instance segmentation 控制哪些像素真正允许接受这个变化：

\[
I_{\mathrm{adv}}
=
I_{\mathrm{MuJoCo}}
+
\sum_k
M_k
\left(
F_{\mathrm{adv},k}
-
F_{\mathrm{clean},k}
\right)
\]

即：

```text
Visibility-Masked Renderer Delta Composition
```

历史实现明确指出 renderer mask 不得替代 MuJoCo occlusion alpha。

### 修复要求

Codex 应审计原始 training compositor。

如果当前仍然采用：

```text
adv foreground
× renderer mask
+
background without object
```

这种整体 foreground replacement，

则需要修复为 deployment-consistent texture-delta composition。

要求：

```text
zero texture delta
→ exact clean MuJoCo image

only visible target pixels
receive renderer texture delta

multi-instance order
does not change result

gradient
flows only through adversarial renderer branch
```

---

# 18. BUG 13 — Runtime Texture / Material 通过猜测节点名称定位

原逻辑假设：

```text
texture name ~= tex-{object_name}
material name ~= mat-{object_name}
```

但 HOPE 等资产可能实际使用：

```text
tex-textured
textured
```

因此可能找不到真正被 XML 引用的 texture/material。

后续修复改成优先根据：

```text
XML texture file reference
+
material → texture relation
```

定位真实节点，再把 name matching 作为 fallback。

### 修复要求

Runtime texture activation 应优先依据实际 XML relationship：

```text
texture.file
material.texture
```

而不是依赖 object_name naming convention。

虽然 Step 1 首先使用 Spatial bowl，也应在 Step 0 一次性修复这类 baseline asset bug。

---

# 19. 不属于本轮 Baseline Bug Fix 的历史修改

Codex 不得把 `modified-tex3d` 中所有后续 commit 都移植。

以下属于研究方法、诊断工具或后续实验，不是本轮 baseline correctness repair：

```text
spectral texture parameterization

K=128 / K=256 / K=512

Shared-SigLIP attack

dual-view feature objective

gradient norm protection

PCGrad

fixed support

spectral guard

Action hinge / κ objective

CWA-style analysis

cross-model gradient diagnostics

Gate 6g–6j

P2 / OpenVLA-OFT transfer experiments
```

另外：

```text
256 action-token classes
vs.
255 continuous bin centers
```

的历史修复发生在后续 Gate 6g audit schema 中，不应自动视为原 Tex3D action codec bug。

不得因为看到该 commit 就修改原 attack semantics。

---

# 20. BPDA / STE 的处理原则

历史最终实现使用：

```text
exact PIL/uint8 forward
+
continuous PyTorch backward
```

即 BPDA/STE。

原因是即便修复：

```text
branch ordering
bicubic
antialias
```

后，Tensor bicubic 与真实 PIL processor 之间仍存在小差异，并且部分 OpenVLA action token 恰好位于很窄的 decision margin 附近。

但是本任务仍采用：

```text
BPDA is fallback,
not default assumption
```

首先实现最简单的：

```text
checkpoint-derived differentiable path
```

然后做真实 equivalence audit。

如果：

```text
official processor
vs.
differentiable path
```

在自然 LIBERO clean frame 上达到：

```text
7 / 7 action tokens identical
```

则：

```text
DO NOT introduce BPDA
```

如果不能达到：

```text
record exact mismatch
```

并停止。

不要由 Codex 自行升级 BPDA。

我们在查看结果后再决定。

---

# 21. 推荐修复顺序

不要同时修改所有代码后再测试。

按照依赖关系实施：

```text
Phase A
Model-input correctness

BUG 1
BUG 2
BUG 3
BUG 4
BUG 5
        ↓
processor/model equivalence
        ↓

Phase B
Deployment-view correctness

BUG 6
BUG 7
        ↓
clean deployment-view equivalence
        ↓

Phase C
Renderer semantics

BUG 8
BUG 11
BUG 12
        ↓
renderer / MuJoCo alignment smoke
        ↓

Phase D
Evaluation / asset semantics

BUG 9
BUG 10
BUG 13
        ↓

Phase E
one-step end-to-end Tex3D smoke
```

每个 phase 单独 commit。

---

# 22. Phase A — Processor Equivalence Test

至少使用：

```text
3 deterministic synthetic images
+
1 real LIBERO Spatial task0 state0 observation
```

比较：

```text
official processor
vs.
new differentiable preprocessing
```

报告：

```text
pixel_values shape

global MAE
global Linf

per-branch MAE
per-branch Linf

resolved branch ordering

resize mode
antialias
mean/std
```

然后使用真实 OpenVLA：

```text
official processor clean input
vs.
new preprocessing clean input
```

比较：

```text
7 action tokens
token Hamming

decoded action L2
decoded action Linf
```

---

# 23. Phase B — Deployment Effective View Test

固定：

```text
libero_spatial
task_id = 0
state_id = 0
```

验证完整 clean deployment path：

```text
MuJoCo observation
      ↓
512×512 Policy Source
      ↓
224×224 Pre-Crop Canvas
      ↓
center crop
      ↓
processor
      ↓
OpenVLA
```

并确认：

```text
changing replay/video resolution
does not change policy input
```

同时确认 attack training 使用相同：

```text
Policy Source
Pre-Crop
center-crop specification
```

---

# 24. Phase C — Renderer Alignment Test

对：

```text
libero_spatial
task 0
state 0
```

比较：

```text
MuJoCo segmentation
vs.
nvdiffrast projection
```

至少报告：

```text
IoU
visible recall
center offset
bbox
```

要求默认 zero position offset。

同时验证：

```text
all shared-texture Akita bowl instances
```

均被识别。

必须输出：

```text
instance count
instance names
per-instance visible pixel count
```

---

# 25. Zero-Delta Compositor Test

这一项非常重要。

令：

```text
texture delta = 0
```

则 training compositor 输出必须满足：

\[
I_{\mathrm{composited}}
=
I_{\mathrm{MuJoCo,clean}}
\]

允许的误差应接近数值精度。

如果 zero-delta 时图像已经变化：

```text
FAIL
```

因为这意味着后续 O2 displacement 可以来自 renderer mismatch，而不是 texture perturbation。

---

# 26. Gradient Test

在修复后的：

```text
renderer
→ compositor
→ deployment view
→ processor
→ OpenVLA
→ original Tex3D loss
```

路径中检查：

```text
loss finite

image gradient:
finite
non-zero

texture gradient:
finite
non-zero
```

---

# 27. One-Step Tex3D Smoke

仅运行：

```text
libero_spatial
task 0

train state:
0

frames:
1

attack iterations:
1
```

保持原始 Tex3D objective：

```text
Action loss
+
final hidden-state negative-MSE feature loss
```

不得加入 O2。

不得改变 loss weights 以追求攻击效果。

必须报告：

```text
action loss
feature loss
total loss

texture gradient norm

parameter before/after change

maximum texture perturbation

artifact paths
```

---

# 28. One-Episode Evaluation Smoke

one-step texture bake 后：

```text
install Active Texture into MuJoCo
```

然后进行：

```text
1 rollout
```

正式 evaluation 必须：

```text
read MuJoCo camera frame directly
```

不得再次用 nvdiffrast 替换 foreground。

检查：

```text
rollout completes

policy input provenance valid

video saved

XML restored

original texture restored

temporary backup removed
```

任务 success/failure：

```text
NOT a scientific metric
```

---

# 29. 必须增加的 Regression Tests

至少包括：

```text
[1] generation input_ids / attention_mask stay aligned

[2] vision branch order follows checkpoint config

[3] resize semantics follow checkpoint processor config

[4] normalization follows processor config

[5] clean and adversarial paths share preprocessing contract

[6] policy input independent from replay resolution

[7] deployment center crop has one shared specification

[8] renderer default offset = zero

[9] shared-texture instances all discovered

[10] multi-instance texture gradients accumulate

[11] zero texture delta gives clean MuJoCo image

[12] visibility mask uses MuJoCo front-most pixels

[13] runtime texture is resolved from XML references

[14] texture activation and cleanup restore original assets

[15] differentiable path returns finite non-zero texture gradient
```

---

# 30. 不做完整攻击实验

Step 0 不运行：

```text
5000-step formal attack

held-out attack success comparison

OFT evaluation

π0.5 evaluation
```

本轮只证明：

```text
baseline correctness
```

而不是攻击有效性。

---

# 31. Codex 最终交付

Codex 必须返回一份结构化报告。

## A. Bug audit table

格式：

| Bug | Baseline status | Fix | Test |
|---|---|---|---|
| BUG 1 | reproduced / ... | ... | PASS |
| ... | ... | ... | ... |

13 项全部必须出现。

---

## B. Changed files

逐个说明：

```text
path
change
reason
```

---

## C. Final data flow

必须画出：

```text
MuJoCo observation
      ↓
Policy Source
      ↓
Differentiable renderer texture delta
      ↓
MuJoCo-visibility compositor
      ↓
Deployment Effective View
      ↓
checkpoint-derived preprocessing
      ↓
OpenVLA
      ↓
original Tex3D objective
      ↓
texture gradient
```

以及 evaluation：

```text
baked texture
      ↓
MuJoCo Active Texture
      ↓
MuJoCo camera
      ↓
Deployment Effective View
      ↓
OpenVLA
```

---

## D. Numerical evidence

报告：

```text
processor MAE/Linf
action-token Hamming

renderer IoU/recall
shared instance count

zero-delta compositor error

gradient norm

one-step texture update
```

---

## E. Test commands

提供可以直接复现的：

```text
CPU tests

processor equivalence smoke

renderer alignment smoke

one-step GPU smoke

one-episode rollout smoke
```

---

## F. Git state

必须报告：

```text
repository:
/home/xmq/src/tex3d

base:
bfd0726cad6af4514251af3a717736cb8746ecf6

branch:
<actual>

HEAD:
<actual>

git status:
<actual>
```

---

# 32. Step 0 最终 Go / No-Go

只有以下全部成立：

```text
[ ] OpenVLA model inputs correct

[ ] clean processor equivalence acceptable

[ ] training and deployment effective view aligned

[ ] renderer geometry aligned

[ ] shared texture instances handled

[ ] zero texture delta reproduces clean MuJoCo image

[ ] texture gradient finite and non-zero

[ ] evaluation uses real MuJoCo Active Texture

[ ] runtime texture correctly installed

[ ] XML and texture assets restored

[ ] original Tex3D objective unchanged

[ ] one-step end-to-end smoke passes
```

才能输出：

```text
STEP 0 BASELINE READY
```

否则：

```text
STEP 0 NOT READY
```

不得自动开始 Step 1。