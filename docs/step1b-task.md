# Step 1B — 5000-Iteration O2 Mature Trajectory with Frozen Checkpoints

## 0. 任务目的

本任务不是设计新方法，也不是进入 Step 2。

唯一目的：

> 在当前已经验证通过的 OpenVLA O2-only Tex3D pipeline 上执行一次完整 5000-iteration optimization，并沿同一条 optimization trajectory 保存多个固定 checkpoint，用于之后分析 source optimization strength 与跨模型 O2/P2 response 之间的关系。

核心实验变量只有：

```text
optimization iteration
```

其它所有配置必须保持当前 Step 1 不变。

---

# 1. Git 基线

工作仓库：

```text
Explo1t-Qi/fixed-tex3d
```

当前冻结 Step 1 分支：

```text
feat/step1-o2-p2-transfer-mvp
```

冻结 Step 1 HEAD：

```text
e8d7d5eb97d08cbad50d1f8295651867c43d9be6
```

建议从该 commit 创建新分支：

```text
feat/step1b-mature-o2-trajectory
```

不要继续修改已经标记 `STEP 1 COMPLETE` 的历史实验语义。

---

# 2. 必须保持不变的部分

以下内容全部冻结：

```text
suite                     libero_spatial
task_id                   0
object                    akita_black_bowl

attack_objective          o2_displacement
attack_lr                 0.05
seed                      7

train states              0–9
train_frames_per_state    1
num_train_init_states     10
num_frames_to_attack      20

photometric_calib_frames  5

frame_collect_with_policy False
collect_grasp_frames      False

num_trials_per_task       0
live_test_enabled         False

renderer                  unchanged
texture parameterization  unchanged
epsilon / budget          unchanged
SignSGD                   unchanged
OpenVLA preprocessing     unchanged
O2 extractor              unchanged
non-zero O2 init          unchanged
```

禁止：

```text
修改 O2 loss
修改 initialization
修改 learning rate
修改 epsilon
修改 optimizer
增加 action loss
增加 π0.5 loss
增加 wrist objective
增加 spectral method
增加 EoT
增加新的 BPDA
```

---

# 3. 不要修改原 Step 1 formal gate

当前：

```text
step1_formal=True
```

对应已经完成的 10-iteration Step 1。

必须保持其历史行为：

```text
attack_iters = 10
```

不得简单把现有 formal gate 中的 `10` 改成 `5000`。

新增一个独立且非常窄的模式，例如：

```text
step1b_mature_trajectory: bool = False
```

当：

```text
step1b_mature_trajectory=True
```

时，冻结：

```text
attack_objective = o2_displacement
attack_iters = 5000
train states = 0–9
其余配置与 Step 1 完全一致
```

同时：

```text
step1_formal
```

和：

```text
step1b_mature_trajectory
```

不得同时为 True。

---

# 4. 固定 checkpoint schedule

本实验预注册以下 checkpoint：

```text
10
100
500
1000
2000
5000
```

定义为：

> 完成第 N 次 parameter update 之后的纹理状态。

注意 optimization loop 当前使用 zero-based `i`。

因此 checkpoint 条件必须基于：

```python
iteration = i + 1
```

而不是 `i`。

必须保证：

```text
checkpoint 10
=
完成 10 次 SignSGD update 后
```

---

# 5. 只训练一次

严禁：

```text
分别训练 10 iter
分别训练 100 iter
分别训练 500 iter
...
```

正确流程：

```text
single initialization
       ↓
single optimization trajectory
       ↓
iter 10    → save
iter 100   → save
iter 500   → save
iter 1000  → save
iter 2000  → save
iter 5000  → save
```

所有 checkpoint 必须来自完全相同的一条 optimization trajectory。

---

# 6. Checkpoint 保存位置

建议：

```text
<RUN_ROOT>/
    config.json
    asset_restoration.json

    training/
        final_attack_texture.png
        parameter.pt
        texture_sha256.txt
        loss_history.npy
        training_summary.json
        Ep0_step_metrics.jsonl
        Ep0_gradient_log.txt

        checkpoints/
            checkpoint_manifest.json

            iter_000010/
                parameter.pt
                attack_texture.png
                metadata.json

            iter_000100/
                parameter.pt
                attack_texture.png
                metadata.json

            iter_000500/
                ...

            iter_001000/
                ...

            iter_002000/
                ...

            iter_005000/
                ...
```

不要把 held-out analysis 文件写入 `training/checkpoints/`。

---

# 7. Checkpoint 保存语义

建议增加一个小型 helper，例如：

```text
_save_step1b_checkpoint(...)
```

在 parameter update 完成之后调用。

每次 checkpoint 必须保存：

## parameter

```text
parameter.pt
```

内容：

```text
renderer.get_texture_param().detach().cpu()
```

## baked texture

```text
attack_texture.png
```

必须使用与最终 texture 相同的：

```text
renderer.get_baked_adv_texture()
```

路径。

## metadata

至少记录：

```text
iteration
attack_objective
parameter_sha256
texture_sha256
parameter_linf
maximum_texture_perturbation
renderer_epsilon
seed
attack_lr
source_git_commit
```

如果记录 iteration loss/O2 displacement，必须明确其语义：

当前 step metric 是在 parameter update 之前计算的。

因此字段应命名为类似：

```text
pre_update_total_loss
pre_update_o2_displacement
```

不要把它错误描述成 checkpoint texture 更新后的重新测量值。

不需要为了 checkpoint 额外做 OpenVLA forward。

---

# 8. Checkpoint 保存不得影响训练

checkpoint hook 必须：

```text
torch.no_grad()
```

不得：

```text
调用 held-out states
调用 π0.5
运行 rollout
调用 live test
改变 optimizer
改变 parameter
改变 NumPy/PyTorch RNG
改变 renderer calibration
修改 MuJoCo asset lifecycle
```

保存 checkpoint 只是 serialization side effect。

---

# 9. Checkpoint Manifest

生成：

```text
training/checkpoints/checkpoint_manifest.json
```

至少包含：

```text
schema_version
checkpoint_schedule
completed_checkpoints
source_git_commit
attack_objective
total_iterations
seed
attack_lr
train_state_ids
heldout_state_ids
```

每个 checkpoint：

```text
iteration
parameter_path
parameter_sha256
texture_path
texture_sha256
maximum_texture_perturbation
```

5000 iter 全部完成后：

```text
completed_checkpoints
```

必须精确等于：

```text
[10,100,500,1000,2000,5000]
```

---

# 10. Final consistency

5000 checkpoint 与训练结束后的正式 final artifact 来自同一个最终 renderer state。

必须验证：

```text
iter_005000/parameter.pt
```

与：

```text
training/parameter.pt
```

内容一致。

并验证：

```text
iter_005000/attack_texture.png
```

与：

```text
training/final_attack_texture.png
```

SHA256 一致。

如果不一致：

```text
FAIL
```

---

# 11. 不在训练期间分析 held-out set

训练 5000 iter 期间：

```text
states 10–19
```

不得被加载用于：

```text
checkpoint selection
early stopping
hyperparameter tuning
O2 evaluation
P2 evaluation
```

完整 5000 iteration 必须先结束。

只有：

```text
training complete
+
all checkpoints frozen
```

以后才能进行 held-out post-hoc analysis。

---

# 12. Post-hoc checkpoint analysis

训练完成之后，对：

```text
10
100
500
1000
2000
5000
```

依次运行当前已经验证过的 pipeline：

```text
checkpoint texture
       ↓
step1_collect_heldout_pairs.py
       ↓
states 10–19 raw MuJoCo pairs
       ↓
step1_openvla_analysis.py
       ↓
O2
       ↓
step1_pi05_witness.py
       ↓
PyTorch π0.5 P2
       ↓
step1_analyze_transfer.py
```

π0.5 必须继续使用：

```text
PyTorch PI0Pytorch
model.safetensors
model.eval()
torch.no_grad()
parameters frozen
embed_image()
official embed_prefix slice check
```

不得回退 JAX witness。

---

# 13. Post-hoc 输出结构

建议：

```text
<RUN_ROOT>/
    trajectory/

        iter_000010/
            heldout_pairs/
            openvla/
            pi05/
            analysis/

        iter_000100/
            ...

        iter_000500/
            ...

        iter_001000/
            ...

        iter_002000/
            ...

        iter_005000/
            ...
```

不要覆盖之前 Step 1 的：

```text
step1-o2-p2-formal-v1
```

---

# 14. 每个 checkpoint 要得到的指标

继续沿用 Step 1：

## source

\[
d_{O2}^{(i)}
=
\operatorname{mean}(\Delta O2_i^2)
\]

## witness

\[
d_{P2}^{(i)}
=
\operatorname{mean}(\Delta P2_i^2)
\]

## state trend

\[
\rho_{\mathrm{state}}
=
Spearman(d_{O2},d_{P2})
\]

## token-index trend

每个 state：

\[
\rho_{\mathrm{token}}^{(i)}
=
Spearman(r_O^{(i)},r_P^{(i)})
\]

最后重点比较：

```text
iteration
mean d_O2
mean d_P2
state rho
token rho mean
token rho median
token rho min/max
```

---

# 15. 建议增加 trajectory summary

如果可以用很小改动完成，可新增一个 dependency-light 汇总脚本：

```text
scripts/step1b_summarize_trajectory.py
```

只读取已经完成的：

```text
trajectory/iter_*/analysis/summary.json
```

生成：

```text
trajectory/trajectory_summary.csv
trajectory/trajectory_summary.json
```

CSV 至少：

```text
iteration
mean_d_O2
median_d_O2
mean_d_P2
median_d_P2
state_spearman_rho
state_spearman_p
token_spearman_mean
token_spearman_median
token_spearman_min
token_spearman_max
texture_sha256
```

这个脚本不得重新运行模型。

如果实现会造成明显 scope creep，则可以不做，最后由现有 JSON 手工汇总。

---

# 16. Tests

在现有 52 tests 基础上增加最小 regression。

至少验证：

### schedule

```text
[10,100,500,1000,2000,5000]
```

固定且排序唯一。

### iteration semantics

```text
i=9
→ checkpoint iteration 10
```

### no checkpoint on unrelated iteration

例如：

```text
i=10
→ 不保存 iteration 11
```

### serialization

checkpoint parameter / texture / metadata 正常保存。

### manifest

所有预期 checkpoint 完成时 manifest 完整。

### final equivalence

5000 checkpoint 与 final artifact hash 相同。

### historical Step 1 preservation

```text
step1_formal=True
attack_iters=10
```

仍然通过旧 formal validator。

### Step 1B gate

```text
step1b_mature_trajectory=True
attack_iters=5000
```

通过。

任何其它 frozen parameter 改动必须 fail-fast。

---

# 17. CPU validation

修改后先执行：

```text
tests/unit
```

要求：

```text
all pass
```

并保证：

```text
git status clean
```

---

# 18. 不需要重新做的内容

本任务不需要重新实现：

```text
Step 0 baseline
O2 extractor
P2 extractor
PyTorch π0.5 adapter
held-out pair logic
wrist fix
OpenVLA consumer
π0.5 consumer
Spearman implementation
```

也不需要重新做原来的 10-step formal experiment。

---

# 19. 服务器最终命令要求

Codex 完成代码后，必须给用户一份：

```text
docs/step1b-server-validation.md
```

以及聊天中可以直接复制执行的完整 Bash 命令。

命令必须包含以下阶段：

```text
A. sync + HEAD / clean check
B. CPU regression
C. 5000-iteration single training run
D. checkpoint completeness/hash audit
E. training结束后依次处理6个checkpoint
F. each checkpoint MuJoCo pair collection
G. OpenVLA O2 extraction
H. PyTorch π0.5 P2 extraction
I. transfer analysis
J. optional trajectory summary
K. final artifact/hash audit
L. XML/texture restoration
M. backup search
N. git status
```

使用服务器当前已验证路径和 GPU：

```text
GPU 7
```

不要让我手工补变量。

命令应直接定义：

```text
REPO
LOG_ROOT
OPENVLA_PY
OPENVLA_CKPT
LIBERO_ROOT_PATH
JOINT_PY
OPENPI_ROOT
SHARED_ROOT
PI05_CKPT
RUN_ID
RUN_ROOT
```

所有 output directory 必须 fresh / fail-closed。

---

# 20. 本轮科学目标

本任务不要求提前解释结果。

最终只希望得到：

\[
t
\rightarrow
\{
d_{O2},
d_{P2},
\rho_{state},
\rho_{token}
\}
\]

其中：

\[
t\in
\{10,100,500,1000,2000,5000\}
\]

用于回答：

> 随着 OpenVLA O2 source optimization 越来越充分，π0.5 representation response 以及跨模型 token/state structure 是增强、保持还是消失？

---

# 21. 完成条件

只有满足：

```text
5000 iterations completed once
all 6 checkpoints saved from same trajectory
checkpoint hashes auditable
5000 checkpoint == final artifact
no held-out leakage during training
all CPU tests pass
server training completes
all assets restored
complete server command provided
```

才标记：

```text
STEP 1B IMPLEMENTATION READY
```

注意：

不要因为代码实现完成就宣称：

```text
STEP 1B SCIENTIFIC RESULT COMPLETE
```

科学结果只有服务器 5000-run 与全部 checkpoint post-hoc analysis 完成后才能给出。
