# Gate 4 预检：动作表示的三种语义、接触时间线与标签口径（回应 `gate4_review_0e7bb26.md`）

本轮**没有训练**。目的：在启动 300+300 之前，把"表示"定义成可比较、可验收的训练-部署契约，
并修正上一轮 `p0_substep_analysis.md` 里两处口径错误（命令路径 vs 目标偏差、接触与重推理不同时刻）。

## 1. 三种表示，只有两种能精确重建专家命令

同一专家序列 `A[t]`（下发目标）、实测 `S[t]`、锚点 `q0 = S[t]`：

| 表示 | 标签 | 执行器 | 重建专家命令？ |
| --- | --- | --- | --- |
| **anchor** | `A[t+k] − S[t]` | `fixed` | **是**（预检 `anchor_fixed_max = 0.0` rad） |
| **target_increment** | `k=0: A[t]−S[t]`；`k>0: A[t+k]−A[t+k−1]` | `recursive` | **是**（`target_increment_recursive_max = 0.0` rad） |
| observed_residual | `A[t+k] − S[t+k]` | `recursive` | **否**（`max = 0.379`、末步 `0.376` rad） |

数据：`Gate 4 preflight --dataset`（train 61 episodes / 2521 windows，audit 脚本 `scripts/exp42/gate4_preflight.py`）。
结论：**observed_residual 不能**与 anchor 在同一物理目标下比较（跟踪滞后直接进入标签）；
若以后要测它，必须配 `execute_k=1`，作为独立阶段（不放进本轮 300+300）。

## 2. 归一化空间：表示会同时改变数值分布

| 表示 | 关节 qspan / DROID | 出界比例（±1） |
| --- | --- | --- |
| anchor（16 步累计） | **0.175–0.761** | 0 |
| target_increment | **0.023–0.057** | 0 |
| observed_residual | 0.024–0.061 | 0 |

即：`target_increment` 的标签在 DROID 尺度下只有 2.3%–5.7%，与之前被错误归给"训练标签"的单步 2.4–6.1% 同量级。
**所以 A/B 表示的对照必然是"表示 + 对应目标分布"的联合效果，不能单独归因于 DROID 先验**
（与审计 §三.5 一致）；本轮不宣称能分离这两者。

## 3. 接触与重推理不在同一时刻（修正上一轮）

`Gate 4 preflight --rollout` 输出（`gate4_contact_timeline_B90_ep0*.json`）：

| episode | 首次接触（子步） | 首次接触后的重推理 | 接触 burst 数 | 接触子步 |
| --- | --- | --- | --- | --- |
| B90 ep00 | tick 9 / k 6（执行后测得） | tick 10 / k 0 | **15** | 51 |
| B90 ep01 | tick 25 / k 1 | tick 26 / k 0 | 1 | 2 |

B90 ep00 的 51 个"接触子步"分布在 **15 个离散 burst** 里，不能当作一段连续接触期；
且首次接触发生时，产生该 chunk 的观测（tick 9 开头）**还没有接触**。

## 4. 修正后的三层指标 + 2×2 分层（`substep_fk_pairing.json`）

- `target_error_i = FK(q_target_i) − FK(q_before_i)`（**目标偏差**，不是路径）
- `command_path_i = FK(q_target_i) − FK(q_target_{i−1})`（相邻命令之间的路径，跨 chunk 连续）
- `realized_path_i = FK(q_after_i) − FK(q_before_i)`（实际执行路径）

分层 = 「chunk 生成时机（首次接触之后？）× 执行本子步时是否有接触」：

| episode | 组 | n | 命令路径 Δ(x,y) | 实际路径 Δ(x,y) | 物体 Δ(x,y) | 命令 +X 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| B90 ep00 | 接触前生成 · 无接触 | 78 | **+0.186, +0.047** | +0.186, +0.047 | 0, 0 | 0.59 |
| B90 ep00 | 接触前生成 · 本次才接触 | 2 | −0.022, −0.000 | −0.021, +0.004 | 0, 0 | 0.50 |
| B90 ep00 | 接触后生成 · 无接触 | 231 | **−0.294, +0.034** | −0.317, +0.060 | +0.015, +0.004 | 0.49 |
| B90 ep00 | 接触后生成 · 接触中 | 49 | **−0.078, +0.191** | −0.057, +0.157 | −0.000, **+0.050** | 0.47 |
| A90 ep00 | 接触前生成 · 无接触 | 95 | +0.136, −0.111 | +0.184, −0.093 | 0, 0 | 0.57 |
| A90 ep00 | 接触后生成 · 无接触 | 256 | −0.088, **+0.336** | −0.115, +0.330 | −0.013, −0.005 | 0.54 |
| A90 ep00 | 接触后生成 · 接触中 | 8 | −0.005, +0.000 | +0.013, +0.001 | +0.008, +0.001 | 0.38 |

**读法（修正上一轮的错误表述）**：
1. 上一轮"接触期命令 −0.343 m 后退"是把**目标偏差**求和当成路径，**错误**；按路径口径，
   接触内只有 −0.078，大后退（−0.294）发生在**接触之后的无接触阶段**；
2. 接触前的接近确实是 +X（+0.186）；
3. 首次接触那一下发生在"接触前生成的 chunk"里（开环，模型无法反应），随后重推理才看到接触；
4. 接触后模型**先退开**，再以 +Y 为主重新靠近（+0.191），于是方块只得到 +0.050 的 Y；
5. 命令路径与实际路径几乎相等（B90 ep00：+0.186 vs +0.186、−0.294 vs −0.317），
   **控制跟踪不是主因**；真正的跟踪误差 `max_q_track_err` = 0.0053–0.0926 rad（B90 ep1 最大），
   不是此前误读的 1e-8。

## 5. 夹爪标签口径（审计 P1）已修正

- 我上一轮新增的 `build_anchor_horizon_arrays` 把夹爪写成常数 `g_action[t]`，**这是 bug**，
  已改为逐行 `g_action[t:t+horizon]`；单测覆盖"接近→闭合"窗口（row0=1.000、row6=0.425）。
- 真实训练用的 `train_lora.make_item` 一直是逐行 `chunk[:,7]/0.04`，**step901 未受影响**；
  但若不修，下一轮把 helper 当权威监督会损坏 8.09%（204/2521）含未来夹爪变化的窗口。

## 6. Gate 1 回归：两种表示都要通过同一 Adapter + 物理控制器

`run_gate1_regression.sh`（A90 ep5/6、B90 ep0/1，measured 锚点）：

| 表示 | adapter 回放严格成功 | 目标 vs 专家 | 实际最大跟踪误差 | 物体/接触 |
| --- | --- | --- | --- | --- |
| anchor / fixed | **4/4** | 5.6e-17 rad | 0.0053–0.0926 rad | 与专家回放逐项一致 |
| target_increment / recursive | **4/4** | **0.0 rad** | 同上 | 与专家回放逐项一致（含墙面 669/922 步） |

→ 两种表示都能把专家命令序列**精确**送进同一执行链；因此 Gate 4 的 A/B 差异只能来自
"模型在这两种目标分布上学到什么"，而不是执行器差异。这也满足审计 §三.2 的验收要求
（命令序列一致 + 严格判据，不只看最终状态）。

## 7. 建议的 Gate 4 设计（仍**未启动**）

| 项 | Group A | Group B |
| --- | --- | --- |
| 表示 | anchor（`A[t+k]−S[t]`） | target_increment（`k=0: A[t]−S[t]`；`k>0: A[t+k]−A[t+k−1]`） |
| 执行器 | `fixed` | `recursive` |
| 其余 | 同一 base `pi05_droid`、同 LoRA、同 batch/optimizer/LR/数据顺序/随机源、horizon 16、DROID stats、state、夹爪、32D loss、安全阈值（0.30 rad 限速、拒绝即停） | 同左 |
| 预算 | 300 **实际 optimizer update** | 同左 |
| 必须记录 | 权重与 stats SHA256、51/51 LoRA 张量、每组 normalized q01/q99 与饱和率 | 同左 |
| 离线评测 | 统一换算成**物理关节目标与 EE 位移**；按技能 ×（接触前/后生成）× k × 夹爪时序分层 | 同左 |
| 物理评测 | 仅当离线有可分差异；同 checkpoint 选择规则、同策略种子、同目标、同安全约束；**专家回放不算自主成功** | 同左 |

限制（写进结论）：A/B 只能得出"表示及其目标分布的联合效果"；若 B 只是 normalized 空间的 7D MAE 变好、
而 B90 的 EE +X 行为不变，则**不能**宣称学会 B90。

## 8. 任务 1/3 落地（2026-11-04 第二轮）

### 8.1 FK 纯净性（审计 §四）

`substep_fk_pairing.py` 的 FK 原来在 `resetJointState` 后调用 `sim.step()`——那会真实推进 240 Hz 物理，
把刚重置的关节再按旧电机目标推回去。已改为纯 FK：

```python
p.resetJointState(robot, j, q[j], targetVelocity=0.0)
p.getLinkState(robot, 11, computeForwardKinematics=True)   # link 11 = panda_grasptarget，与 sim.pose() 一致
```

回归测试 `scripts/exp42/test_fk_purity.py`：**PASS**
（10 次 FK 调用后 `sim.steps` 240→240、自由落体方块 z 漂移 0、顺序无关、与 `sim.pose()` 差 0.000e+00）。

重算后的配对数字（`substep_fk_pairing.json`）与旧报告相比最大差 **0.006 m**，结论不变：

| episode/组 | 旧（link6 + sim.step） | 新（link11 + 纯 FK） |
| --- | --- | --- |
| B90 ep00 接触前 | +0.186, +0.047 | +0.187, +0.048 |
| B90 ep00 接触后无接触 | −0.294, +0.034 | −0.294, +0.038 |
| B90 ep00 接触中 | −0.078, +0.191 | −0.080, +0.184 |
| A90 ep00 接触后无接触 | −0.088, +0.336 | −0.089, +0.339 |

### 8.2 训练/推理接线（审计 §六.1）

| 项 | 状态 |
| --- | --- |
| `train_lora.make_item(..., label_mode=)` | 已支持 `anchor` / `target_increment`（夹爪标签始终逐行未来值） |
| `build_batch(..., label_mode=)` | 已透传 |
| CLI `--label-mode` | 已加 |
| checkpoint `manifest.json` | 每次保存写 `label_mode / anchor_reference / horizon / loss_dims / steps_done / batch / seed` |
| `eval_rollouts.py --label-mode` | 自动选 `fixed`/`recursive`；`--ckpt-manifest` 会校验 `label_mode/horizon`，不一致直接退出 |
| `action_adapter` 注释 | 已注明 `observed_residual` 不能配 recursive（重建误差 0.379 rad） |

### 8.3 训练前闭环契约测试（`contract_sweep.py`）

B90 ep0 的 4 个窗口 × 专家真值，两种表示各自走 `Normalize → Unnormalize → 解码 → 物理目标`：

| 归一化噪声 σ | anchor/fixed 物理目标误差 | increment/recursive | 触发裁剪值数 |
| --- | --- | --- | --- |
| 0 | **0.000** | **0.000** | 0 / 0 |
| 0.01 | 0.021 | 0.055 | 0 / 0 |
| 0.05 | 0.103 | 0.272 | 0 / 0 |
| 0.10 | 0.202 | 0.520 | 0 / 0 |
| 0.20 | 0.404 | 0.990 | **7**（仅 increment） |

- 归一化往返误差 **1.1e-16**（两种表示相同）；
- 噪声在 increment 上按 `Σϵ_j` 累积，物理目标误差约为 anchor 的 **2.4–2.6×**，
  与审计 §五 的推导一致；
- σ=0.2 时 increment 表示会先撞上安全裁剪（7 个值），anchor 不会。

→ 契约测试**通过**（数值可逆、无专家标签裁剪、噪声传播量级已知）；
A/B 的因果结论仍只能写"表示的联合效果"，因为两种表示的目标分布差 7–8 倍（§2）。
