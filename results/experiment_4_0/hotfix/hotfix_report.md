# 实验 4.0 Hotfix：动作语义、相机可见性、时序与 RNG

基准提交 `557aaea`（本次 hotfix 在其之上，不回退任何 4.0 原始记录）。

| 模块 | 结论 | 依据 |
| --- | --- | --- |
| **Camera** | **PARTIAL**：外部相机已冻结为 v2（物体像素占比 0.71%→1.0%，机械臂从多数阶段不可见 → 5–22% 可见），A90/B90 多阶段均已在画面内；**但“训练域视觉分布对齐”未验证**（没有 DROID 真实图像对照） | `camera_before_after/`、`camera_config_final.json`、`hotfix_semantics_tests.json` A 组 7/7 |
| **Gripper** | **PASS**：观测与动作方向一致且为 `1=open, 0=closed`，归一化 `÷0.04`、阈值 0.5；证据来自官方 pi 内部夹爪空间（`aloha_policy._gripper_to_angular`：0→PUPPET_POSITION_CLOSE 0.01844，1→OPEN 0.05800）+ 真实仿真手指位移 | `hotfix_semantics_tests.json` B 组 7/7、`official_semantics_audit.md` |
| **Joint Action** | **PASS（保留原实现）**：适配器无任何额外比例因子，`q_target-q == action[:7]`（逐位相等），不存在 0.2 重复缩放；动作是“关节增量(rad/步)”，越界按官方 ±1 拒绝而非缩放 | `hotfix_semantics_tests.json` C 组 5/5 |
| **Timing** | **PASS（定义）/ PARTIAL（墙钟）**：闭环默认改为 15 Hz（每 tick 16 个 240 Hz 物理步，15 步 chunk = 1.00 s）；明确标记为**仿真时间闭环**，推理 155 ms 与 15 Hz 的关系已量化（execute_k=8 → 533 ms 预算），但未做异步/实时实现 | `hotfix_semantics_tests.json` D 组 5/5 |
| **RNG** | **PASS（接线与语义）/ PARTIAL（模型级复现）**：实现逐回合 `__episode_seed__` → 服务器重置 `jax.random.key(seed)`，seed 记入审计；模型级“两次独立 episode 逐位一致”因时间未在 GPU 上跑 | `hotfix_semantics_tests.json` E 组 5/5 |
| **Closed Loop** | **PARTIAL**：相机与时序改动后**没有**重跑物理闭环；4.0 基线闭环记录保留在 `../closed_loop_smoke/`（B90 21 步跑满预算、A90 14 步安全停止） | `../closed_loop_smoke/` |
| **Reproducibility** | **PASS**：相机配置版本化 + SHA256、脚本、种子、manifest、日志齐全 | `camera_config_final.json`、`manifest.json` |

**Infrastructure PASS**（模型加载/推理/通信/驱动仿真）在 4.0 已成立且本次未改动该部分。
**Semantic & Visual Adapter**：Gripper / Joint Action / Timing / RNG 通过，Camera 为 PARTIAL。
**Atomic Skill Capability**：未评测（按本次范围）。

## 本次实际改了什么（diff 摘要）

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `src/exp40/camera_v2.py` | 新增冻结相机配置 v2 + v1 基线 + SHA256 | 让 A90/B90 操作区与机械臂进入模型输入，并可版本化复现 |
| `scripts/exp40/hotfix_camera_audit.py` | 新增：候选视角 × 多阶段 × 分割 mask 指标 | 用客观像素计数而不是主观印象选相机 |
| `scripts/exp40/hotfix_camera_final.py` | 新增：before/after 多阶段 224×224 渲染与指标 | 交付对照证据 |
| `scripts/exp40/hotfix_semantics_tests.py` | 新增：A/B/C/D/E 五组 29 条判据 | 每条修复都配可反驳测试 |
| `scripts/exp40/level4_closed_loop.py` | 策略频率改为可配置（默认 15 Hz）；trace 增加 `chunk_roundtrip_s`；JSON 记录 timing 语义 | Hotfix D |
| `scripts/exp40/serve_policy.py` | `AuditingPolicy` 支持 `__episode_seed__` 重置 `Policy._rng` 并记入审计；记录容错 | Hotfix E |
| `src/exp40/policy_client.py` | `infer(obs, episode_seed=...)`，不改写调用方 obs | Hotfix E |

**未改动**：`src/vla/**`、实验 1–3 的任何配置/结果、`results/experiment_4_0/` 下 4.0 原始工件、模型权重。

## 相机改了什么、为什么

| | v1（4.0 基线，= 实验 2/3 的 `top`） | v2（冻结） |
| --- | --- | --- |
| target | (0.55, 0.21, 0.02) | (0.56, 0.13, 0.06) |
| distance | 1.05 | 0.70 |
| yaw / pitch | 50° / −52° | 50° / −35° |
| fov | 50° | 60° |
| 方块像素占比（5 阶段均值） | A90 0.71% / B90 0.71% | A90 1.03% / B90 1.00% |
| 机械臂像素占比 | A90 4.4% / B90 5.1% | A90 6.6% / B90 8.5% |
| 方块在画面内 | 5/5 阶段 | 5/5 阶段 |

选择理由：v1 是为“一眼看全桌面（含托盘）”（实验 1–3 的示教采集视角）设计的远景，物体只占 0.71%，
且多数阶段机械臂完全在画面外；v2 把视线中心移到**操作走廊**（物体→墙 / 物体→放置区），
拉近并抬高俯角、加宽 FOV，使方块与机械臂同时可见，同时墙、托盘、放置区仍在画面内（见
`camera_before_after/*_after_*_model224.png`）。没有使用任何物体真值瞄准，也未按任务成功率调参。

**仍未做到**：没有把 Simulate 渲染分布与 DROID 真实图像做定量对齐（缺 DROID 图像样本），
因此“几何可见性通过”成立，“训练域分布匹配”不成立。

## 如何复现

```bash
# 相机审计（候选 × 多阶段，保存全部图像与指标）
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/hotfix_camera_audit.py
# 冻结相机 before/after
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/hotfix_camera_final.py
# 五组判据（29 条）
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/hotfix_semantics_tests.py
```

## 已知缺陷（本次发现但未修）

1. 归档 `closed_loop_smoke/trace_*.csv` 的表头把 7 个关节写成一列 `q1..q7`，与数据行字段数不一致
   （本次审计脚本按位置解析绕过）。新代码写 CSV 时应展开列名。
2. `sim.collisions()` 只排除父子 link 对，两指尖互碰需在 `safety.py` 白名单（4.0 已加）。
3. 推理 155 ms/chunk 与 15 Hz 实时之间仍靠 `execute_k=8` 的预算兜底，未实现异步推理。
