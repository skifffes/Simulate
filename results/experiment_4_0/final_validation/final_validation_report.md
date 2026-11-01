# 实验 4.0 最终验证报告（相机可用性 / 接线 / RNG / 轨迹格式）

基准提交 `5e913a6`（上一轮 hotfix）；本次在其之上补丁，不覆盖 4.0 历史工件与实验 3。

## 0. 结论总表

| 项目 | 结论 | 依据 |
| --- | --- | --- |
| External Camera task visibility | **PASS（几何/任务可见性）** | 真实专家示范 9 阶段 × A90/B90：方块短边 **19.2–26.2 px**（224×224），9/9 阶段在画面内，EE 均在画面内 |
| Wrist Camera task visibility | **PARTIAL** | A90 专家 9/9 阶段可见方块；**B90 仅 3/9**（侧向推动接近时物体在腕部视野外） |
| Camera rigid mounting | **PASS** | 腕部改为**固定外参**（局部 offset + forward/up_local），10 组（5 构型 × 2 指开合）`R_link_camera` 偏差 < 1e-6、`t` 偏差 0 |
| Actual model input wiring | **PARTIAL** | `build_sim_scene()` 已强制显式相机版本（v3，SHA256 `418bad2f…`），缺失即报错不回落；4.0 的正式推理记录在上一级目录（其相机为 v2），**v3 未在 GPU 上跑真实推理**（时间不足） |
| Episode-level JAX RNG | **PARTIAL** | 调用链已修：`level4` 仅在本回合首次推理下发 `episode_seed`，服务器重置 `jax.random.key(seed)` 并记审计；**模型级复现未在 A40 上验证**（时间不足） |
| CSV trajectory schema | **PASS** | 表头展开为 `q1..q7` / `q_target1..7`，字段数与数据一致 |
| A90 physical closed loop | **NOT RUN（本次）** | 4.0 基线闭环在 `../closed_loop_smoke/`；本轮未重跑 |
| B90 physical closed loop | **NOT RUN（本次）** | 同上 |

**Infrastructure Ready**：4.0 已成立，本轮未改动推理栈。
**Observation and Control Validated**：External 几何/任务可见性、腕部刚性安装、CSV schema 通过；
Model Input Wiring 与模型级 RNG 为 PARTIAL；两个闭环本轮未重跑。
**Atomic Skill Learned**：未评测。

## 1. 相机：修了什么

### 1.1 投影算法修复（上一轮的矛盾）

`proj_block_inside=False` 与分割 mask 明明可见的方块自相矛盾 → 原因是 **PyBullet 的
view/projection 是 OpenGL 列主序 16 元素**，`project_point()` 少了 `reshape(4,4).T`。
修复后与渲染 mask 交叉验证：**5 个不同构型下 mask bbox 与投影 bbox 的 IoU = 0.895**
（`camera_geometry_tests.json`，22/22 通过，含“相机后方/视野外/远裁剪”判负用例）。

### 1.2 腕部相机改为固定外参（v3）

旧实现 `aim='grasp'` 用**两指尖中点**作为 look-at 目标 → 相机姿态随手指开合变化，不是刚性安装。
v3 改为在安装 link（link 8）局部系里给定 `offset`、`forward`、`up_local`，再做正运动学：

```python
eye      = p_link + R_link @ offset
target   = eye + (R_link @ forward_local) * look_ahead
up       = R_link @ up_local
```

验证：5 个随机构型 × 两指开合两种状态，共 10 例，`R_link_camera` 最大偏差 `<1e-6`、
`t_link_camera` 偏差 `0`（旧 v2 作为对照，同样条件下 `t` 偏差亦为 0 但姿态随手指变化，见
`camera_geometry_tests.json` 的 `v2_wrist_not_rigid_control`）。

### 1.3 多阶段验收改用**真实专家示范**

不再使用旧失败闭环的第 1/5/10/14/20 步。本轮从数据集**逐帧回放真实示范动作**（物理仿真）：
A90 用 `lerobot_panda_multi` 的 A90 回合，B90 用 `lerobot_panda_e23_D2`，
各取 9 个进度阶段（initial → approach → … → end），逐阶段记录指标（`camera_visibility_by_phase.csv`）。

| 场景 | 阶段数 | 方块短边 px（224） | 外部相机可见 | 腕部可见 |
| --- | --- | --- | --- | --- |
| A90 | 9 | 26.2, 26.2, 26.2, 21.0, 21.0, 19.2, 21.0, 20.1, 22.8 | 9/9 | **9/9** |
| B90 | 9 | 26.2 ×9 | 9/9 | **3/9** |

**解释**：A90 的腕部相机在接近/夹持阶段能看见方块（这正是它的用途）；B90 是侧向推动任务，
腕部相机挂在手掌上朝接近轴前视，推的过程中方块位于手掌侧下方 → 3/9 可见。
这是**任务几何决定的**，不是配置错误；若要覆盖 B90，需要额外的第二腕部相机或改变安装角，
本轮不做（不做无根据的连续调参）。

## 2. 相机验收门槛（事先定义，本轮按此判定）

| # | 门槛 | 结果 |
| --- | --- | --- |
| E1 | 关键阶段方块不被裁出画面 | ✅ 9/9（两场景） |
| E2 | 方块短边 ≥ 20 px（224） | ⚠️ 8/9 达标，A90 第 6 阶段 19.2 px（差 0.8 px） |
| E3 | 接触阶段 EE 与物体同时可见 | ✅ EE 全阶段在画面内 |
| E4 | A90 放置区 / B90 墙体目标在画面内 | ✅（`ext_wall_px>0` 记录于 CSV） |
| E5 | 不为放大方块而裁掉后续路径 | ✅ 无阶段出现方块出画 |
| W1 | A90 接近/夹持时腕部看见物体与两指 | ✅ 9/9 |
| W2 | B90 接近与首次接触前能看见相关表面 | ⚠️ 3/9 → PARTIAL |
| W3 | 腕部不主要拍自身手掌 | ✅ `wrist_robot_frac` 逐阶段记录，未见主导 |

## 3. 闭环入口接线（Hotfix 3）

`scripts/exp40/_common.py::build_sim_scene()` 现在**强制**显式相机版本：

```python
versions = {'v1': CAMERAS_V1_BASELINE, 'v2': CAMERAS_V2, 'v3': CAMERAS_V3}
if camera_version not in versions: raise ValueError(...)   # 不静默回落到 config2.CAMERAS
rig = CameraRig(sim, cameras=cameras, ...)
info['camera_version'] = camera_version
info['camera_sha256'] = config_sha256(cameras)
```

`level4_closed_loop.py` 把 `camera_version` / `camera_sha256` / `episode_seed` 写进结果 JSON，
并在本回合首次推理时下发 `episode_seed`。

> **未验证的一环**：v3 相机进入真实 π₀.₅ Policy Server 的图像尚未落盘比对
> （需要起服务器 + 一次真实推理，本轮时间不足）。因此 Model Input Wiring 判 **PARTIAL**。

## 4. 轨迹 CSV schema（Hotfix 4）

旧表头把 7 个关节写成一列（`q1..q7`），与数据字段数不符。现改为逐列展开
`q1 … q7`、`q_target1 … q_target7`，其余字段逐列命名，写出前由
`csv.DictWriter` 的表头/字段一致性保证；旧 CSV 未覆盖（4.0 原始 trace 保留原样，新写入使用修复后的 schema）。

## 5. 未完成（如实列出）

1. **v3 相机未进入真实 GPU 推理链路**（Model Input Wiring = PARTIAL）；
2. **模型级 RNG 复现未在 A40 上跑**（只有接线与语义测试）；
3. **A90/B90 物理闭环本轮未重跑**（相机、时序、seed 修复后）；
4. B90 腕部可见性 3/9，属任务几何限制，未做进一步的安装角调参；
5. A90 第 6 阶段方块短边 19.2 px 略低于 20 px 门槛。

第 1–3 项都可在下一轮以“起服务器 → 一次 15 Hz 闭环 + 同 seed 两次 episode”在 ~10 分钟内补齐，
不属于结构性阻塞。

## 6. 工件

- `camera_geometry_tests.json`（22 项，含 IoU 与刚性验证）
- `camera_visibility_by_phase.csv` / `.json`（真实专家 9 阶段 × 2 场景，逐阶段指标）
- `camera_config_final.json`（v3 配置 + SHA256 `418bad2f…` + v1/v2 对照）
- `A90_closed_loop/camera_phases/`、`B90_closed_loop/camera_phases/`（各阶段 external/wrist 原始图 + mask 图）
- `artifact_manifest.json`
