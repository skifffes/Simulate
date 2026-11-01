# 实验 4.0：π₀.₅ 基座迁移与 PyBullet 原子技能环境适配

本目录是实验 4.0 的全部可复现工件。**实验 3 的模型、配置、CSV、轨迹与报告未被修改**（见文末隔离说明）。

- Simulate commit（实验开始时）：`e0483131afda5f8ca28f37e1aaaec29adc7fe97f`
- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`
- 结论：见 [`experiment_4_0_report.md`](experiment_4_0_report.md)（PASS / PARTIAL / BLOCKED 三选一，必须基于实测）

## 目录结构

```text
results/experiment_4_0/
    README.md                    本文件
    environment_report.md        A40 环境、glibc 2.17 依赖冲突、网络实测与复现命令
    compatibility_matrix.csv     D2 与 π₀.₅ 的逐项接口对照（11 个组件）
    checkpoint_manifest.json     pi05_droid 每个文件的路径/大小/md5（Orbax OCDBT 分片）
    model_smoke/                 Level 1：模型独立推理记录、输入张量、动作样本
    observation_checks/          Level 2/3：真实仿真观测适配记录（含原始图与实际发送图）
    action_checks/               Level 3：动作适配器单元测试（32 项判据）
    closed_loop_smoke/           Level 4/5：闭环轨迹 CSV、安全停止记录、A90/B90 接口检查
    experiment_4_0_report.md     最终报告（Observed / Supported / Not Verified / Blockers）
    artifact_manifest.json       本目录所有工件的清单与 sha256
```

## 代码位置（源码，不在本目录）

| 文件 | 作用 |
| --- | --- |
| `src/exp40/droid_contract.py` | π₀.₅ DROID checkpoint 的真实输入/输出约定，逐条带 openpi 源码行号 |
| `src/exp40/observation_adapter.py` | PyBullet 观测 → DROID 键名 payload（含 sha256/统计审计） |
| `src/exp40/action_adapter.py` | 8 维策略动作 → 关节目标 + 夹爪目标（含拒绝规则，不做无限制裁剪） |
| `src/exp40/safety.py` | 关节限位 / EE 工作空间 / 单步幅度 / NaN / 碰撞 / 物体异常 / 超时 |
| `src/exp40/chunk.py` | action chunk 缓冲、重推理时机、指令切换与 reset 语义 |
| `src/exp40/policy_client.py` | 官方 websocket + msgpack-numpy 协议客户端（放宽 keepalive） |
| `scripts/exp40/serve_policy.py` | 策略服务器（checkpoint / random 两种模式 + 输入输出审计钩子） |
| `scripts/exp40/level1_model_inference.py` | Level 1 |
| `scripts/exp40/level2_offline_sim_obs.py` | Level 2 |
| `scripts/exp40/level3_adapter_checks.py` | Level 3 |
| `scripts/exp40/level4_closed_loop.py` | Level 4 |
| `scripts/exp40/level5_env_interface.py` | Level 5 |
| `server/exp40/*.sh`, `server/exp40/*.py` | 服务器侧环境、下载、运行脚本与全部探测记录 |

## 一句话复现

```bash
# 策略侧（A40，GPU2）
bash server/exp40/run_server.sh checkpoint 2 8000
# 仿真侧
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/level4_closed_loop.py --scene B90 --ticks 20
```

完整步骤见 `environment_report.md` 第 6 节。

## 与实验 3 的隔离

- 本实验不读、不写 `results/experiment_2_*`、`results/experiment_3*` 下的任何文件；
- 不修改 `outputs/smolvla_*` 下的 D2 checkpoint；
- 不修改 `src/vla/**`、`scripts/eval_multi_atomic_vla.py` 等历史代码，只以只读方式复用其中的场景与相机定义；
- 新代码集中在 `src/exp40/` 与 `scripts/exp40/`，历史目录零改动（可用 `git status` 核对）。

## 后续两轮补丁（相机可用性 / 接线 / RNG / CSV）

- `hotfix/`：相机 v2 冻结、夹爪与关节尺度证据链、15 Hz 时序、逐回合 seed 接线（29 条判据）
- `final_validation/`：投影矩阵修正、腕部固定外参 **camera v3**（SHA256 `418bad2f…`）、
  真实 Policy Server 输入审计、模型级 RNG 复现、A90/B90 15 Hz 闭环、CSV schema 修复
- `replay_fix/`：**最新（v2 有效）**——B90 专家回放改用**正确映射的 episode**
  （`episode_index=241 ↔ episode_id=61`，误差 9e-8 m），方块被真实推动 **18.8 cm**、
  **第 94 帧与墙接触**（404 物理步）；腕部 B90 6/6、A90 7/7 可见；mask 改为
  RGB/segvis/segids 三件套。详见 [`replay_fix/replay_fix_report.md`](replay_fix/replay_fix_report.md)。

> **旧结论作废**：
> 1) `final_validation/camera_visibility_by_phase.csv` 的 B90 行与所有 `final_validation/**/*_mask.png`
>    （错误场景 + RGB 冒充 mask）；
> 2) `replay_fix/` 根下的 v1 工件（`replay_provenance.json`、`camera_visibility_corrected.csv`、
>    `contact_phase_validation.csv`、`camera_corrected_frames/`）——它们基于 `episode_index=64`
>    （实际属于 A0/A90 段）与错误的墙体猜测。**以 `replay_fix/v2/` 为准**，不得混合统计。
>    A90 的可见性结论不受影响（其场景本就正确），但 mask 需用 v2 版本。

**使用哪个相机**：训练/评测一律用 `src/exp40/camera_v2.py` 的 `CAMERAS_V3`，
`build_sim_scene()` 已强制该版本（缺配置即报错，不静默回落旧 `config2.CAMERAS`）。
`hotfix/` 里的 v2 与 4.0 原始记录均为历史对照，不要用于新数据。
