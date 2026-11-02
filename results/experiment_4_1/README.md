# 实验 4.1：π₀.₅ 多状态原子技能数据集与严格物理评价

本目录是实验 4.1 的全部工件。**实验 1–3 与 4.0 的历史工件未被修改或覆盖。**

- 基准 commit：`6425025`（4.0 最终提交）；采集时的实际 commit 见 `artifact_manifest.json`
- 相机：`exp4.0-camera-v3`，SHA256 `418bad2f33ef5a9e`
- 动作契约：`exp4.1-action-v1`（20 Hz、12 物理步/控制周期、绝对关节目标 → π₀.₅ 增量语义）

## 目录

| 路径 | 内容 |
| --- | --- |
| `environment_and_action_contract.md` | **Gate 1**：环境、240/20 Hz 时序、动作契约、可逆转换、相位检查方法 |
| `scene_space/goal_registry.json` | 9 个 A 放置目标 + 3 个 B 墙面目标 + 3 个 B 物体起点 + 2 个机器人起点，含 pilot 可达性 |
| `scene_space/b_start_states.json` | B 启动条件（物体 + 机器人）与"预配对"关系 |
| `evaluation/strict_eval_config.yaml` | 严格评价器冻结配置（阈值 + 依据） |
| `evaluation/evaluator_tests.json` | **17 个真实 PyBullet 正负案例**结果 |
| `evaluation/calibration_report.md` | 阈值校准依据、与旧 v2.2 的差异、已知局限 |
| `pilot/attempts.csv` | Pilot 全部尝试（成功与失败都保留） |
| `pilot/pilot_summary.json` | 按目标/条件分组的成功率、终态唯一性 |
| `dataset/records/*.json` | 每条尝试的完整记录（含接触摘要、检查项、指标） |
| `dataset/quality_audit.csv` | 采集阶段的逐条质量审计 |
| `dataset/diversity_audit.json` | 物体/机器人/联合终态去重计数（带容差） |
| `dataset/dataset_statistics.json` | state / 绝对动作 / 增量动作 / 夹爪的分布统计（供 4.2 归一化决策） |
| `dataset/split_manifest.json` | **预注册** split（按目标/条件分组，不按帧） |
| `dataset/readback_verification.json` | 写盘→读回→特征校验结果 |
| `dataset/fps_feasibility.json` | 15 Hz vs 20 Hz 的专家可行性对照 |
| `preflight/` | **验收后 Data Preflight**：夹爪标签修复、15Hz 比例纠正、评价器 **v1.2**（双指同时接触/滑移/角速度）与重放抽样、OpenPI transforms 回环、split 冻结、`n_success` 汇总修复、180/180 视频解码 |
| `reports/experiment_4_1_report.md` | 最终报告（RQ1–RQ10 + Observed/Supported/Not Verified/Blockers） |

数据集本体（Parquet + 视频，体积大）**不入 Git**，保存在服务器：
`/scratch/lijh/exp40/repo/data/exp41/{A90,B90}`（LeRobot v2.1，20 Hz），
元数据与统计在 `dataset/` 下，可用 `scripts/exp41/verify_dataset.py` 复核。

## 复现

```bash
# 1) 严格评价器（17 个物理正负案例，Gate 2）
python scripts/exp41/evaluator_tests.py

# 2) 场景/目标注册表
python scripts/exp41/make_registry.py

# 3) Pilot（不录帧，只验证可行性与多样性）
python scripts/exp41/collect41.py --mode pilot --per-condition 3 --out results/experiment_4_1/pilot

# 4) 有界采集（录帧写 LeRobot）
python scripts/exp41/collect41.py --mode collect --target-a 45 --target-b 45 \
    --out results/experiment_4_1/dataset --data data/exp41

# 5) 回环验证 + 多样性/split/统计
python scripts/exp41/verify_dataset.py --data data/exp41 --out results/experiment_4_1/dataset
python scripts/exp41/audit_dataset.py --data data/exp41 --out results/experiment_4_1/dataset

# 6) 15 Hz 可行性
python scripts/exp41/fps_feasibility.py --out results/experiment_4_1/dataset
```

## 一句话结论

见 `reports/experiment_4_1_report.md`。要点：
**A90 在 9 个目标区域上 45/45 严格成功，去重出 31 种真实终态；B90 在 3 个墙面目标区、
2 种机器人启动构型下 45/62（72.6%）真实推墙成功（接触 628–875 个物理步、法向力 ≥25.7 N）；
严格评价器 17/17 物理案例通过，能识别近墙不接触、初始贴墙、顶翻、方向错误、机器人代推等失败；
但当前 B 专家不能向 −Y 横移，所以"同一起点按目标切换动作"的条件泛化尚未实现。**
