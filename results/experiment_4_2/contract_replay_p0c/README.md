# 实验 4.2 Contract Replay + P0c 诊断

本目录回答一个问题：**修复后的执行链路能否准确复现专家物理行为？剩余失败该归因于谁？**

- 基准：`e128b35`（本轮修复与工件另提交，SHA 见 `artifact_manifest.json`）
- 结论速览：[`final_report.md`](final_report.md)（Q1–Q8）

## 结果索引

| 文件 | 内容 |
| --- | --- |
| `final_report.md` | **最终报告**：Gate 0–4 判定 + Q1–Q8（OBSERVED / SUPPORTED / HYPOTHESIS / CONTRADICTED） |
| `execution_contract_verification.md` | Gate 0：修复后链路核查（含本轮新发现的**第三个**契约缺陷与修复） |
| `expert_replay_results.md` / `.csv` | Gate 1：专家 vs Adapter 物理回放（8/8 严格成功，目标误差 1e-17） |
| `expert_vs_adapter_trace.csv` | Gate 1 逐 tick 轨迹（2916 行，含 anchor/专家目标/adapter 目标/实际 q/EE/物体/接触） |
| `b90_direction_analysis.md` | Gate 2：预测 EE / 实际 EE / 物体三层方向分离 |
| `b90_contact_phase_direction.json` | 接触阶段的模型 vs 专家方向统计 |
| `startup_supervision_audit.md` / `startup_supervision_audit.json` | Gate 3：90 条 episode 启动时序与监督可识别性 |
| `root_cause_matrix.md` | 13 项候选根因的证据与置信度 |
| `gate4_preflight.md` | **Gate 4 预检**：三种动作表示、接触时间线、命令路径口径修正、A/B 设计 |
| `gate4_dataset_preflight.json` | 真实 train 上三种表示的重建误差与归一化跨度 |
| `gate4_contact_timeline_B90_ep00/01.json` | 首次接触 / 首次接触后重推理 / 接触 burst 数 |
| `p0_substep_analysis.md` | **P0 补强**：拒绝硬阻断 / 限速 / 逐子步日志 + 分阶段配对（接触期命令反转） |
| `substep_fk_pairing.json` | 逐子步命令 EE / 实际 EE / 物体配对（按接触阶段切分） |
| `action_contract_audit.json` | 真实训练标签（16 步锚点）逐 k 统计 + DROID 归一化跨度（17.5–76%） |
| `contract_unit_tests.json` | 12 项契约单测（fixed/recursive、重锚、缩放裁剪、夹爪米、拒绝、限速） |
| `artifact_manifest.json` | 源码/数据/checkpoint 哈希与服务器路径 |

> `p0c_training_comparison.md` 与 `rollout_summary.csv` **未生成**：P0c 训练条件不满足（见 `final_report.md` Q7），
> 本轮也没有开展新的正式评测轮次（只有 4 条单变量诊断 rollout）。

## 复现命令

```bash
# Gate 0 契约单测（本地 CPU）
python scripts/exp42/test_contract_p0c.py

# Gate 1 专家回放（服务器 simulate venv，CPU）
python scripts/exp42/gate1_expert_replay.py --skill A90 --episodes 5,6 --label-anchor measured
python scripts/exp42/gate1_expert_replay.py --skill B90 --episodes 0,1 --label-anchor measured

# Gate 2 方向分析
python scripts/exp42/b90_direction_analysis.py --rollout-json results/experiment_4_2/rollouts_fixed_contract/B90_ep00.json
python scripts/exp42/b90_contact_phase.py --model-json .../B90_ep00.json .../B90_ep01.json \
    --expert-csv results/experiment_4_2/contract_replay_p0c/expert_vs_adapter_trace_B90_measured.csv

# Gate 3 启动审计
python scripts/exp42/startup_supervision_audit.py

# 模型 rollout（GPU1 起策略服务器）
bash server/exp42/run_eval_fixed.sh /scratch/lijh/exp42/run2/ckpts/params_nested 8142 1 2 45
bash server/exp42/run_execute_k1.sh        # execute_k=1 单变量诊断
```

## 版本

| 项 | 值 |
| --- | --- |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| checkpoint | step901（902 次更新），LoRA npz md5 `7cd19b1851a4955e6995fb2189422dcc` |
| 评价器 | `exp4.1-strict-v1.2` |
| 控制 | 20 Hz × 12 物理步；`execute_k=8`（另做 `k=1` 诊断）；`action_gain=1.0` |
| 契约 | `anchor_reference=fixed`、`gripper_mode=continuous`、第 k 步裁剪上限 `0.25*(k+1)` |
