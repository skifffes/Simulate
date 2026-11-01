# Skill Composition Stage Handoff (实验 3.1–3.5)

## 核心问题

在冻结 SmolVLA D2、仅各自训练原子技能的条件下，真实 A90 终态能否通过物理规划进入真实 B90 训练起点，并在那里可靠执行？

本阶段结论是能力边界，不是新算法。

## 已完成

- 3.1：60 个 B 起点审计；按物体位置差三分位 Near/Medium/Far；预注册 7 个目标（含 ep064 anchor）。
- 3.2：7 目标 × A/B/C × 5 配对 seed = 105 次；全部规划/执行/收敛；B/C 35 对状态与行为完全一致；A/B 场景差 1.53–17.88 mm。
- 3.3：复用 3.2 做场景改变的首动作/接触诊断；方向依目标，非单调。
- 3.4：5 个 A90 快照只有 1 个唯一终态，`A_TERMINAL_DIVERSITY_LIMITED`。
- 3.5：汇总漏斗与失败分类。

## 可信结论

1. B/C 在 7 个目标上无额外 transition 影响；两者 success 均 28/35，逐 seed 相同。
2. 规划可达性在本组目标上不是瓶颈；7/7 到达 qB，最大 q 误差 1.55e-5 rad。
3. A/B 场景改变可以改变 D2 行为，但方向依目标；不能声称物体越远越差。
4. 参考 B 本身不稳定（ep074 1/5，ep082 4/5）。
5. v2.2 success ≠ wall contact；本阶段 wall contact 稀少。

## 不可信/已作废

- 旧 AtoB 3/9 vs Bref 6/9：旧种子未配对，不能作 transition 因果证据。
- v2 continuous 5/5：目标 q 未到达（0.671 rad），已作废。
- v2.2 成功不等于真实推墙成功。

## 关键路径

- 3.1/3.2 数据：`results/experiment_3_2/`
- 3.3：`results/experiment_3_3/`
- 3.4：`results/experiment_3_4/`
- 3.5：`results/experiment_3_5/final_stage_report.md`
- 受控 v4：`results/experiment_3_controlled/`
- 代码：`scripts/exp31_audit.py`、`exp31_select.py`、`exp32_episode.py`、`exp32_analyze.py`
- 服务器原始 rollout：`/scratch/lijh/simulate_vla/results/experiment_3_2/rollout/`

## 当前限制

- 仅 1 个唯一 A 终态；只能研究单 A → 多 B。
- B Reference 是重建场景，不是原始训练动态/视频。
- 7 个目标、5 seeds；目标级样本小。
- 直线规划只做碰撞采样，不是动态安全证书。
- wall contact 日志未做独立力学校准。

## 下一步

优先换真实 B 起点、分析物体位置/EE–物体相对几何与首动作/接触的关系；若需要更强 VLA，应先区分 B90 原子策略失败与场景/规划失败，而不是直接归因于模型能力。
