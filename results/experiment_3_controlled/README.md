# 实验 3 受控诊断 v2：A/B/C 状态对齐、连续 transition 与冻结 D2

## 目标

在 **ep064 f000** 上先验证启动状态，再比较冻结 D2：

- **A snapshot**：B 参考场景（object_B）；
- **B snapshot**：A 场景 + 诊断性 qB 放置（object_A）；
- **C snapshot**：A 场景 + 恢复规划终态（30-step settle）；
- **B continuous**：A 场景 + qB 放置，同进程启动 D2；
- **C continuous**：A 场景 + 真实 q_A→q_B，**不恢复快照**，同进程启动 D2。

## 关键结果

- B/C snapshot 首推理 q L2 **9.1e-7 rad**，相对平移 **1.9e-4 mm**，相对旋转0°，top/wrist MAE **0**；首动作七关节 L2 **0**。
- B/C continuous 5 seeds **结果完全一致**（5/5；gap/tilt相同；首动作L2=0）；无 transition 残余。
- C snapshot **4/5**：seed4 tilt19.78°；B snapshot 5/5。首动作和状态几乎相同，属后续数值/动力学放大。
- A/B：物体差 **1.864 mm**、top MAE0.058；首动作 q L2 0.0084–0.0103 rad；均5/5。
- v2.2 成功 **24/25**，**wall_contact 0/25**；成功不是墙接触。
- 旧 AtoB 3/9 vs Bref6/9 未复现：旧 runner 未显式配对策略种子（A row seed53050005 vs B/C 64）。

## 文件

| 路径 | 内容 |
|---|---|
| `state_alignment.md` | 首推理状态对齐（完整 SE(3)） |
| `data/initial_states_v2.csv` | 25 次实际启动状态 |
| `data/state_alignment_v2.csv` | 逐 seed 差异（含 rel_pos/rel_rot） |
| `data/b90_rollout_results_v2.csv` | 25 次 rollout |
| `data/first_action_v2.csv` | 完整 q1..q7 + grip 首动作 |
| `data/observation_hashes_v2.csv` | 首推理 top/wrist SHA-256 |
| `summaries/group_summary_v2.csv` | 分组结果 |
| `summaries/first_action_diff_v2.csv` | 七关节动作 L2 |
| `e3c_v2_traces.tar.gz` | 25 次轻量 trace（policy/physics/contact；无视频） |
| `videos/AtoB_full.mp4` | A90→transition→**continuous C** B90 |
| `report.md` | 完整分析 |

## 运行

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python scripts/run_b90_controlled.py \
  --group C --mode continuous \
  --snap results/experiment_3_controlled/initial_states/C_transition.json \
  --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
  --out results/experiment_3_controlled/rollout_v2 --seed 0
python scripts/exp3c_analyze_v2.py
python scripts/exp3c_visualize_v2.py
```

## 限制

- 1 个 B 目标、1 个唯一 A 终态、5 个配对 seed。
- B reference 是重建场景，非原始训练快照。
- 控制器独立命令目标未完整保存；`q_target` 是实际保持 q，D2 action target 已存。
- C snapshot seed4 放大机制未识别。
- 规划为碰撞采样直线，不是连续时间动态可行轨迹。
