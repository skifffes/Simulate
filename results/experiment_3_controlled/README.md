# 实验 3 受控诊断 v3：目标到达后的连续 transition

## 目标

在 **ep064 f000** 上比较：

- **A snapshot**：B 参考场景；
- **B snapshot**：A 场景 + 诊断 qB 放置；
- **C snapshot**：A 场景 + 恢复规划终态；
- **B continuous v3**：A 场景 + qB 放置，同进程启动 D2；
- **C continuous v3**：A 场景 + 真实 q_A→q_B，**不恢复快照**，收敛后同进程启动 D2。

## 关键结果

- v2 continuous 没到 qB（q 差 0.671 rad、EE 207 mm、q6 dq −17.64），**结论作废**。
- v3 continuous 用收敛条件：201 步（0.8375 s）到达 qB；q 误差 **5.2e-6 rad**，dq **5.7e-13**。
- v3 B/C continuous：q/EE/relative pose/action7 **全0**；top hash 相同；5 seeds 均 **4/5**，失败均 seed4 tilt 19.26°，gap/tilt 相同。
- Snapshot：A 5/5，B 5/5，C 4/5；C seed4 tilt19.78°。
- 25 次 v2.2 成功 **22/25**，**wall_contact 0/25**。
- A/B：物体差 1.864 mm，top MAE0.058，首动作七关节L2 0.0084–0.0103 rad；均5/5。
- 旧 3/9 vs6/9 未配对种子，不作因果证据。

## 文件

| 路径 | 内容 |
|---|---|
| `state_alignment.md` | snapshot + v3 continuous 启动对齐 |
| `data/initial_states_v3.csv` | v3 10 次启动状态（含 q_target=qB） |
| `data/state_alignment_v3.csv` | B/C v3 逐 seed 差异 + action7 |
| `data/b90_rollout_results_v3.csv` | v3 结果 + 收敛步数/误差 |
| `data/first_action_v3.csv` | 全精度 raw 首动作 q1..q7 |
| `data/observation_hashes_v3.csv` | 首推理 top/wrist hash |
| `summaries/group_summary_v3.csv` | B/C 分组 |
| `e3c_v3_traces.tar.gz` | 10 次轻量 trace |
| `videos/AtoB_full.mp4` | A90→transition→v3 continuous C B90 |
| `report.md` | v3 报告 |

## 运行

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src venv/bin/python scripts/run_b90_controlled_v3.py \
  --group C --snap results/experiment_3_controlled/initial_states/B_place.json \
  --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
  --out results/experiment_3_controlled/rollout_v3 --seed 0
python scripts/exp3c_analyze_v3.py
python scripts/exp3c_hashes_v3.py
```

## 限制

- 1 个 B 目标、1 个唯一 A 终态、5 seeds。
- B reference 为重建场景。
- 收敛阈值是工程阈值，只验证 ep064 f000。
- 控制器内部 target/queue 未完整审计；本轮 q_target=qB、raw action 已保存。
- v2.2 success ≠ wall contact；0/25 wall contact。
