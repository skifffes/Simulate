# 实验 3.2：跨 B 起点的 A/B/C 受控连接

## 运行

```bash
python scripts/exp31_audit.py
python scripts/exp31_select.py
python scripts/exp32_episode.py --target ep079_f000 --group C --policy ... --out results/experiment_3_2/rollout --seed 0
bash server/run_e32.sh
python scripts/exp32_analyze.py
```

## 数据

- `b90_all_candidates.csv`：60 起点审计
- `candidate_selection.json`：预注册 7 目标
- `planning_results.csv`：7 目标规划
- `rollout_results.csv`：105 次
- `state_alignment.csv`：35 对 A/B 与 B/C
- `first_action_comparison.csv`
- `contact_metrics.csv`
- `summary_by_target.csv`
- `scene_effect.csv`
- 报告：`experiment_3_2_report.md`

## 结果

- 7/7 规划、执行、收敛；max q 误差 1.55e-5 rad。
- B/C 35 对状态与行为完全一致；B/C 均 28/35。
- A 31/35；A/B 场景差 1.53–17.88 mm，方向依目标，非单调。
- wall contact A1/B2/C2；v2.2 成功不等于墙接触。
