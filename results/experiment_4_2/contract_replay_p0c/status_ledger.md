# 状态台账：做完了什么 / 没做完什么（2026-11-04 18:20）

本文件是**按用户要求建立的常驻清单**：每个结论都标注证据文件与状态；
未完成项不再用"应该/可能"带过，直接写"未做"。

## 0. 结论修订：早期"训练无效"的判断已被推翻

| 早期结论（何时） | 现在的状态 | 依据 |
| --- | --- | --- |
| 闭环 0/8、0/4，模型"完全没学会" | **已修订**：主因是**训练不充分**（300 步 = 0.48 epoch、LR 仍处 warmup 上升沿） | `learning_curve.md`：300/600/900/1200/1800 五点的幅度 0.52→0.24→0.077→0.059→0.048、余弦 −0.19→0.85 |
| "B90 接触期反向、往 +Y" | **已修订**：早期版本在"贴着推不动"的状态，1800 步已能定向 +X 推动 0.153 m、横向漂移 1.8 cm | `curve1800_rollout.md` |
| "动作表示（anchor vs increment）无可分效应" | **结论作废**：A/B 各 300 步处于 warmup，不能用于表示结论；需 ≥1200 步重做 | `ab_repr_training.md` + `learning_curve.md` |
| "归一化压窄信号" | **未证实且已弱化**：真实 16 步锚点标签跨度是 DROID 的 17.5–76%，clip 比例 0、往返误差 1e-16 | `action_contract_audit.json` |

**仍然成立**的结论：
- 执行链路三处契约错误（锚点、夹爪单位、累计裁剪）已修复，专家回放 8/8 严格成功、目标误差 1e-17（`expert_replay_results.md`）；
- Adapter 拒绝检查与逐周期限速已实现（`test_contract_p0c.py` 16 项；实测拒绝/限速均为 0）；
- 过拟合 8 窗口可到 loss 0.0064、采样 MAE 0.025 rad（`gate4_overfit.md`）→ pipeline 能学。

## 1. 已做完（有证据、可复现）

| # | 事项 | 证据 |
| --- | --- | --- |
| 1 | 契约缺陷 1/2/3 修复 + 回归 | `src/exp40/action_adapter.py`、`verify_anchor_and_gripper.py`、`test_contract_p0c.py`（16/16） |
| 2 | 专家回放（两种表示 × 2 技能 × 2 episode） | `expert_replay_results.csv`：8/8 严格成功 |
| 3 | checkpoint/资产核查（assets md5 = 官方 DROID、51/51 LoRA） | `server/exp42/check_assets.py` 输出 |
| 4 | 16 步整段评价（anchor 均匀 vs increment 累积 9.36 rad） | `full_chunk_eval.json` |
| 5 | 过拟合判别实验（8 窗口 / 150 步） | `overfit_log.json`、`overfit_eval_chunk.json` |
| 6 | warmup 事实 + LR 覆盖修复 | 官方 `CosineDecaySchedule(warmup=1000)`；`train_lora.build_config` |
| 7 | 学习曲线 300/600/900/1200/1800 | `learning_curve.md`、`expert_state_analysis_curve{300,600,900,1200,1800}.json` |
| 8 | 阶段拆分（B90 四段）+ 唯一状态统计（30 帧 / 10 窗口起点 / 3 episodes） | `expert_state_picks_v2.json`、`gate4_overfit.md` §5 |
| 9 | 真·主动推动帧（8 帧）评测 | `expert_state_analysis_push_active.json` |
| 10 | 1200 步闭环（k=8 与 k=1） | `curve1200_rollout.md`：0/4；B90 接触 845 步、advance 0.015 m |
| 11 | **1800 步闭环** | `curve1800_rollout.md`：0/4，但 **B90 ep0 advance 0.153 m**、漂移 1.8 cm |
| 12 | `/scratch` 清理（+76 GB，166 GB 可用）与保留项自检 | `server_cleanup.md` |
| 13 | 安全脚本替换（固定 PID kill → 属主+命令校验、默认 dry-run） | `server/exp42/kill_gpu_procs_safe.sh` |

## 2. 没做完（明确列出，不含糊）

| # | 未完成事项 | 影响 | 可补做成本 |
| --- | --- | --- | --- |
| 1 | **1500 步 checkpoint 未评测**（LoRA 已保存） | 曲线缺一个点，不影响结论方向 | 合并+推理 ≈ 4 min |
| 2 | **1800→2400 步未训练** | 不知道是否还能继续提升、能否推到墙 | ≈ 50 min |
| 3 | **B90 墙面接触仍未达成**（1800 步 ep0 只差这一项） | B90 原子技能仍未 PASS | 需更长时序或更长 tick |
| 4 | **A90 抓取阶段未诊断**（两条 rollout 均 0 接触） | A90 原子技能仍未 PASS | 离线分析夹爪目标时序 ≈ 15 min |
| 5 | **动作表示对照未在 ≥1200 步重做** | "表示是否有影响"仍未定论 | 2×1200 步 ≈ 90 min |
| 6 | 多 seed 曲线（仅 seed 0） | 无方差估计 | 每点 ≈ 35 min |
| 7 | 全量 61 episode 的离线评测（只有 30 个状态） | 代表性有限 | 需要批量推理 |
| 8 | A→B 零样本组合实验 | 按约定在原子技能通过前**不启动** | — |
| 9 | 服务器 GPU1–4 被他人占用（`/scratch/haozhe/...`，uid 1085） | 只有 GPU0 可用，未触碰他人进程 | — |

## 3. 汇报规则（自本轮起）

1. 每次回复分两栏：**已完成（含证据路径）** / **未完成（含原因与补做代价）**；
2. 任何"改善/变好"必须给同口径的数字对比，不写"明显提升"这类无数字表述；
3. 凡是"未运行"的，一律写"未运行"，不写成"预计/应该"；
4. 训练类结论必须同时给出：updates、epoch、LR 相位、checkpoint 文件名、评测集与 seed。
