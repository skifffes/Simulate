# 实验 4.0 Replay Fix（v2 为当前有效版本）

基准提交 `97de2c7`。本目录修复上一轮 `final_validation/` 与 `replay_fix/` 的两个缺陷：
**B90 专家回放用错 episode** 与 **分割 mask 被写成了 RGB**。

> **v1 工件（本目录根下 `camera_visibility_corrected.csv`、`contact_phase_validation.csv`、
> `camera_corrected_frames/`、`replay_provenance.json`）已作废**：它们基于
> `episode_index=64`（实际属于 A0/A90 段）与错误的墙体猜测，结论无效。
> 保留仅为历史对照，**不得与 v2 数据混合统计**。

## 结构

| 路径 | 内容 |
| --- | --- |
| `replay_fix_report.md` | **主报告**：根因、恢复误差、物理重放保真度、接触/阶段证据、相机结论、Q1–Q8 |
| `camera_decision.md` | 相机是否修改（结论：**不修改 v3**） |
| `episode_mapping.json` | 480 集混合数据集的逐集任务清单 + B90 首帧 q 与 `b_training_start_states.csv` 的精确配对（91/120） |
| `v2/B90/` | ep241 的 `provenance.json`、`tracking_fidelity.csv`、`contact_events.csv`、`camera_visibility.csv`、`segmentation_validation.json`、`frames/` |
| `v2/A90/` | ep120 同上 |
| `artifact_manifest.json` | 全目录工件与 sha256 |

## 两句话结论

1. **B90 的失败是索引映射错误**：`episode_index` 与 `episode_id` 不是一回事。换成正确配对
   （`episode_index=241 ↔ episode_id=61`）后，方块被真实推动 **18.8 cm**、**第 94 帧与墙接触**
   （404 个物理步），`wall_contact_verified` 这次有证据。
2. **相机全部保留 v3**：B90 6/6、A90 7/7 阶段腕部可见目标；外部相机 6/6、7/7 拍到墙。
   mask 改为 RGB / segvis / segids 三件套，`rgb_sha256 != segvis_sha256` 全部成立。

## 一条命令复现

```bash
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/replay_fix_v2.py --task B90 --episode-index 241
/scratch/lijh/simulate_vla/venv/bin/python scripts/exp40/replay_fix_v2.py --task A90 --episode-index 120
```
