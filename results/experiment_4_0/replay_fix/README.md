# 实验 4.0 Replay Fix（B90 场景恢复 / 阶段证据 / 相机复核）

基准提交 `75bd153`。本目录修复上一轮 `final_validation/` 的两个缺陷：
**B90 重放用错物体场景** 和 **分割 mask 被写成了 RGB**。旧目录保留不覆盖。

| 文件 | 内容 |
| --- | --- |
| `replay_fix_report.md` | Q1–Q8 逐项结论 + Observed/Supported/Not Verified/Invalidated |
| `camera_decision.md` | 相机是否修改的决策与依据（结论：**不修改**） |
| `replay_provenance.json` | episode 来源、数据集 parquet、恢复误差、首接触帧、是否有墙接触、`PHYSICAL_REPLAY` 标记 |
| `episode_restoration_checks.json` | 恢复误差与预注册阈值 |
| `contact_phase_validation.csv` | 逐帧 `block_x/dx`、机器人接触数、墙接触数 |
| `camera_visibility_corrected.csv` | 正确场景下各阶段 External/Wrist 指标 |
| `camera_corrected_frames/` | RGB + `*_segvis.png`（着色 mask）+ `*_segids.npy`（原始 int32 body id） |
| `segmentation_validation.json` | mask 与 RGB 的哈希差异断言、唯一 id 列表 |
| `artifact_manifest.json` | 本目录工件清单与 sha256 |

## 一句话结论

恢复 episode 64 的真实场景后，腕部相机在**首次接触**与**推动**阶段能清楚看到方块 →
**相机不需要改**；但该 episode 在本次重放中**没有真正推动方块（Δx=0.9 mm）、也没有触墙**，
因此 `wall_contact` 标签改为 `wall_contact_not_verified`，且不得当作推墙示范。
