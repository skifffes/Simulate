# 实验 4.1 Data Preflight（验收后修复）

对验收意见的逐条处理。**没有重新采集数据**：90 条演示原样保留，修复集中在训练导出、评价器、
验证器与分组定义。

- 评价器版本：`exp4.1-strict-v1` → **`exp4.1-strict-v1.1`**
- 训练导出契约：**`exp4.1-training-export-v1`**（新文件 `src/exp41/training_export.py`）
- 工件：`gripper_label_audit.json`、`time_contract.json`、`reeval_v11.json`、`reeval_sample.json`、
  `video_and_transforms.json`、`openpi_transform_check.json`

## 1. 夹爪训练标签（review #1，最高优先级）——**已修复，且证明原实现确实错**

原 `audit_dataset.py` 用 `state[:, 7:9]` 计算"训练动作夹爪"。现在训练导出改为：

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `state8[7]` | `mean(state[7:9]) / 0.04` | 观测夹爪（0=闭合，1=全开） |
| `action8[7]` | **`action[7] / 0.04`** | **专家下发的单指宽度目标** |
| `action8[7]_bin` | `action8[7] > 0.5` | 与官方 client 下发方式的对照版本 |

实测（45 条 A90）：

| 指标 | 数值 |
| --- | --- |
| 出现"状态仍张开、动作已命令闭合"帧的 episode | **45 / 45** |
| `state8[7]` 均值 / 张开比例 | 0.795 / **1.000** |
| `action8[7]` 均值 / 张开比例 | 0.755 / **0.574** |
| 动作夹爪取值 | 0.425（闭合）/ 1.0（张开） |
| 单条 episode 的最大 `action−state` 差 | 低至 **−0.575**（抓取瞬间） |

也就是说：按旧写法，抓取阶段模型会收到"保持张开"的监督；现在闭合事件被正确标注为 0.425。

## 2. 20 Hz → 15 Hz 比例（review #2）——**已纠正**

| 量 | 正确值 |
| --- | --- |
| 保持关节速度不变 | **Δq₁₅ = Δq₂₀ × (20/15) = 1.3333 ×** |
| 旧文档错误写法 | ×15/20 = 0.75 ×（比例写反） |

`time_contract.json` 已冻结正确公式，并给出建议：**4.2 只选一个频率**（沿用 20 Hz，或按 15 Hz
重新采集/按时间戳重采样并用严格评价器复验），不要对已采集的闭环轨迹做简单线性缩放——
它不保留接触动力学。

## 3. 评价器补强（review #4）——**已加强；旧数据不静默改标签**

`exp4.1-strict-v1.1` 新增：

| 新增判据 | 实现 |
| --- | --- |
| 双指抓取 | `finger_contact_stats()` 分别统计 **link 9 与 link 10** 的接触步数与法向力，要求**两指都** ≥5 步、≥0.05 N |
| 释放后滑移 | 稳定窗口**最后 0.2 s** 内物体位移 ≤10 mm |
| 尾段角速度 | 稳定窗口尾段角速度 ≤0.5 rad/s（整段角速度包含 90° 旋转，不能用于稳定性） |

**旧记录无法重算这两项**：当时 `contact_summary.robot_block.links` 汇总的是 `link_a`（方块侧，恒 −1），
且没有保存稳定窗口尾段轨迹。`reeval_v11.json` 因此把 45 条标为 **`not_decidable_from_record`**，
`n_downgraded = 0`，旧标签原样保留。

**外部证据（重放 6 条，`reeval_sample.json`）**：

| 指标 | 结果 |
| --- | --- |
| 双指接触 | **6 / 6**（finger9 933–950 步、finger10 935–948 步，两指都在） |
| 释放后滑移 | **6 / 6** ≤0.1 mm |
| 尾段角速度 | **6 / 6** ≤0.018 rad/s |
| 严格成功 | **5 / 6** |

第 6 条（`A_x+3_y-3`，新目标点抖动）在 v1.1 下因 `tilt_ok` 失败（放置后倾斜超 12°），
这是**新发现**：A90 专家并非 100% 可靠，45/45 只代表已归档批次；独立重放抽样约 5/6。
已在主报告的 RQ1 与 Blockers 中更新。

## 4. 回环与 OpenPI transforms（review #3）——**已补**

| 检查 | 结果 |
| --- | --- |
| 视频解码 | `imageio` 解码 **90/90 个视频文件**（每技能 2 相机 × 45）；抽样帧数 189–198，与 parquet 行数一致 |
| OpenPI transforms | 在 `openpi/.venv` 里 `import openpi.transforms` **成功** |
| 8 维 state / 8 维 action | `(198, 8)` / `(198, 8)`，float64，`action8[7] ∈ [0.425, 1.0]`，`|Δq|max = 0.049` |
| Normalize → Unnormalize 回环 | 误差 **5.6e-17 / 6.9e-18**，`pass = true` |
| 归一化后范围 | state ∈ [−2.73, 2.09]，action ∈ [−2.84, 5.08] |

`normalized_action_range` 上限 5.08 说明**自算 q01/q99 与 DROID stats 不同**，
4.2 必须显式决定用哪套统计（这是 B5 的实证）。

## 5. 汇总布尔错误（n_success=107）——**已修复**

`collection_summary.json` 的 `n_success` 现在按 `"true"/"false"` 解析：

```
n_attempts = 107, n_success = 90
by_skill = {A90: {attempts 45, success 45}, B90: {attempts 62, success 45}}
```

旧值 107 的原因：历史 CSV 读回后 `strict_success` 是字符串，非空字符串恒为真。
修复同时写入 `n_success_fixed_by_preflight = true` 与 `bug_note`。

## 6. 分组冻结（review：split 由结果生成）——**已改为结果无关的预注册文件**

`split_manifest.json`（`exp4.1-split-v1`）现在只从 `scene41.py` 的目标/条件定义生成，
与采集结果无关，并显式区分两档测试：

| 档位 | 内容 | 用途 |
| --- | --- | --- |
| `test_atom_extrapolation` | 目标/条件本身在训练中未出现 | 更难的外推，**不能**用来归因组合失败 |
| `composition_tier1`（4.3） | 两侧原子条件都见过，但 (A 终态, B 起点) **配对**未出现 | 隔离组合连接问题的主测试 |

并写入泄漏规则（近重复不得跨集合、禁止 A→B 轨迹、B 演示必须独立启动）。

## 7. 仍未解决（如实记录）

1. **A90 45/45 是已归档批次**；独立重放 6 条为 5/6，专家在 `A_x+3_y-3` 有倾斜失败 → 成功率不是 100%；
2. **B 目标条件维度**仍未实现（−Y 横移不可达），主数据仍是预配对；
3. **归一化统计**未定（OpenPI 检查显示自算 q99 与 DROID 差异显著）；
4. 旧 45 条 A90 无法在 v1.1 下**重新判定**双指/滑移，只能靠重放抽样给外部证据；
5. 未做 15 Hz 数据的实际采集或重采样验证。
