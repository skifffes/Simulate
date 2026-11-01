# 实验 4.0 Hotfix 目录

基准提交 `557aaea`。本目录只放**本次 hotfix** 的工件，4.0 的原始记录在上一级目录，未覆盖。

| 文件 | 内容 |
| --- | --- |
| `hotfix_report.md` | 五个模块的结论（PASS/PARTIAL）、diff 摘要、相机改动理由、复现命令、已知缺陷 |
| `official_semantics_audit.md` | 夹爪方向、七关节尺度、0.2 系数、控制频率、视觉约定的官方源码证据链 |
| `camera_before_after/` | 修复前后 × A90/B90 × 多阶段 × 外部/腕部的 256 原始图与 **224×224 实际模型输入**，以及 `camera_before_after_metrics.csv`、`camera_visibility_metrics.csv`（含候选视角对比） |
| `camera_config_final.json` | 冻结的相机配置（v2）+ 修复前基线（v1）+ SHA256 |
| `hotfix_semantics_tests.json` | 29 条判据（A 相机 / B 夹爪 / C 关节尺度 / D 时序 / E RNG）逐条结果 |
| `manifest.json` | 本目录工件的路径/大小/sha256 清单 |

## 阶段状态的来源（避免误读）

- `home`：场景初始化后的真实状态；
- `trace_tN`：归档的 π₀.₅ **真实闭环轨迹**第 N 个控制步的 `q`（`../closed_loop_smoke/trace_*.csv`），
  仅用于相机诊断重放，**不是**编造的专家成功过程，也不代表任务被执行过。

## 结论速览

- Camera：**PARTIAL**（几何可见性达标并冻结；训练域分布未对齐）
- Gripper：**PASS**（`1=open`，观测与动作一致，保留原实现 + 补证据）
- Joint Action：**PASS**（无额外缩放，`q_target-q == action[:7]`）
- Timing：**PASS 定义 / PARTIAL 实时**（默认 15 Hz、每 tick 16 物理步、同步仿真闭环）
- RNG：**PASS 接线 / PARTIAL 模型级**（逐回合 seed 可用，未在 GPU 上重复两次 episode 验证）
- Closed Loop：**PARTIAL**（相机/时序改动后未重跑；4.0 基线记录在上一级目录）
