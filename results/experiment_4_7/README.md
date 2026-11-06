# 实验 4.7：π₀.₅ 训练—推理—物理执行端到端契约验收

本轮不训练、不扩数据、不改超参，只验证"训练 → 推理 → 执行"链路的数值一致性。
固定 4 条严格成功 episode（A90 ep0/ep5、B90 ep0/ep2）与冻结的 exp4.6 LoRA step3000 checkpoint。

## 结论

**契约通过（图像-状态时序一项 PARTIAL）**：
- 训练与在线推理的模型输入**逐位一致**（324/324 字段，max abs 0.0）；
- Anchor-16 标签经部署端反归一化 + Adapter 后**浮点精确**还原专家目标（1e-16），
  仅 0.35% 的行被安全裁剪（max 0.013 rad，已单独标注）；
- **专家逐周期回放 4/4 严格成功，Oracle chunk 4/4 成功**；
- 控制器跟踪 0.012 rad、夹爪跟踪 0.0065；FK/EE 坐标系一致（差 0.0 m）；
- 离线 vs Policy 推理差 1.3–3.7e-3 rad（bf16 精度级）。

闭环仍失败，但主导项是**模型预测误差 + 闭环漂移**，不是链路不一致（见 `end_to_end_error_budget.md`）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `contract_test_plan.md` | 容差与 Gate 定义（先冻结） |
| `observation_contract.csv` | 324 条字段级对比（shape/dtype/maxabs/sha） |
| `preprocess_comparison.md` | Gate 1 结论 |
| `action_roundtrip.csv` / `action_roundtrip.md` | Anchor-16 全行重建（2,828 行） |
| `adapter_clipping_audit.json` | k 分区裁剪/接受率统计 |
| `expert_execution_trace.csv` | 四层执行 trace（2,828 tick） |
| `oracle_chunk_results.csv/json` | A/B1/B2/unsafe 严格成功率与指标 |
| `control_tracking_audit.md` | Gate 3 判读 |
| `fk_frame_audit.md` / `fk_consistency.csv` | Gate 4 坐标系审计（20 构型） |
| `first_deviation_v2.csv` | 用运行时 EE 重算的首次偏离（tick0=0.0 m） |
| `policy_inference_parity.csv` | 离线 vs Policy 输出对比 |
| `end_to_end_error_budget.md` | 五类误差分开报告 |
| `ACCEPTANCE.md` | 逐层 PASS/FAIL/PARTIAL 表 + 六问 |
| `final_report.md` | 最终报告 |

## 复现

```bash
# Gate 1/2（需要合并后的 step3000 checkpoint）
CUDA_VISIBLE_DEVICES=2 EXP42_DATA=/scratch/lijh/exp46/export_overfit \
  python scripts/exp47/gate12_contract.py

# Gate 3（专家与 Oracle 回放）
python scripts/exp47/gate3_replay.py

# Gate 4（坐标系 + 首次偏离）
python scripts/exp47/gate4_fk_audit.py
python scripts/exp47/first_deviation_v2.py

# Gate 5（离线 vs Policy）
CUDA_VISIBLE_DEVICES=2 python scripts/exp47/gate5_parity.py
```

服务器工件：`/scratch/lijh/exp47/merged/step3000`（12.6 GB 合并 checkpoint）；
本轮所有输出在 `results/experiment_4_7/`，无新增训练 checkpoint。
