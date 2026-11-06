# 实验 4.5 Gate 2 / Gate 3 / Gate 4：冻结、梯度、资源与续训审计

所有结论来自真实运行：`scripts/exp45/gate2_audit.py`（6 步真实更新）+ `scripts/exp45/train_ae.py`
（续训测试）。

## Gate 2：冻结与梯度审计（`gate2_audit.json` / `parameter_partition.json`）

| 项 | 值 |
| --- | --- |
| 总参数 | **3,353,433,872（3353.4 M）** |
| 可训练参数 | **430,098,464（430.1 M，12.83%）** |
| 可训练张量 | **19**（Action Expert 15 + `action_in/out_proj` 4 个 + `time_mlp_*` 4 个） |
| 冻结张量 | **32**（`PaliGemma/img` 22 + `PaliGemma/llm` 非 `_1` 10） |
| optimizer state | 40 个叶，**860,196,930 元素 = 2 × 可训练参数**（Adam μ、ν） |

6 步真实更新后：

| 检查 | 结果 |
| --- | --- |
| 冻结张量数值改变的个数 | **0 / 32** |
| 可训练张量数值改变的个数 | **19 / 19** |
| 未改变的专家张量 | **0** |
| `gate2_pass` | **true** |

→ 视觉语言主干**没有任何 optimizer 更新**；Action Expert 与动作投影**全部参与更新**，
不存在"只更新投影层、专家内部没动"的情况。

可训练张量明细（`parameter_partition.json` 有完整 51 行）：

| 张量 | shape | 参数 |
| --- | --- | ---: |
| `.../layers/mlp_1/gating_einsum` | [18,2,1024,4096] | 150.99 M |
| `.../layers/mlp_1/linear` | [18,4096,1024] | 75.50 M |
| `.../layers/attn/q_einsum_1/w` | [18,8,1024,256] | 37.75 M |
| `.../layers/attn/attn_vec_einsum_1/w` | [18,8,256,1024] | 37.75 M |
| `.../pre_attention_norm_1/Dense_0/kernel` | [18,1024,3072] | 56.62 M |
| `.../pre_ffw_norm_1/Dense_0/kernel` | [18,1024,3072] | 56.62 M |
| `.../final_norm_1/Dense_0/kernel` | [1024,3072] | 3.15 M |
| `.../attn/kv_einsum_1/w` | [18,2,1,1024,256] | 9.44 M |
| `time_mlp_in` / `time_mlp_out` | 1024×1024 ×2 | 2.10 M |
| `action_in_proj` / `action_out_proj` | 32×1024 / 1024×32 | 0.06 M |

## Gate 3：A40 可行性（`gate3_benchmark.json` + 主训练日志）

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA A40 46,068 MiB（独占，训练用 GPU0） |
| batch size | **4**（未用梯度累积，effective batch = 4） |
| forward/backward/update | 全部成功 |
| **峰值显存** | **34,375 MiB（34.4 GB）** |
| JIT 编译（冷） | **247 s** |
| JIT 编译（缓存命中） | **16.6 s** |
| 吞吐 | **1.05–1.10 step/s**（12,000 步约 3.0–3.2 h） |
| loss / grad | 第 1 步 1.1620 / 9.24；第 1200 步 0.0028 / 0.168（无 NaN） |
| checkpoint 保存 | 1.59 GB（trainable-only）+ optimizer 3.44 GB，均可写 |

→ A40 上**可行**：batch 4、34.4 GB、>1 step/s，没有触发 OOM。
（注意：XLA 提示无法把内存压到 12.1 GiB 以下，属正常 rematerialization 警告。）

## Gate 4：真实 checkpoint resume（`resume_run1.csv` / `resume_run2.csv`）

第一次测试发现 **off-by-one 缺陷**：warm-up step 消耗了一次更新/一批数据，
而 opt-state 在 warm-up 之前恢复，导致续训后 loss 序列整体错位一格
（run2 step7 = run1 step8）。修复方式：在 warm-up step **之后**重新恢复参数与 optimizer 状态。

修复后（同 seed、12 步、第 6 步保存）：

| step | 连续训练 loss | 续训 loss | 差异 |
| ---: | ---: | ---: | --- |
| 7 | 0.9048845768 | 0.9048845768 | 0 |
| 8 | 1.8879981041 | 1.8879981041 | 0 |
| 9 | 2.0104656219 | 2.0104656219 | 0 |
| 10 | 1.7754880190 | 1.7754880190 | 0 |
| 11 | 1.8024613857 | 1.8024613857 | 0 |
| 12 | 1.7556833029 | 1.7556833029 | 0 |

→ 恢复了 **参数 + optimizer state + scheduler step + JAX/NumPy RNG + step 计数**，
续训与连续训练**逐位一致**。Gate 4 **PASS**。
