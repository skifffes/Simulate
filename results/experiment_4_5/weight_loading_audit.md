# 实验 4.5 Gate 1：模型结构与权重加载审计

目标：确认 Action Expert-only 微调建立在**原始 `pi05_droid` 权重**之上，且动作预测路径的
预训练权重被完整加载（不是随机初始化）。

## 1. 模型结构（`params_tree.json`，51 个张量）

| 模块 | 张量数 | 参数量 |
| --- | ---: | ---: |
| `PaliGemma/img`（SigLIP 视觉编码器） | 22 | 414.8 M |
| `PaliGemma/llm`（Gemma 主干 + Action Expert） | 25 | 2936.5 M |
| `action_in_proj` / `action_out_proj` | 4 | 0.06 M |
| `time_mlp_in` / `time_mlp_out` | 4 | 2.10 M |
| **合计** | **51** | **3353.4 M** |

Action Expert 在参数树里是 `PaliGemma/llm/**/*_1` 系列张量（`layers/attn/q_einsum_1`、
`mlp_1`、`pre_attention_norm_1`、`pre_ffw_norm_1`、`final_norm_1`），
与 PaLiGemma 主干（`.../layers/attn/q_einsum`、`mlp`、`final_norm`）**在张量层面是分开的**，
因此可以做到"冻结 VLM、只训练专家"，不需要元素级 mask。

## 2. 权重加载验证（`gate1_weight_loading.json`）

方法：固定 RNG 创建**未加载**模型 → 记录张量统计 → 加载 `pi05_droid` 权重 → 再记录。

| 张量 | 随机初始化 std | 加载后 std | 说明 |
| --- | ---: | ---: | --- |
| `.../mlp_1/gating_einsum`（专家） | 0.03125 | **0.03411** | 来自 checkpoint |
| `.../attn/q_einsum_1/w`（专家） | 0.03125 | **0.03393** | 来自 checkpoint |
| `.../final_norm_1/Dense_0/kernel`（adaRMS） | **0.00000** | **0.00968** | 随机初始化是 0，加载后非零 → **确实加载了预训练 adaRMS 权重** |
| `action_out_proj/kernel` | 0.03112 | **0.02036** | 来自 checkpoint |
| `.../mlp/gating_einsum`（VLM） | 0.02210 | **0.03841** | 来自 checkpoint |
| `.../embedder/input_embedding` | 0.01000 | **0.19284** | 来自 checkpoint |

`final_norm_1/Dense_0/kernel` 是决定性证据：该层随机初始化恒为 0，加载后 0.00968，
说明专家 adaRMSNorm 权重不是随机初始化。

## 3. 契约一致性

- action_dim=32、action_horizon=16、prompt 长度 200、两路 v3 相机（256²→224²）、
  `anchor_reference=fixed`、DROID normalization：与 4.4 完全一致（见 `training_config.json`）。
- 参数树 51 个张量、无 lora 张量（`gemma_2b` / `gemma_300m` 变体），
  因此 4.4 的 LoRA checkpoint **不能**直接用于本配置，反之亦然（服务端配置分开注册：
  `exp42_pi05_droid_lora` vs `exp45_pi05_ae`）。

## 4. 结论

Gate 1 **PASS**：专家结构与预训练权重兼容、adaRMS/投影权重均来自 checkpoint、
动作维度与 horizon 未改变。
