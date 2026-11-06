# Gate 1：训练输入 vs 推理输入（`observation_contract.csv`）

## 方法

对 4 条固定 episode × 9 个阶段帧（共 36 帧），从**同一原始观测**分别走两条路径：

- **训练路径**：`train_lora.make_item()` → `train_lora.make_transforms()`（含 DROID Normalize）→ 模型输入张量；
- **推理路径**：与 `serve_policy` 客户端完全相同的 dict（`observation/exterior_image_1_left`、
  `observation/wrist_image_left`、`observation/joint_position`、`observation/gripper_position`、`prompt`）
  → `Policy._input_transform` → 模型输入张量。

逐字段比较 shape / dtype / max abs / mean abs / SHA256，token 与 mask 要求逐位相等。
见 `scripts/exp47/gate12_contract.py`。

## 结果：**324 / 324 字段 PASS**

| 字段 | 样本 | max abs | 说明 |
| --- | ---: | ---: | --- |
| `image.base_0_rgb` | 36 | **0.0** | top 相机（224×224，float32 归一化后逐位相同） |
| `image.left_wrist_0_rgb` | 36 | **0.0** | wrist 相机 |
| `image.right_wrist_0_rgb` | 36 | **0.0** | 占位相机（全零 + mask False） |
| `image_mask.*` | 36 | **0.0** | True/True/False，逐位相同 |
| `state` | 36 | **0.0** | 9D→32D padding 后逐位相同 |
| `tokenized_prompt` | 36 | **0.0** | 200 token，逐位相同 |
| `tokenized_prompt_mask` | 36 | **0.0** | 逐位相同 |

SHA256 也全部一致（`observation_contract.csv` 的 `sha_a`/`sha_b` 列）。

## 结论

- 同一离线观测经两套预处理后**完全相同**（max abs 严格为 0，非"在容差内"）；
- 颜色顺序、resize/interp、top/wrist 映射、第三路占位 mask、prompt 文本与 token、DROID stats 版本均一致；
- 这一结论只覆盖"**同一离线观测**经过不同预处理"，**不涉及**"重新渲染的仿真图像是否与历史记录一致"
  （后者受渲染随机性影响，本轮未测试，也不应混为一谈）。

## 具体源码位置

- 训练路径：`scripts/exp42/train_lora.py::make_item`（L131）、`make_transforms`（L85）；
- 推理路径：`openpi/policies/policy.py::Policy.infer`（L68-105，`_input_transform`/`_output_transform`）；
- 归一化资产：`openpi-assets/checkpoints/pi05_droid/assets/droid`（两路径同一 `assets_dir`）。
