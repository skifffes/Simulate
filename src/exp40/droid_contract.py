"""π₀.₅ DROID checkpoint 的真实观察与动作约定。

本文件里的每个常量都必须能追到官方源码或 checkpoint 自身的 norm_stats，不允许凭印象填写。
引用格式：`<相对 openpi 仓库的路径>:<行号>`，对应 openpi commit 215abfb217dbac7d5f1273282331b9b1866c0479。

一、模型配置
    `src/openpi/training/config.py:630`
        TrainConfig(name="pi05_droid", model=Pi0Config(action_horizon=15, pi05=True), ...)
    `src/openpi/models/pi0_config.py:25-26`
        action_dim = 32（模型内部宽度），action_horizon = 50（默认值，被 config 覆盖为 15）
    => 模型输出 (15, 32)，DroidOutputs 只保留前 8 维：`src/openpi/policies/droid_policy.py:92`

二、输入
    `src/openpi/policies/droid_policy.py:31-61`（DroidInputs，model_type=PI05 分支）
        image  = {"base_0_rgb": 外部相机, "left_wrist_0_rgb": 腕部相机, "right_wrist_0_rgb": 全零+maks=False}
        state  = concat([observation/joint_position (7), observation/gripper_position (1)]) -> 8 维
        prompt = 语言指令
    输入键名（客户端侧，见 `examples/droid/main.py:119-127`）：
        "observation/exterior_image_1_left", "observation/wrist_image_left",
        "observation/joint_position", "observation/gripper_position", "prompt"

三、预处理链（`src/openpi/policies/policy_config.py:73-84` 组装顺序）
    DroidInputs -> Normalize -> ResizeImages(224, 224) -> TokenizePrompt -> PadStatesAndActions(32)
    归一化方式：量化归一化（quantile）
        `src/openpi/training/config.py:187`  use_quantile_norm = model_type != PI0  => PI05 为 True
        `src/openpi/transforms.py:141-145`   x_norm = (x - q01) / (q99 - q01) * 2 - 1
    输出反归一化：`src/openpi/transforms.py:175-181`，超出 8 维的部分原样透传（所以 8 维之后是占位）。

四、动作语义（由 checkpoint 自带的 `assets/droid/norm_stats.json` 判定）
    actions 8 维，量化统计（q01 / q99）：
        dim 0-6：[-0.46, 0.45] / [-0.81, 0.77] / [-0.45, 0.45] / [-0.93, 0.79] /
                 [-0.65, 0.65] / [-0.65, 0.66] / [-0.76, 0.73]，均值 ≈ 0
                 => 关节增量（rad / 控制步），不是绝对关节角
        dim 7  ：[0.0, 0.9998]，mean 0.4514
                 => 归一化夹爪开合量（0=闭合，1=全开），绝对值，不随 state 变化
    官方 DROID 客户端的使用方式印证了这一点（`examples/droid/main.py`）：
        :80   RobotEnv(action_space="joint_velocity", gripper_action_space="position")
        :132  注释 “action chunk of 10 joint velocity actions (7) + gripper position (1)”
        :141-146 夹爪以 0.5 为阈值二值化
        :149  动作在送入控制器前 clip 到 [-1, 1]
        :23   控制频率 DROID_CONTROL_FREQUENCY = 15 Hz
    因此本实验的映射是：
        q_target = q_current + clip(a[0:7], -1, 1)      # 关节增量（rad）
        gripper  = a[7] ∈ [0, 1]                        # 0=闭合，1=全开
    这是“策略动作 → 适配目标”的一步，不等于“控制器指令”，更不等于“实际运动”。

五、state 语义
    dim 0-6：7 个关节角（rad），Franka Panda 关节顺序
    dim 7  ：归一化夹爪位置 [0, 1]
    PyBullet 端的映射：gripper 标量 = 单指位移 / 单指最大位移（0.04 m），见 observation_adapter。
"""
from __future__ import annotations

# ---- checkpoint 身份 ---------------------------------------------------------
CHECKPOINT_NAME = 'pi05_droid'
CHECKPOINT_GCS = 'gs://openpi-assets/checkpoints/pi05_droid'
CHECKPOINT_ASSET_ID = 'droid'
OPENPI_COMMIT = '215abfb217dbac7d5f1273282331b9b1866c0479'

# ---- 客户端输入键名 ---------------------------------------------------------
KEY_IMAGE_BASE = 'observation/exterior_image_1_left'
KEY_IMAGE_WRIST = 'observation/wrist_image_left'
KEY_JOINT_POSITION = 'observation/joint_position'
KEY_GRIPPER_POSITION = 'observation/gripper_position'
KEY_PROMPT = 'prompt'

# ---- 维度 -------------------------------------------------------------------
STATE_DIM = 8          # 7 关节 + 1 夹爪
ACTION_DIM = 8         # 7 关节增量 + 1 夹爪
MODEL_ACTION_DIM = 32  # 模型内部宽度（PadStatesAndActions），只有前 8 维有物理含义
ACTION_HORIZON = 15    # pi05_droid 的 action_horizon

# ---- 客户端图像规格 ---------------------------------------------------------
# 官方客户端在机器人侧先 resize_with_pad 到 224x224 再发送（examples/droid/main.py:120-123），
# 服务端的 ResizeImages(224, 224) 因此是 no-op。
CLIENT_IMAGE_HW = (224, 224)

# ---- 归一化 ----------------------------------------------------------------
NORMALIZATION = 'quantile'   # pi05 => use_quantile_norm=True
# checkpoint 自带 assets/droid/norm_stats.json 的关键区间（实验 4.0 记录，供离线核对）
ACTION_Q01 = (-0.4580, -0.8076, -0.4472, -0.9268, -0.6456, -0.6460, -0.7616, 0.0)
ACTION_Q99 = (0.4476, 0.7652, 0.4480, 0.7944, 0.6484, 0.6628, 0.7344, 0.9998)
GRIPPER_RANGE = (0.0, 1.0)   # 0=闭合，1=全开

# ---- 控制频率 ---------------------------------------------------------------
DROID_CONTROL_HZ = 15        # 官方 DROID 采集/执行频率
SIM_POLICY_HZ = 20           # 本仓库 SmolVLA 阶段的策略频率（vla/config.py:9）
SIM_HZ = 240

# ---- 语言指令（保持实验 1–3 原文，不在 4.0 中改写） -------------------------
PROMPT_B90 = 'push the orange block along the positive X direction to the wall'
PROMPT_A90 = None  # 见 docs/experiment_4_0_repo_audit.md；4.0 只记录，不改写


def action_dim_slice(n: int = ACTION_DIM) -> slice:
    """模型返回 (horizon, 32) 时，物理有效维度的切片。"""
    return slice(0, n)
