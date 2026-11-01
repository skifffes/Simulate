"""**桩模块（stub）**：只为让 `openpi.training.data_loader` 能 import，不提供任何数据加载功能。

为什么需要它：
    openpi 的 `src/openpi/training/checkpoints.py:16` 在模块顶层 import 了
    `openpi.training.data_loader`，而 data_loader 在模块顶层
    `import lerobot.common.datasets.lerobot_dataset`。也就是说，即使只做推理
    （policy_config → checkpoints.load_norm_stats），也必须能 import lerobot。
    在本机（CentOS 7 / glibc 2.17）安装 lerobot 会连带拉入 torchcodec 0.4.0 与
    rerun-sdk 0.23.1，两者都只有 manylinux_2_31/2_28 的 wheel，无法安装。

诚实性说明：
    * 本桩只实现到“可 import”为止，**任何真正被调用的成员都会立刻抛 RuntimeError**；
    * 实验 4.0 不训练、不加载 LeRobot 数据集，因此推理路径不会触碰这些符号；
    * 桩的安装方式写进 environment_report.md 与复现脚本，不隐藏在环境里。
"""
from __future__ import annotations


class _Unavailable:
    """被真正调用时立刻失败，避免静默产生错误的训练/数据行为。"""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f'{self._name} 是实验 4.0 的 lerobot 桩，不能真正使用：'
            '本次只做推理，训练/数据加载路径不受支持（见 environment_report.md）')

    def __getattr__(self, item):
        raise RuntimeError(f'{self._name}.{item} 是实验 4.0 的 lerobot 桩，不可用')


LeRobotDataset = _Unavailable('LeRobotDataset')
LeRobotDatasetMetadata = _Unavailable('LeRobotDatasetMetadata')
