"""Hotfix A 冻结版相机配置（v2）。

设计依据（见 `results/experiment_4_0/hotfix/hotfix_report.md` 与 `camera_visibility_metrics.csv`）：

- 旧的 `top`（= External/Base）相机：`target=(0.55,0.21,0.02)`、`distance=1.05`、`yaw=50`、`pitch=-52`、`fov=50`。
  实测（分割 mask，256×256）：方块只占 0.71% 像素，且机械臂在多数阶段完全不在画面内。
  它是实验 1–3 为“看到整块桌面（含托盘和墙）”设计的远景，用于 π₀.₅ 的过肩类视角过远。
- v2 的目标是**让 A90/B90 的操作走廊（物体→墙 / 物体→放置区）和机械臂本体同时进入画面**，
  而不是只放大物体：`distance 0.65→0.70`、`pitch -52→-35`、`yaw 50`、`fov 50→60`、
  `target` 由桌面中心移到操作走廊中心 `(0.56, 0.13, 0.06)`。
  实测方块像素占比 ~1.2%（约 1.7×），机械臂同时可见（robot_frac 0.061）。
- 相机仍是**固定第三人称**：位置不随机器人或物体变化，不使用任何物体真值瞄准。
- Wrist 相机保持实验 2/3 的安装（link 8，offset (0.09,0,-0.01)，aim=grasp，fov 75），
  因为它是随末端运动的相对安装，且抓取阶段目标可见（见 hotfix 报告 §A3）。

本文件是**唯一**的相机版本源：`CAMERA_CONFIG_SHA256` 由下面的规范化 JSON 计算，
写入 `results/experiment_4_0/hotfix/camera_config_final.json`，后续实验必须引用同一哈希。
"""
from __future__ import annotations

import hashlib
import json

VERSION = 'exp4.0-camera-v2'

# 与 vla.cameras.CameraRig 完全兼容的相机字典（键名 top/wrist 保持不变：
# top 是历史命名，实际角色是 External/Base；wrist 是腕部相机）。
CAMERAS_V2 = {
    'top': {
        'type': 'fixed',
        'target': (0.56, 0.13, 0.06),
        'distance': 0.70,
        'yaw': 50.0,
        'pitch': -35.0,
        'fov': 60.0,
        'near': 0.02,
        'far': 5.0,
    },
    'wrist': {
        'type': 'wrist',
        'link': 8,
        'offset': (0.09, 0.0, -0.01),
        'aim': 'grasp',
        'look_ahead': 0.02,
        'target_down': 0.02,
        'fov': 75.0,
        'near': 0.005,
        'far': 3.0,
    },
}

# 修复前基线（= src/vla/atomic_tasks/config2.py 的 CAMERAS，逐字复制，便于 before/after）
CAMERAS_V1_BASELINE = {
    'top': {
        'type': 'fixed',
        'target': (0.55, 0.21, 0.02),
        'distance': 1.05,
        'yaw': 50.0,
        'pitch': -52.0,
        'fov': 50.0,
        'near': 0.02,
        'far': 5.0,
    },
    'wrist': {
        'type': 'wrist',
        'link': 8,
        'offset': (0.09, 0.0, -0.01),
        'aim': 'grasp',
        'look_ahead': 0.02,
        'target_down': 0.02,
        'fov': 75.0,
        'near': 0.005,
        'far': 3.0,
    },
}


def _canonical(cameras: dict) -> str:
    return json.dumps(cameras, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def config_sha256(cameras: dict | None = None) -> str:
    return hashlib.sha256(_canonical(cameras or CAMERAS_V2).encode('utf-8')).hexdigest()


CAMERA_CONFIG_SHA256 = config_sha256()

__all__ = ['VERSION', 'CAMERAS_V2', 'CAMERAS_V1_BASELINE', 'CAMERA_CONFIG_SHA256', 'config_sha256']
