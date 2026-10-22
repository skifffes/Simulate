"""实验 2 的固定约定：控制频率、相机参数与四个原子技能的语言指令。"""
from __future__ import annotations

# 与实验 1 完全一致的控制时序：240 Hz 物理，20 Hz 策略。
SIM_HZ = 240
FPS = 20
STEPS_PER_ACTION = SIM_HZ // FPS  # 12

IMAGE_SIZE = (256, 256)
ROBOT_TYPE = 'panda'

# 第三人称相机需要同时看到：初始物体 (0.55, 0)、A 放置区 (0.55, 0.18)、
# 挡板 (x=0.78)、托盘 (0.55, 0.40)。视场比实验 1 更宽。
CAMERAS = {
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

TASKS = {
    'A0': 'pick up the orange block and place it in the target area, '
          'keeping its long side aligned with Y',
    'A90': 'pick up the orange block and place it in the target area, '
           'with its long side aligned with X',
    'B': 'push the orange block along the positive X direction to the wall',
    'C': 'pick up the orange block and place it into the tray, '
         'with its long side aligned with Y',
}

STATE_NAMES = [f'panda_joint{i + 1}.pos' for i in range(7)] + ['finger1.pos', 'finger2.pos']
ACTION_NAMES = [f'panda_joint{i + 1}.pos' for i in range(7)] + ['gripper.pos']


def feature_dict():
    from ..lerobot_writer import image_feature, vector_feature

    features = {
        'observation.state': vector_feature(STATE_NAMES),
        'action': vector_feature(ACTION_NAMES),
    }
    for name in CAMERAS:
        features[f'observation.images.{name}'] = image_feature(IMAGE_SIZE, FPS)
    return features
