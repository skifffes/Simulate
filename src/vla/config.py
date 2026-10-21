"""实验 1 的固定约定：控制频率、观测定义、动作定义、相机参数与任务文本。

所有模块（采集、写入、回放、评测）都从这里取常量，保证专家数据与策略执行
使用完全相同的时序和动作语义。
"""

# 仿真固定 240 Hz；策略控制 20 Hz，即一个动作被电机保持 12 个仿真步。
SIM_HZ = 240
FPS = 20
STEPS_PER_ACTION = SIM_HZ // FPS  # 12

# 观测图像与数据集中的图像尺寸（H, W, 3），RGB。
IMAGE_SIZE = (256, 256)
# 评测时允许把同一动作连续执行 5~10 个控制周期再重新观察。
EVAL_EXEC_STEPS = 8
EVAL_EXEC_STEPS_RANGE = (5, 10)

TASK = 'pick up the orange block and place it on the target position'
ROBOT_TYPE = 'panda'

# 7 个手臂关节 + 两指实际位移。
STATE_NAMES = [f'panda_joint{i + 1}.pos' for i in range(7)] + ['finger1.pos', 'finger2.pos']
# 7 个手臂关节目标角 + 单指目标宽度（夹爪电机目标）。
ACTION_NAMES = [f'panda_joint{i + 1}.pos' for i in range(7)] + ['gripper.pos']

# 相机：top 为第三人称固定视角，wrist 固定在手掌坐标系上朝手指方向看。
CAMERAS = {
    'top': {
        'type': 'fixed',
        'target': (0.48, 0.08, 0.09),
        'distance': 0.75,
        'yaw': 45.0,
        'pitch': -30.0,
        'fov': 34.0,
        'near': 0.02,
        'far': 5.0,
    },
    'wrist': {
        'type': 'wrist',
        'link': 8,
        # 侧挂腕部相机：沿手掌侧向 9 cm、略微前移，看向两指中点下方 2 cm，
        # 使夹爪张合与工件同时完整落在画面内（详见 diagnostics/camera_probe）。
        'offset': (0.09, 0.0, -0.01),
        'aim': 'grasp',
        'look_ahead': 0.02,
        'target_down': 0.02,
        'fov': 75.0,
        'near': 0.005,
        'far': 3.0,
    },
}

# 数据集目录（相对仓库根目录）。
DATASET_DIRNAME = 'data/lerobot_panda_pick_place'


def feature_dict():
    """返回 LeRobot v2.1 的 features 定义（不含 DEFAULT_FEATURES）。"""
    from .lerobot_writer import image_feature, vector_feature

    features = {
        'observation.state': vector_feature(STATE_NAMES),
        'action': vector_feature(ACTION_NAMES),
    }
    for name in CAMERAS:
        features[f'observation.images.{name}'] = image_feature(IMAGE_SIZE, FPS)
    return features
