"""实验 4.1 场景与目标注册表：多 A 放置目标、多 B 启动条件与多个 B 墙面目标。

设计原则（对应 prompt §5/§6）：
  * **权威 body ID** 一律来自 `build_scene()` 的返回值，绝不按碰撞盒尺寸猜测；
  * 目标、启动条件、机器人启动构型分别有独立 ID，不允许用一个 episode 编号混指；
  * 相机使用实验 4.0 冻结的 `exp4.0-camera-v3`（外部 v2 视角 + 腕部固定外参），
    版本与 SHA256 随每条记录一起保存；
  * 初始状态检查（不贴墙、不穿模、稳定）在专家执行**之前**完成。

坐标约定沿用 `scene2.py`：+X 为推墙方向，+Y 为 A 的搬运方向，+Z 向上。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict, field

import numpy as np

from exp40 import camera_v2
from manipulation import create_block, enable_contact_solver
from simulation import Simulation
from vla.atomic_tasks.scene2 import (
    BLOCK_FRICTION, BLOCK_HALF, BLOCK_MASS, PLACE, START, WALL_HALF, WALL_INNER_X,
    START as B90_START, make_tray, make_wall,
)

TABLE_TOP_Z = 0.0          # plane.urdf 顶面
BLOCK_HALF_Z = BLOCK_HALF[2]


# --------------------------------------------------------------------- 目标
@dataclass(frozen=True)
class AGoal:
    """A90 的放置目标：物体中心 (x, y)、目标长轴朝向（A90 要求沿 X → target_axis='x'）。"""

    goal_id: str
    x: float
    y: float
    target_axis: str = 'x'      # A90 要求长轴沿 X
    region_id: str = 'A_place'

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BGoal:
    """B90 的墙面目标区：物体推到墙内侧，且中心 y 落在 [y_lo, y_hi]。"""

    goal_id: str
    y_lo: float
    y_hi: float
    wall_inner_x: float = WALL_INNER_X

    @property
    def y_center(self) -> float:
        return 0.5 * (self.y_lo + self.y_hi)

    def as_dict(self) -> dict:
        d = asdict(self)
        d['y_center'] = self.y_center
        return d


# A 的 3×3 放置网格：以旧 PLACE=(0.55,0.18) 为中心，±3 cm 网格（先按几何粗筛，再物理验证）
A_GOALS = [
    AGoal('A_x-3_y-3', 0.52, 0.15), AGoal('A_x00_y-3', 0.55, 0.15), AGoal('A_x+3_y-3', 0.58, 0.15),
    AGoal('A_x-3_y00', 0.52, 0.18), AGoal('A_x00_y00', 0.55, 0.18), AGoal('A_x+3_y00', 0.58, 0.18),
    AGoal('A_x-3_y+3', 0.52, 0.21), AGoal('A_x00_y+3', 0.55, 0.21), AGoal('A_x+3_y+3', 0.58, 0.21),
]

# B 的三个墙面目标区（沿 Y 分布的三个区域，每个宽 6 cm）
B_GOALS = [
    BGoal('B_wall_y_low', 0.12, 0.18),
    BGoal('B_wall_y_mid', 0.18, 0.24),
    BGoal('B_wall_y_high', 0.24, 0.30),
]

# B 的物体启动场景（物体中心 y、初始 x，以及 yaw）
B_OBJECT_STARTS = [
    {'start_id': 'B_obj_y015', 'x': 0.548, 'y': 0.155, 'yaw': math.pi / 2},
    {'start_id': 'B_obj_y021', 'x': 0.553, 'y': 0.210, 'yaw': math.pi / 2},
    {'start_id': 'B_obj_y027', 'x': 0.546, 'y': 0.270, 'yaw': math.pi / 2},
]

# B 的机器人启动构型（两类：HOME 与关节空间偏离 HOME 的合法构型）
B_ROBOT_STARTS = [
    {
        'robot_start_id': 'B_robot_home',
        'q': [0.0, -0.5, 0.0, -2.1, 0.0, 1.65, 0.78],
        'gripper': 0.035,
    },
    {
        'robot_start_id': 'B_robot_offset',
        'q': [0.35, -0.65, -0.10, -1.95, 0.15, 1.75, 0.95],
        'gripper': 0.035,
    },
]

# 预先配对的 B 条件：(物体起点, 目标区)。物体起点落在目标区的可达走廊内，
# 因此不需要横向修正即可完成 +X 推墙。跨区配对（需要横向修正）单独作为
# "目标条件可区分性"可行性测试，不进入主数据（见 results/experiment_4_1/reports）。
B_ALIGNED_PAIRS = [
    ('B_obj_y015', 'B_wall_y_low'),
    ('B_obj_y021', 'B_wall_y_mid'),
    ('B_obj_y027', 'B_wall_y_high'),
]


A_GOAL_JITTER_M = 0.008     # 每条 episode 在目标点附近的确定性抖动（±8 mm，仍在 2 cm 容差内）


def a_goal_point(goal: 'AGoal', seed: int) -> tuple[float, float]:
    """把"目标区域"落成一个具体的放置点：目标中心 + 预注册的确定性抖动。

    这样 A 的终态多样性来自**不同的目标点**，而不是同一点的重复采样；
    指令文本会写出实际目标点，评价器也按该点算误差（容差仍是 2 cm）。
    """
    rng = np.random.default_rng(int(seed) * 7919 + 17)
    return (float(goal.x) + float(rng.uniform(-A_GOAL_JITTER_M, A_GOAL_JITTER_M)),
            float(goal.y) + float(rng.uniform(-A_GOAL_JITTER_M, A_GOAL_JITTER_M)))


def registry() -> dict:
    return {
        'camera_sha256': camera_v2.CAMERA_CONFIG_SHA256,
        'a_goals': [g.as_dict() for g in A_GOALS],
        'b_goals': [g.as_dict() for g in B_GOALS],
        'b_object_starts': B_OBJECT_STARTS,
        'b_robot_starts': B_ROBOT_STARTS,
        'block_half_extents': list(BLOCK_HALF),
        'wall_inner_x': WALL_INNER_X,
        'table_top_z': TABLE_TOP_Z,
        'registry_sha256': hashlib.sha256(json.dumps({
            'a': [g.as_dict() for g in A_GOALS], 'b': [g.as_dict() for g in B_GOALS],
            'b_obj': B_OBJECT_STARTS, 'b_robot': B_ROBOT_STARTS,
            'cam': camera_v2.CAMERA_CONFIG_SHA256,
        }, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }


# --------------------------------------------------------------------- 场景
class Scene41:
    """一个显式构造的场景，暴露全部权威 body ID 与初始状态。"""

    def __init__(self, kind: str, *, obj_pos, obj_yaw: float, robot_q=None, gripper=0.035,
                 camera_version: str = 'v3'):
        import pybullet as p

        if kind not in ('A90', 'B90'):
            raise ValueError(kind)
        self.kind = kind
        self.sim = Simulation(gui=False, reference_cube=False)
        enable_contact_solver(self.sim)
        self.block = create_block(self.sim, BLOCK_HALF, BLOCK_MASS, tuple(obj_pos),
                                  orientation=p.getQuaternionFromEuler([0.0, 0.0, float(obj_yaw)]),
                                  lateral_friction=BLOCK_FRICTION)
        self.wall = make_wall(self.sim.client)
        self.trays = make_tray(self.sim.client)
        # 推墙允许指/掌接触；A90 允许两指接触（抓取）。这里统一放开机器人-物体接触，
        # 由评价器按任务语义判断，而不是让 collisions() 直接判失败。
        self.sim.allowed_contacts = {(self.block, link) for link in (8, 9, 10)}
        self.plane = self.sim.plane
        self.robot = self.sim.robot
        if robot_q is not None:
            for j, v in enumerate(robot_q):
                p.resetJointState(self.robot, j, float(v), physicsClientId=self.sim.client)
            for j in (9, 10):
                p.resetJointState(self.robot, j, float(gripper), physicsClientId=self.sim.client)
            self.sim.command(np.asarray(robot_q, dtype=float))
        for j in (9, 10):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, float(gripper), force=30,
                                    physicsClientId=self.sim.client)
        for _ in range(120):
            self.sim.step()
        self.cameras = {'v1': camera_v2.CAMERAS_V1_BASELINE, 'v2': camera_v2.CAMERAS_V2,
                        'v3': camera_v2.CAMERAS_V3}[camera_version]
        self.camera_version = camera_version
        self.camera_sha256 = camera_v2.config_sha256(self.cameras)

    # ---- 状态查询 ----
    def object_state(self) -> dict:
        import pybullet as p

        pos, quat = p.getBasePositionAndOrientation(self.block, physicsClientId=self.sim.client)
        lin, ang = p.getBaseVelocity(self.block, physicsClientId=self.sim.client)
        euler = p.getEulerFromQuaternion(quat)
        return {'pos': np.asarray(pos, float), 'quat': np.asarray(quat, float),
                'lin_vel': np.asarray(lin, float), 'ang_vel': np.asarray(ang, float),
                'yaw': float(np.arctan2(np.sin(euler[2]), np.cos(euler[2]))),
                'roll_deg': float(math.degrees(np.arctan2(np.sin(euler[0]), np.cos(euler[0])))),
                'pitch_deg': float(math.degrees(np.arctan2(np.sin(euler[1]), np.cos(euler[1]))))}

    def robot_state(self) -> dict:
        import pybullet as p

        q = np.asarray(self.sim.q(), float)
        dq = np.asarray([p.getJointState(self.robot, j, physicsClientId=self.sim.client)[1]
                         for j in range(7)], float)
        pos, quat = self.sim.pose()
        fingers = np.asarray([p.getJointState(self.robot, j, physicsClientId=self.sim.client)[0]
                              for j in (9, 10)], float)
        fvel = np.asarray([p.getJointState(self.robot, j, physicsClientId=self.sim.client)[1]
                           for j in (9, 10)], float)
        return {'q': q, 'dq': dq, 'ee_pos': np.asarray(pos, float), 'ee_quat': np.asarray(quat, float),
                'fingers': fingers, 'finger_vel': fvel}

    def contacts(self, body_a: int, body_b: int) -> list:
        import pybullet as p

        return p.getContactPoints(bodyA=body_a, bodyB=body_b, physicsClientId=self.sim.client)

    def close(self):
        import pybullet as p

        if p.isConnected(self.sim.client):
            p.disconnect(self.sim.client)


def build_a_scene(goal: AGoal, *, seed: int, jitter: float = 0.008, robot_q=None,
                  camera_version: str = 'v3') -> Scene41:
    """A90 场景：物体在标准起点（长轴 Y），带小抖动；机器人默认 HOME。"""
    rng = np.random.default_rng(int(seed))
    # A90 的起点 y 固定，与放置目标无关（否则就不是"多 A 终态"而是多起点）
    pos = (START[0] + rng.uniform(-jitter, jitter), START[1] + rng.uniform(-jitter, jitter),
           BLOCK_HALF_Z + 1e-5)
    yaw = float(rng.uniform(-0.05, 0.05))
    return Scene41('A90', obj_pos=pos, obj_yaw=yaw, robot_q=robot_q, camera_version=camera_version)


def build_b_scene(obj_start: dict, robot_start: dict, *, seed: int = 0,
                  camera_version: str = 'v3') -> Scene41:
    """B90 场景：物体按给定启动条件放置，机器人按给定构型启动。"""
    rng = np.random.default_rng(int(seed))
    pos = (float(obj_start['x']) + rng.uniform(-0.003, 0.003),
           float(obj_start['y']) + rng.uniform(-0.003, 0.003),
           BLOCK_HALF_Z + 1e-5)
    yaw = float(obj_start['yaw']) + rng.uniform(-0.05, 0.05)
    return Scene41('B90', obj_pos=pos, obj_yaw=yaw,
                   robot_q=np.asarray(robot_start['q'], float),
                   gripper=float(robot_start['gripper']), camera_version=camera_version)
