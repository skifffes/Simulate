"""实验 2 的统一场景：10×4×4 cm 长方体、+X 挡板、沿 Y 的狭长托盘。

坐标约定（与 docs/experiment_2_scene.md 一致，不得随意交换）：
    +X  桌面横向，B 把物体推向挡板的方向
    +Y  A 搬运物体的主要方向
    +Z  竖直向上

物体初始平放：长轴沿 Y（10 cm），短轴沿 X（4 cm），高 4 cm。
A0 = 保持长轴沿 Y 放置；A90 = 搬运中绕 Z 真实旋转 90°，终态长轴沿 X。
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from manipulation import create_block, enable_contact_solver
from simulation import Simulation

# ---- 物体 -----------------------------------------------------------------
BLOCK_HALF = (0.02, 0.05, 0.02)      # 4 cm (X) × 10 cm (Y) × 4 cm (Z)
BLOCK_MASS = 0.15
BLOCK_FRICTION = 1.2
START = (0.55, 0.0, 0.021)           # 底面贴桌，中心高 2 cm

# ---- A 的放置区 ------------------------------------------------------------
PLACE = (0.55, 0.18, 0.021)
GRASP_Z = 0.031                      # 指尖落在物体半高 0.02 m 处（指尖比 EE 低 1.1 cm）

# ---- B 的挡板 --------------------------------------------------------------
WALL_INNER_X = 0.78
WALL_HALF = (0.01, 0.15, 0.06)
WALL_POS = (WALL_INNER_X + WALL_HALF[0], 0.18, WALL_HALF[2])

# ---- C 的托盘 --------------------------------------------------------------
TRAY = (0.55, 0.40)
TRAY_RAIL_DX = 0.046                 # 两条导轨中心距托盘中心 ±4.6 cm
# 夹爪手掌宽约 7.1 cm，托盘内宽必须大于它，否则下放时手掌会撞导轨。
TRAY_RAIL_HALF = (0.006, 0.09, 0.012)
TRAY_INNER_GAP = 2 * (TRAY_RAIL_DX - TRAY_RAIL_HALF[0])   # 0.080 m

# ---- 夹爪姿态 --------------------------------------------------------------
# 朝下抓取（绕 X 翻转 180°），再绕 Z 转 yaw：
#   yaw=0    两指沿世界 Y 开合（夹 4 cm 短边，物体长轴沿 X）
#   yaw=±90  两指沿世界 X 开合（夹 4 cm 短边，物体长轴沿 Y）
DOWN_ORIENTATION = tuple(p.getQuaternionFromEuler([np.pi, 0.0, 0.0]))


def grasp_quat(yaw_rad: float):
    return p.getQuaternionFromEuler([np.pi, 0.0, float(yaw_rad)])


def block_yaw(block, client: int = 0) -> float:
    """物体的偏航角（rad），长轴沿 Y 时为 0。"""
    _, quat = p.getBasePositionAndOrientation(block, physicsClientId=client)
    euler = p.getEulerFromQuaternion(quat)
    return float(np.arctan2(np.sin(euler[2]), np.cos(euler[2])))


def make_wall(client):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(WALL_HALF),
                                       physicsClientId=client)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=list(WALL_HALF),
                                 rgbaColor=[0.45, 0.45, 0.5, 1.0], physicsClientId=client)
    return p.createMultiBody(0, collision, visual, basePosition=list(WALL_POS),
                             physicsClientId=client)


def make_tray(client):
    bodies = []
    for sign in (-1, 1):
        pos = (TRAY[0] + sign * TRAY_RAIL_DX, TRAY[1], TRAY_RAIL_HALF[2])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(TRAY_RAIL_HALF),
                                           physicsClientId=client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=list(TRAY_RAIL_HALF),
                                     rgbaColor=[0.20, 0.35, 0.75, 1.0], physicsClientId=client)
        bodies.append(p.createMultiBody(0, collision, visual, basePosition=list(pos),
                                        physicsClientId=client))
    return bodies


def build_scene(seed: int | None = None, jitter: float = 0.0, friction_jitter: float = 0.0,
                start=None, yaw: float = 0.0):
    """建场景；返回 (sim, block, wall, tray_bodies, info)。

    start/yaw 用于让 B/C 从“类 A 终态”的初始状态开始（独立原子任务，不是 A→B 轨迹）。
    """
    rng = np.random.default_rng(0 if seed is None else int(seed))
    base = np.array(START if start is None else start, dtype=float)
    if jitter > 0:
        base[0] += rng.uniform(-jitter, jitter)
        base[1] += rng.uniform(-jitter, jitter)
    friction = BLOCK_FRICTION
    if friction_jitter > 0:
        friction = float(np.clip(BLOCK_FRICTION + rng.uniform(-friction_jitter, friction_jitter),
                                 0.8, 2.0))
    sim = Simulation(gui=False, reference_cube=False)
    enable_contact_solver(sim)
    block = create_block(sim, BLOCK_HALF, BLOCK_MASS, tuple(base),
                         orientation=p.getQuaternionFromEuler([0.0, 0.0, float(yaw)]),
                         lateral_friction=friction)
    wall = make_wall(sim.client)
    tray = make_tray(sim.client)
    # B 的推动允许手掌/指侧接触物体：推不是抓，接触本身是任务要求。
    # A/C 仍用 non_finger_contacts 单独检查“手掌碰到物体”的抓取失败。
    sim.allowed_contacts = {(block, 8), (block, 9), (block, 10)}
    for _ in range(60):
        sim.step()
    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    info = {
        'seed': seed,
        'block_start': list(np.asarray(pos, dtype=float)),
        'block_yaw': block_yaw(block, sim.client),
        'block_friction': friction,
        'wall_inner_x': WALL_INNER_X,
        'tray': list(TRAY),
        'tray_inner_gap': TRAY_INNER_GAP,
    }
    return sim, block, wall, tray, info


__all__ = [
    'BLOCK_HALF', 'START', 'PLACE', 'GRASP_Z', 'WALL_INNER_X', 'TRAY',
    'TRAY_INNER_GAP', 'DOWN_ORIENTATION', 'grasp_quat', 'block_yaw',
    'build_scene', 'make_wall', 'make_tray',
]
