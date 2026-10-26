"""实验 2.3 阶段 1：独立原子技能（B 优先）的物理合法多起点生成器。

设计目标（对应 prompt.md 第五阶段）：

1. 不再只从 HOME 出发，而是在**物理有效**的状态空间里采样机器人起始构型；
2. 每个候选起点必须通过：关节限位余量、静态碰撞、夹爪-物体间距、
   相机观察条件、到 B 首次接近位姿的 IK 可达性与关节路径粗检；
3. 覆盖三类来源：
   - ``home_neighborhood``：HOME 邻域关节扰动（小幅构型泛化）；
   - ``multi_ik``：不同末端位置/姿态 + 不同 IK 分支（结构化工作空间覆盖）；
   - ``task_relevant``：推动走廊附近的预接近构型（任务相关但非前驱技能状态）；
4. 只生成"独立原子任务的合法起点"，不读取 A 的终态、不复制测试组合状态。

注意：本模块只做**静态合法性**检查与关节路径粗检，真正的执行可行性由
B 专家冒烟（scripts/smoke_multistart_b.py）验证；两者不能混为一谈。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from simulation import HOME
from vla.atomic_tasks.scene2 import block_yaw, grasp_quat

# 与采集脚本一致：每根手指 0.035（总开口约 7 cm），VLA 观测里的起始夹爪状态。
START_FINGER = 0.035
# 关节限位安全余量（rad）。太靠近限位的构型不进入数据。
LIMIT_MARGIN = 0.12
# 起点处夹爪中心到物体中心的最小距离（m）：必须还有接近空间，不能已经接触。
MIN_BLOCK_CLEARANCE = 0.06
# 相机像素坐标检查用的渲染尺寸（只做投影，不真正渲染）。
CAM_SIZE = (128, 128)


@dataclass
class Candidate:
    mode: str
    q: np.ndarray
    seed: int
    ee_target: list[float] | None = None
    gripper_yaw: float = 0.0
    meta: dict = field(default_factory=dict)


def home_array() -> np.ndarray:
    return np.asarray(HOME, dtype=float)


def set_robot_state(sim, q, finger: float = START_FINGER) -> None:
    """把机器人**初始化**到给定构型（只允许在 episode 开始前调用）。"""
    q = np.asarray(q, dtype=float)
    for j in range(7):
        p.resetJointState(sim.robot, j, float(q[j]), physicsClientId=sim.client)
    for j in (9, 10):
        p.resetJointState(sim.robot, j, float(finger), physicsClientId=sim.client)
    p.performCollisionDetection(physicsClientId=sim.client)


def robot_contacts(sim) -> list[dict]:
    """静态检查用：返回当前机器人所有非预期接触（允许 base-地面）。"""
    return sim.collisions(allowed=set())


def joint_margin(q, lower, upper) -> float:
    return float(np.min(np.minimum(np.asarray(q) - lower, upper - np.asarray(q))))


def ik_solve(sim, position, orientation, rest) -> tuple[np.ndarray, bool]:
    result = p.calculateInverseKinematics(
        sim.robot, sim.ee, list(map(float, position)), list(map(float, orientation)),
        lowerLimits=sim.lower.tolist(), upperLimits=sim.upper.tolist(),
        jointRanges=(sim.upper - sim.lower).tolist(), restPoses=list(map(float, rest)),
        maxNumIterations=300, residualThreshold=1e-5, physicsClientId=sim.client,
    )
    q = np.asarray(result[:7], dtype=float)
    ok = bool(np.all(q >= sim.lower - 1e-6) and np.all(q <= sim.upper + 1e-6))
    return q, ok


def ee_position(sim) -> np.ndarray:
    state = p.getLinkState(sim.robot, sim.ee, computeForwardKinematics=True,
                           physicsClientId=sim.client)
    return np.asarray(state[4], dtype=float)


def _project(point, view, proj, width, height) -> tuple[bool, float, float, float]:
    """把世界点投影到像素；返回 (是否在视锥内, u, v, ndc_z)。"""
    # PyBullet 的 view/projection 是列主序（OpenGL 风格），与 numpy 行主序相反。
    clip = np.asarray(proj, dtype=float).reshape(4, 4).T @ \
        np.asarray(view, dtype=float).reshape(4, 4).T @ np.array([*point, 1.0])
    if abs(clip[3]) < 1e-9:
        return False, -1.0, -1.0, 0.0
    ndc = clip[:3] / clip[3]
    inside = bool(-1.0 <= ndc[0] <= 1.0 and -1.0 <= ndc[1] <= 1.0 and -1.0 <= ndc[2] <= 1.0)
    u = (ndc[0] + 1.0) * 0.5 * width
    v = (1.0 - ndc[1]) * 0.5 * height
    return inside, float(u), float(v), float(ndc[2])


def camera_visibility(sim, block, rig) -> dict:
    """记录物体中心在 top / wrist 相机中的投影（不做遮挡渲染）。

    top 是固定相机，B 的任何合法起点都应该看得到物体；
    wrist 在 HOME 起点本来就看不到物体（原始数据也如此），因此只记录、不硬性拒绝。
    """
    pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    pos = np.asarray(pos, dtype=float)
    out = {}
    for name in ('top', 'wrist'):
        if name == 'top':
            view, proj = rig._static[name]
        else:
            view, proj = rig._wrist_matrices(rig.cameras[name])
        inside, u, v, z = _project(pos, view, proj, CAM_SIZE[0], CAM_SIZE[1])
        out[name] = {'inside': inside, 'u': round(u, 1), 'v': round(v, 1),
                     'ndc_z': round(z, 4)}
    return out


def _sample_home(rng, sim, block_pos, yaw_g, n) -> Candidate:
    """HOME 对照：与实验 2 采集/评测完全一致的起始构型。"""
    return Candidate('home', home_array(), int(rng.integers(1, 2 ** 31 - 1)),
                     gripper_yaw=float(yaw_g))


def _sample_home_neighborhood(rng, sim, block_pos, yaw_g, n) -> Candidate:
    scales = np.array([0.06, 0.22, 0.10, 0.18, 0.20, 0.28, 0.42])
    q = home_array() + rng.uniform(-1.0, 1.0, 7) * scales
    return Candidate('home_neighborhood', q, int(rng.integers(1, 2 ** 31 - 1)),
                     gripper_yaw=float(yaw_g),
                     meta={'scale': scales.tolist()})


def _sample_multi_ik(rng, sim, block_pos, yaw_g, n) -> Candidate:
    by = float(block_pos[1])
    pos = np.array([rng.uniform(0.30, 0.50), by + rng.uniform(-0.10, 0.10),
                    rng.uniform(0.15, 0.28)])
    yaw = float(rng.choice([0.0, math.pi / 2, -math.pi / 2]))
    flip = -1.0 if rng.random() < 0.5 else 1.0
    rest = home_array() + np.array([
        rng.uniform(-0.15, 0.15), flip * rng.uniform(0.0, 0.35), rng.uniform(-0.12, 0.12),
        rng.uniform(-0.25, 0.25), rng.uniform(-0.20, 0.20), flip * rng.uniform(0.0, 0.40),
        rng.uniform(-0.80, 0.80)])
    q, ok = ik_solve(sim, pos, grasp_quat(yaw), rest)
    return Candidate('multi_ik', q, int(rng.integers(1, 2 ** 31 - 1)),
                     ee_target=pos.tolist(), gripper_yaw=yaw, meta={'ik_ok': ok,
                                                                   'rest': rest.tolist()})


def _sample_task_relevant(rng, sim, block_pos, yaw_g, n) -> Candidate:
    pos = np.array([rng.uniform(0.38, 0.52), float(block_pos[1]) + rng.uniform(-0.06, 0.06),
                    rng.uniform(0.15, 0.22)])
    yaw = float(rng.choice([0.0, math.pi / 2, -math.pi / 2]))
    rest = home_array() + np.array([
        rng.uniform(-0.10, 0.25), rng.uniform(0.10, 0.45), rng.uniform(-0.15, 0.15),
        rng.uniform(-0.20, 0.15), rng.uniform(-0.15, 0.20), rng.uniform(0.15, 0.55),
        rng.uniform(-0.60, 0.60)])
    q, ok = ik_solve(sim, pos, grasp_quat(yaw), rest)
    return Candidate('task_relevant', q, int(rng.integers(1, 2 ** 31 - 1)),
                     ee_target=pos.tolist(), gripper_yaw=yaw, meta={'ik_ok': ok,
                                                                    'rest': rest.tolist()})


SAMPLERS = {
    'home': _sample_home,
    'home_neighborhood': _sample_home_neighborhood,
    'multi_ik': _sample_multi_ik,
    'task_relevant': _sample_task_relevant,
}


def b_approach_target(block_pos) -> np.ndarray:
    """B 专家的第一个接近位姿（与 experts.SkillExpert.skill_b 保持一致）。"""
    return np.array([float(block_pos[0]) - 0.10, float(block_pos[1]), 0.07])


def path_static_ok(sim, q0, q1, steps: int = 24) -> bool:
    """关节空间直线路径的静态碰撞粗检。

    只做 resetJointState + performCollisionDetection，不推进动力学；
    因此它证明的是"扫掠体积没有明显穿模"，不能替代真实执行验证。
    """
    for a in np.linspace(0.0, 1.0, steps + 1):
        set_robot_state(sim, np.asarray(q0) * (1 - a) + np.asarray(q1) * a)
        if sim.collisions(allowed=set()):
            return False
    return True


def validate_start(sim, block, q, rig, block_pos=None) -> dict:
    """检查一个候选起点是否物理合法；返回证据字典（不修改调用方的数据）。"""
    q = np.asarray(q, dtype=float)
    lower, upper = np.asarray(sim.lower), np.asarray(sim.upper)
    margin = joint_margin(q, lower, upper)
    checks = {'q': q.tolist(), 'min_limit_margin': round(margin, 4)}

    # 1) 限位
    if margin < LIMIT_MARGIN:
        return {**checks, 'accepted': False, 'reject': 'joint_limit_margin'}

    # 2) 静态接触（任何非预期接触都拒绝；B 起点不允许已经碰物体）
    set_robot_state(sim, q)
    contacts = robot_contacts(sim)
    checks['static_contacts'] = contacts
    if contacts:
        bodies = {(c['body'], c['link_a']) for c in contacts}
        if any(c['body'] == block for c in contacts):
            return {**checks, 'accepted': False, 'reject': 'contact_with_block'}
        return {**checks, 'accepted': False, 'reject': 'static_collision'}

    # 3) 夹爪-物体间距
    if block_pos is None:
        block_pos, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    block_pos = np.asarray(block_pos, dtype=float)
    ee = ee_position(sim)
    clearance = float(np.linalg.norm(ee - block_pos))
    checks['ee_position'] = ee.tolist()
    checks['block_clearance_m'] = round(clearance, 4)
    if clearance < MIN_BLOCK_CLEARANCE:
        return {**checks, 'accepted': False, 'reject': 'too_close_to_block'}

    # 4) 物体稳定、无穿模（场景建好后物体应已静置）
    lin, ang = p.getBaseVelocity(block, physicsClientId=sim.client)
    euler = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(
        block, physicsClientId=sim.client)[1])
    tilt = abs(math.degrees(euler[0])) + abs(math.degrees(euler[1]))
    checks.update({'block_position': block_pos.tolist(),
                   'block_yaw_deg': round(math.degrees(block_yaw(block, sim.client)), 2),
                   'block_tilt_deg': round(tilt, 3),
                   'block_speed': round(float(np.linalg.norm(lin)), 6)})
    if tilt > 2.0 or float(np.linalg.norm(lin)) > 1e-3:
        return {**checks, 'accepted': False, 'reject': 'block_not_stable'}

    # 5) 相机观察条件
    checks['camera'] = camera_visibility(sim, block, rig)
    if not checks['camera']['top']['inside']:
        return {**checks, 'accepted': False, 'reject': 'top_camera_cannot_see_block'}

    # 6) 到 B 首次接近位姿的 IK 可达性
    target = b_approach_target(block_pos)
    q_ik, ik_ok = ik_solve(sim, target, grasp_quat(0.0), q)
    checks['ik_ok'] = bool(ik_ok)
    if not ik_ok:
        return {**checks, 'accepted': False, 'reject': 'b_approach_ik'}
    set_robot_state(sim, q_ik)
    err = float(np.linalg.norm(ee_position(sim) - target))
    checks['b_approach_ik_error_m'] = round(err, 4)
    if err > 0.02:
        return {**checks, 'accepted': False, 'reject': 'b_approach_ik_error'}

    # 7) 起点 -> 接近位姿的关节路径静态粗检
    if not path_static_ok(sim, q, q_ik):
        return {**checks, 'accepted': False, 'reject': 'b_approach_path_collision'}

    # 还原到起点，供真正的专家执行
    set_robot_state(sim, q)
    checks['accepted'] = True
    checks['reject'] = None
    return checks


def sample_valid_start(sim, block, rig, mode, rng, block_pos=None, max_tries=200):
    """在给定场景中采样一个通过合法性检查的起点。"""
    sampler = SAMPLERS[mode]
    rejections = {}
    for _ in range(max_tries):
        cand = sampler(rng, sim, block_pos, 0.0, 0)
        checks = validate_start(sim, block, cand.q, rig, block_pos=block_pos)
        if checks['accepted']:
            cand.meta['checks'] = checks
            return cand, checks, rejections
        rejections[checks['reject']] = rejections.get(checks['reject'], 0) + 1
    return None, None, rejections
