"""实验 0C：可移动工件、真实电机夹爪原语与接触/位姿查询。

本模块不包含任何瞬移物体或固定约束的代码。工件只能通过夹爪的接触
被抬起和搬运；`resetBasePositionAndOrientation` 只在 episode 初始化时
用于把工件摆到固定起点。

夹爪是两个 prismatic 手指关节（link 9/10），URDF 的每指最大力为 20 N、
行程 0~0.04 m（总开口 0~0.08 m）。所有夹持都由 POSITION_CONTROL 电机
产生，夹持力上限不会超过 URDF 限制。
"""
import numpy as np
import pybullet as p

FINGER_LINKS = (9, 10)
HAND_LINK = 8
FINGER_MAX_FORCE = 20.0
FINGER_MAX_WIDTH = 0.04


def create_block(sim, half_extents, mass, position, orientation=(0, 0, 0, 1),
                 lateral_friction=1.2, spinning_friction=0.05, rolling_friction=0.01,
                 linear_damping=0.05, angular_damping=0.05, color=(0.85, 0.35, 0.10, 1.0)):
    """创建可自由移动的长方体（无固定底座）。"""
    half = [float(v) for v in half_extents]
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=sim.client)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=list(color),
                                 physicsClientId=sim.client)
    body = p.createMultiBody(float(mass), collision, visual, basePosition=list(position),
                            baseOrientation=list(orientation), physicsClientId=sim.client)
    p.changeDynamics(body, -1, lateralFriction=float(lateral_friction),
                     spinningFriction=float(spinning_friction),
                     rollingFriction=float(rolling_friction), restitution=0.0,
                     linearDamping=float(linear_damping), angularDamping=float(angular_damping),
                     physicsClientId=sim.client)
    return body


def enable_contact_solver(sim):
    """0C 的抓取依赖真实的摩擦锥接触，而不是外加约束。"""
    p.setPhysicsEngineParameter(enableConeFriction=1, physicsClientId=sim.client)


def set_gripper(sim, width, force=FINGER_MAX_FORCE, max_velocity=0.05):
    """把两指电机目标设为 `width`（每指位移，总开口 = 2*width）。"""
    width = float(np.clip(width, 0.0, FINGER_MAX_WIDTH))
    force = float(np.clip(force, 0.0, FINGER_MAX_FORCE))
    for joint in FINGER_LINKS:
        p.setJointMotorControl2(sim.robot, joint, p.POSITION_CONTROL, targetPosition=width,
                                force=force, maxVelocity=float(max_velocity),
                                physicsClientId=sim.client)


def finger_widths(sim):
    states = p.getJointStates(sim.robot, list(FINGER_LINKS), physicsClientId=sim.client)
    return tuple(float(s[0]) for s in states)


def finger_tips(sim):
    tips = []
    for link in FINGER_LINKS:
        state = p.getLinkState(sim.robot, link, computeForwardKinematics=True,
                               physicsClientId=sim.client)
        tips.append(np.asarray(state[4], dtype=float))
    return tips


def object_state(sim, body):
    position, quaternion = p.getBasePositionAndOrientation(body, physicsClientId=sim.client)
    linear, angular = p.getBaseVelocity(body, physicsClientId=sim.client)
    return {
        'position': np.asarray(position, dtype=float),
        'quaternion': np.asarray(quaternion, dtype=float),
        'linear_speed': float(np.linalg.norm(linear)),
        'angular_speed': float(np.linalg.norm(angular)),
        'tilt_deg': tilt_deg(quaternion),
    }


def tilt_deg(quaternion):
    """工件局部 z 轴相对世界 z 轴的倾角（度）。平放时接近 0。"""
    matrix = np.asarray(p.getMatrixFromQuaternion(quaternion), dtype=float).reshape(3, 3)
    local_z = matrix[:, 2]
    return float(np.degrees(np.arccos(np.clip(local_z[2], -1.0, 1.0))))


def contact_points(sim, body):
    """工件与机械臂之间的全部接触（含零力接触）。"""
    contacts = p.getContactPoints(bodyA=body, bodyB=sim.robot, physicsClientId=sim.client)
    result = []
    for c in contacts:
        result.append({
            'link': int(c[4]),
            'normal_force': float(c[9]),
            'position': np.asarray(c[5], dtype=float),
            'normal': np.asarray(c[7], dtype=float),
            'distance': float(c[8]),
        })
    return result


def finger_contact_forces(sim, body):
    """按手指 link 汇总法向力；返回 (f1, f2, links_with_contact)。"""
    forces = {link: 0.0 for link in FINGER_LINKS}
    for c in contact_points(sim, body):
        if c['link'] in forces and c['normal_force'] > 0.0:
            forces[c['link']] += c['normal_force']
    active = tuple(link for link in FINGER_LINKS if forces[link] > 0.0)
    return forces[FINGER_LINKS[0]], forces[FINGER_LINKS[1]], active


def non_finger_contacts(sim, body):
    """工件与 hand/臂本的接触（抓取时视为非预期碰撞）。"""
    return [c for c in contact_points(sim, body)
            if c['link'] not in FINGER_LINKS and c['normal_force'] > 0.0]


def relative_pose(sim, body):
    """工件在夹爪参考系中的位姿，用于测量运输中的滑移。"""
    obj_pos, obj_quat = p.getBasePositionAndOrientation(body, physicsClientId=sim.client)
    ee_pos, ee_quat = sim.pose()
    inv_pos, inv_quat = p.invertTransform(ee_pos, ee_quat)
    local_pos, local_quat = p.multiplyTransforms(inv_pos, inv_quat, obj_pos, obj_quat)
    return np.asarray(local_pos, dtype=float), np.asarray(local_quat, dtype=float)
