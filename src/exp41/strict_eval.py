"""实验 4.1 统一严格物理评价器（exp4.1-strict-v1）。

与旧版 v2.2 的区别：
  * **真实接触**：在每个 240 Hz 物理步记录 block↔wall / robot↔block / block↔table / block↔tray
    接触（不再只看控制周期末尾，也不再用"离墙 < 2 cm"替代接触）；
  * **权威 body ID**：block/wall/tray/plane 都来自 `Scene41`（即 `build_scene()` 的返回值）；
  * **失败可解释**：`failure_reason` 是枚举值，且保留全部连续指标；
  * **稳定窗口**：专家结束后单独观察，检查速度、滑移、反弹、支撑。

接触法向约定（必须验证，不能凭印象）：PyBullet `getContactPoints` 的
`contactNormalOnB` 是**从 B 指向 A** 的单位向量。评价器对 `bodyA=block, bodyB=wall`
记录该法向，并用正例自检（把方块贴到墙内侧推进）确认"推墙时 n_x < 0"，再写入配置。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field

import numpy as np

from vla.atomic_tasks.scene2 import WALL_INNER_X

CONFIG_VERSION = 'exp4.1-strict-v1.1'


@dataclass(frozen=True)
class StrictEvalConfig:
    version: str = CONFIG_VERSION
    # ---- 通用 ----
    settle_seconds: float = 0.8           # 专家结束后的稳定性观察窗口
    speed_max: float = 0.01               # m/s（物体线速度）
    ang_speed_max: float = 0.5            # rad/s
    penetration_max_m: float = 0.002      # 允许的数值穿透
    # ---- A90 ----
    a_pos_tol_m: float = 0.02
    a_yaw_tol_deg: float = 12.0
    a_tilt_max_deg: float = 12.0
    a_min_lift_m: float = 0.05
    a_min_transport_m: float = 0.10
    a_slip_max_m: float = 0.01
    a_ang_speed_max: float = 0.5          # rad/s，作用到物体的角速度上限
    a_grasp_force_min_n: float = 0.05     # 每个指尖的最小有效法向力
    a_grasp_min_steps: int = 5            # 每个指尖至少持续接触的物理步数
    a_release_grip_min_m: float = 0.030   # 释放后单指开口
    a_retreat_min_m: float = 0.10         # 结束时 EE 离物体距离
    # ---- B90 ----
    b_wall_contact_min_steps: int = 5     # 每个物理步都计数
    b_wall_force_min_n: float = 0.05      # 有效接触的最小法向力
    b_normal_x_min: float = 0.9           # |n_x| 下限（法向与墙面几何相容）
    b_advance_min_m: float = 0.05         # 有效 +X 位移
    b_lateral_tol_m: float = 0.02         # 最终横向漂移
    b_yaw_tol_deg: float = 15.0
    b_tilt_max_deg: float = 12.0
    b_goal_half_width_m: float = 0.03     # 目标区中心容差（目标区本身 ±30 mm）
    b_terminal_gap_max_m: float = 0.006   # 终止时方块前面离墙的最大间隙
    b_gap_min_m: float = -0.001           # 不允许穿模
    b_rebound_max_m: float = 0.005        # 稳定窗口内允许的最大回退


DEFAULT_CONFIG = StrictEvalConfig()


# ------------------------------------------------------------------ 监测器
class PhysicsMonitor:
    """包住 `sim.step()`，在每个物理步记录接触与物体状态。"""

    def __init__(self, scene, wall_body: int | None = None):
        self.scene = scene
        self.block = scene.block
        self.robot = scene.robot
        self.wall = int(wall_body if wall_body is not None else scene.wall)
        self.tray = [int(b) for b in (scene.trays if isinstance(scene.trays, (list, tuple)) else [scene.trays])]
        self.plane = scene.plane
        self.steps = 0
        self.events = {
            'block_wall': [], 'robot_block': [], 'block_plane': [], 'block_tray': [],
            'robot_wall': [],
        }
        self.track = []          # 每物理步的物体状态（用于回退/漂移/抬升判定）
        self._orig_step = None

    # ---- 生命周期 ----
    def attach(self):
        sim = self.scene.sim
        self._orig_step = sim.step

        def logged_step():
            self._orig_step()
            self.steps += 1
            self._log()

        sim.step = logged_step
        return self

    def detach(self):
        if self._orig_step is not None:
            self.scene.sim.step = self._orig_step
            self._orig_step = None

    # ---- 记录 ----
    def _log(self):
        import pybullet as p

        cid = self.scene.sim.client
        # 物体状态
        pos, quat = p.getBasePositionAndOrientation(self.block, physicsClientId=cid)
        euler = p.getEulerFromQuaternion(quat)
        lin, ang = p.getBaseVelocity(self.block, physicsClientId=cid)
        self.track.append((self.steps, float(pos[0]), float(pos[1]), float(pos[2]),
                           math.degrees(abs(euler[0])) + math.degrees(abs(euler[1])),
                           float(np.linalg.norm(lin)), float(np.linalg.norm(ang))))

        def rec(name, body_a, body_b, keep_normal=True):
            for c in p.getContactPoints(bodyA=body_a, bodyB=body_b, physicsClientId=cid):
                self.events[name].append({
                    'step': self.steps,
                    'link_a': int(c[3]), 'link_b': int(c[4]),
                    'pos': [float(v) for v in c[5]],
                    'normal_b_to_a': [float(v) for v in c[7]],
                    'distance': float(c[8]),
                    'normal_force': float(c[9]),
                    'lateral_friction_1': float(c[10]),
                    'lateral_friction_2': float(c[12]),
                })

        rec('block_wall', self.block, self.wall)
        rec('robot_block', self.block, self.robot)
        rec('block_plane', self.block, self.plane)
        for t in self.tray:
            rec('block_tray', self.block, t)
        rec('robot_wall', self.robot, self.wall)

    # ---- 汇总 ----
    def finger_contact_stats(self, min_force: float = 0.0) -> dict:
        """每个指尖（robot link 9 / 10）各自的接触步数与最大法向力。

        `getContactPoints(bodyA=block, bodyB=robot)` 里 `link_b` 是 robot 的 link index，
        因此用手指 link 过滤即可区分左右指；review #4 要求"双指"而不是"任意机器人接触"。
        """
        out = {9: {'steps': 0, 'max_force': 0.0, 'n_entries': 0},
               10: {'steps': 0, 'max_force': 0.0, 'n_entries': 0}}
        seen = {9: set(), 10: set()}
        for e in self.events['robot_block']:
            link = int(e['link_b'])
            if link in out and e['normal_force'] >= min_force:
                seen[link].add(e['step'])
                out[link]['max_force'] = max(out[link]['max_force'], float(e['normal_force']))
                out[link]['n_entries'] += 1
        for link in out:
            out[link]['steps'] = len(seen[link])
        return out

    def tail(self, n_steps: int = 48) -> dict:
        """稳定窗口尾段的物体位移（slip）与最大角速度。"""
        if not self.track:
            return {'n_steps': 0, 'slip_m': 0.0, 'ang_speed_max': 0.0}
        arr = np.asarray(self.track[-max(1, n_steps):], dtype=float)
        p0 = arr[0, 1:4]
        disp = np.linalg.norm(arr[:, 1:4] - p0[None, :], axis=1)
        return {'n_steps': int(arr.shape[0]), 'slip_m': float(disp.max()),
                'ang_speed_max': float(arr[:, 6].max())}

    def summary(self) -> dict:
        def agg(name):
            ev = self.events[name]
            return {
                'n_steps': len({e['step'] for e in ev}),
                'n_entries': len(ev),
                'first_step': ev[0]['step'] if ev else None,
                'last_step': ev[-1]['step'] if ev else None,
                'max_normal_force': max((e['normal_force'] for e in ev), default=0.0),
                'min_distance': min((e['distance'] for e in ev), default=None),
                'links_a': sorted({e['link_a'] for e in ev}),
                'links_b': sorted({e['link_b'] for e in ev}),
                'links': sorted({e['link_b'] for e in ev}),
                'normals': [e['normal_b_to_a'] for e in ev[-5:]],
            }

        out = {name: agg(name) for name in self.events}
        if self.track:
            arr = np.asarray(self.track, dtype=float)
            out['object_track'] = {
                'n_steps': int(arr.shape[0]),
                'x_min': float(arr[:, 1].min()), 'x_max': float(arr[:, 1].max()),
                'y_min': float(arr[:, 2].min()), 'y_max': float(arr[:, 2].max()),
                'z_max': float(arr[:, 3].max()),
                'tilt_max_deg': float(arr[:, 4].max()),
                'speed_max': float(arr[:, 5].max()),
                'ang_speed_max': float(arr[:, 6].max()),
            }
        return out


# ------------------------------------------------------------------ 工具
def axis_error_x(yaw: float) -> float:
    a = abs(math.atan2(math.sin(yaw), math.cos(yaw))) % math.pi
    return float(abs(a - math.pi / 2))


def axis_error_y(yaw: float) -> float:
    a = abs(math.atan2(math.sin(yaw), math.cos(yaw))) % math.pi
    return float(min(a, math.pi - a))


def block_depth(yaw: float) -> float:
    """方块沿 X 的半宽（随 yaw 变化）。"""
    bx, by = 0.02, 0.05
    return bx * abs(math.cos(yaw)) + by * abs(math.sin(yaw))


def wall_gap(block_pos, yaw: float, wall_inner_x: float) -> float:
    return float(wall_inner_x - (float(block_pos[0]) + block_depth(yaw)))


# ------------------------------------------------------------------ 评价
def evaluate_initial_B(scene, cfg: StrictEvalConfig = DEFAULT_CONFIG) -> dict:
    """B 的初始合法性：不贴墙、不穿模、稳定、任务未初始完成。"""
    st = scene.object_state()
    gap = wall_gap(st['pos'], st['yaw'], WALL_INNER_X)
    pw = scene.contacts(scene.block, scene.wall)
    pt = scene.contacts(scene.block, scene.plane)
    min_pen = min((c[8] for c in pt), default=0.0)
    speed = float(np.linalg.norm(st['lin_vel']))
    ok = (gap > 0.005 and not pw and min_pen > -cfg.penetration_max_m and speed < cfg.speed_max)
    reasons = []
    if gap <= 0.005:
        reasons.append('initial_too_close_to_wall')
    if pw:
        reasons.append('initial_already_touching_wall')
    if min_pen <= -cfg.penetration_max_m:
        reasons.append('initial_penetration')
    if speed >= cfg.speed_max:
        reasons.append('initial_unstable')
    return {'task_valid': not reasons, 'reasons': reasons, 'initial_gap_m': gap,
            'initial_penetration_m': float(min_pen), 'initial_speed': speed,
            'initial_object': {'pos': st['pos'].tolist(), 'quat': st['quat'].tolist(),
                               'yaw': st['yaw']}}


def evaluate_A(scene, monitor: PhysicsMonitor, goal, cfg: StrictEvalConfig = DEFAULT_CONFIG,
               *, start_obj_pos=None, settle_state=None) -> dict:
    """A90 严格评价：真实抓取 → 抬升 → 搬运 → 放置 → 释放 → 撤离 → 桌面支撑 → 稳定。"""
    mon = monitor.summary()
    st = settle_state or scene.object_state()
    rs = scene.robot_state()
    pen_table = mon['block_plane']['min_distance']
    pen_table = 0.0 if pen_table is None else float(pen_table)
    robot_block_steps = mon['robot_block']['n_steps']
    contacts_end = scene.contacts(scene.block, scene.robot)
    fingers = rs['fingers']
    ee_to_obj = float(np.linalg.norm(rs['ee_pos'] - st['pos']))
    lift = float(mon.get('object_track', {}).get('z_max', st['pos'][2]) - (start_obj_pos[2] if start_obj_pos is not None else st['pos'][2]))
    transport = float(np.linalg.norm(np.asarray(st['pos'][:2]) - np.asarray(start_obj_pos[:2]))) if start_obj_pos is not None else 0.0
    pos_err = float(np.linalg.norm(np.asarray(st['pos'][:2]) - np.asarray([goal.x, goal.y])))
    yaw_err = axis_error_x(st['yaw']) if goal.target_axis == 'x' else axis_error_y(st['yaw'])
    tilt = abs(st['roll_deg']) + abs(st['pitch_deg'])
    speed = float(np.linalg.norm(st['lin_vel']))
    ang_speed = float(np.linalg.norm(st['ang_vel']))
    supported = len(scene.contacts(scene.block, scene.plane)) > 0

    # ---- 双指抓取：两个指尖各自都要有"足够多步 + 足够力"的接触（review #4）----
    finger_stats = monitor.finger_contact_stats(cfg.a_grasp_force_min_n)
    two_finger = all(finger_stats[f]['steps'] >= cfg.a_grasp_min_steps for f in (9, 10))
    # ---- 释放后的稳定/滑移：只看稳定窗口尾段（最后 0.2 s）----
    tail = monitor.tail(n_steps=int(0.2 * 240))
    slip = tail['slip_m']
    ang_speed_tail = tail['ang_speed_max']

    checks = {
        'grasped_by_two_fingers': bool(two_finger),
        'lifted': lift >= cfg.a_min_lift_m,
        'transported': transport >= cfg.a_min_transport_m,
        'position_ok': pos_err <= cfg.a_pos_tol_m,
        'yaw_ok': yaw_err <= math.radians(cfg.a_yaw_tol_deg),
        'tilt_ok': tilt <= cfg.a_tilt_max_deg,
        'released': (not contacts_end) and float(fingers.min()) >= cfg.a_release_grip_min_m,
        'retreated': ee_to_obj >= cfg.a_retreat_min_m,
        'table_supported': supported,
        'stable': speed < cfg.speed_max,
        'ang_speed_ok': ang_speed_tail <= cfg.a_ang_speed_max,
        'no_slip_after_release': slip <= cfg.a_slip_max_m,
        'no_penetration': pen_table > -cfg.penetration_max_m,
    }
    strict = all(checks.values())
    reason = None
    if not strict:
        order = ['grasped_by_two_fingers', 'lifted', 'transported', 'position_ok', 'yaw_ok',
                 'tilt_ok', 'released', 'retreated', 'table_supported', 'stable',
                 'ang_speed_ok', 'no_slip_after_release', 'no_penetration']
        reason = next(k for k in order if not checks[k])
    return {
        'eval_version': cfg.version, 'skill': 'A90', 'goal_id': goal.goal_id,
        'task_valid': True, 'physical_execution_valid': robot_block_steps > 0,
        'expert_completed': strict, 'strict_success': strict, 'failure_reason': reason,
        'checks': checks,
        'metrics': {
            'position_error_m': pos_err, 'yaw_error_rad': yaw_err, 'yaw_error_deg': math.degrees(yaw_err),
            'tilt_deg': tilt, 'lift_m': lift, 'transport_m': transport,
            'final_speed_m_s': speed, 'final_ang_speed_rad_s': ang_speed,
            'ee_object_distance_m': ee_to_obj,
            'gripper_width_m': float(fingers.mean()), 'robot_block_contact_steps': robot_block_steps,
            'finger9_contact_steps': finger_stats[9]['steps'],
            'finger9_max_force_n': finger_stats[9]['max_force'],
            'finger10_contact_steps': finger_stats[10]['steps'],
            'finger10_max_force_n': finger_stats[10]['max_force'],
            'slip_after_release_m': slip, 'ang_speed_tail_max_rad_s': ang_speed_tail,
            'table_penetration_m': pen_table, 'n_control_steps': None,
            'obj_terminal_pos': st['pos'].tolist(), 'obj_terminal_quat': st['quat'].tolist(),
            'obj_terminal_yaw': st['yaw'], 'robot_q_terminal': rs['q'].tolist(),
            'ee_terminal_pos': rs['ee_pos'].tolist(), 'ee_terminal_quat': rs['ee_quat'].tolist(),
        },
        'contact_summary': mon,
    }


def evaluate_B(scene, monitor: PhysicsMonitor, goal, cfg: StrictEvalConfig = DEFAULT_CONFIG,
               *, start_obj_pos=None, settle_state=None, initial: dict | None = None,
               settle_track: dict | None = None) -> dict:
    """B90 严格评价：合法初始 → 有效推动 → **真实墙面接触** → 目标区 → 姿态 → 稳定结束。"""
    mon = monitor.summary()
    st = settle_state or scene.object_state()
    rs = scene.robot_state()
    start = np.asarray(start_obj_pos if start_obj_pos is not None else st['pos'], dtype=float)
    advance = float(st['pos'][0] - start[0])
    lateral = float(abs(st['pos'][1] - start[1]))
    gap = wall_gap(st['pos'], st['yaw'], goal.wall_inner_x)
    yaw_change = abs(math.atan2(math.sin(st['yaw'] - (initial or {}).get('yaw', st['yaw'])),
                                math.cos(st['yaw'] - (initial or {}).get('yaw', st['yaw'])))) \
        if initial else 0.0
    tilt = abs(st['roll_deg']) + abs(st['pitch_deg'])
    speed = float(np.linalg.norm(st['lin_vel']))
    wall_ever = mon['block_wall']['n_steps'] > 0
    wall_valid = (mon['block_wall']['n_steps'] >= cfg.b_wall_contact_min_steps
                  and mon['block_wall']['max_normal_force'] >= cfg.b_wall_force_min_n)
    normal_x = [abs(e[0]) for e in mon['block_wall']['normals']]
    normal_ok = bool(normal_x) and min(normal_x) >= cfg.b_normal_x_min
    supported = len(scene.contacts(scene.block, scene.plane)) > 0
    rebound = 0.0
    if settle_track is not None and settle_track.get('x_max') is not None:
        rebound = float(settle_track['x_max'] - st['pos'][0])
    y_in_goal = bool((goal.y_lo - cfg.b_goal_half_width_m) <= float(st['pos'][1])
                     <= (goal.y_hi + cfg.b_goal_half_width_m))
    advance_ok = bool(advance >= cfg.b_advance_min_m)
    # 横向判据必须是**目标相关**的：若起点不在目标区内，专家需要主动横向移动，
    # 此时"相对起点的横向漂移 ≤ 2 cm"会误判。只有在起点本来就在目标区时才检查漂移。
    start_y = float(start[1])
    lateral_needed = not (goal.y_lo <= start_y <= goal.y_hi)
    lateral_ok = bool(True if lateral_needed else lateral <= cfg.b_lateral_tol_m)
    gap_ok = bool(cfg.b_gap_min_m <= gap <= cfg.b_terminal_gap_max_m)
    stable_ok = bool(speed < cfg.speed_max)
    supported_ok = bool(supported)

    checks = {
        'initial_valid': bool((initial or {}).get('task_valid', True)),
        'pushed_by_robot': bool(mon['robot_block']['n_steps'] > 0),
        'advance_ok': advance_ok,
        'wall_contact_ever': bool(wall_ever),
        'valid_wall_contact': bool(wall_valid),
        'wall_normal_compatible': bool(normal_ok),
        'terminal_gap_ok': gap_ok,
        'lateral_ok': lateral_ok,
        'goal_region_ok': y_in_goal,
        'yaw_ok': bool(yaw_change <= math.radians(cfg.b_yaw_tol_deg)),
        'tilt_ok': bool(tilt <= cfg.b_tilt_max_deg),
        'table_supported': supported_ok,
        'stable': stable_ok,
        'no_rebound': bool(rebound <= cfg.b_rebound_max_m),
    }
    strict = all(checks.values())
    reason = None
    if not strict:
        order = ['initial_valid', 'pushed_by_robot', 'advance_ok', 'wall_contact_ever',
                 'valid_wall_contact', 'wall_normal_compatible', 'terminal_gap_ok',
                 'goal_region_ok', 'lateral_ok', 'yaw_ok', 'tilt_ok', 'table_supported',
                 'stable', 'no_rebound']
        reason = next(k for k in order if not checks[k])
    return {
        'eval_version': cfg.version, 'skill': 'B90', 'goal_id': goal.goal_id,
        'task_valid': bool((initial or {}).get('task_valid', True)),
        'physical_execution_valid': mon['robot_block']['n_steps'] > 0,
        'expert_completed': strict, 'strict_success': strict, 'failure_reason': reason,
        'checks': checks,
        'metrics': {
            'advance_x_m': advance, 'lateral_drift_m': lateral, 'terminal_gap_m': gap,
            'lateral_needed_for_goal': bool(lateral_needed),
            'wall_contact_steps': mon['block_wall']['n_steps'],
            'wall_contact_force_max_n': mon['block_wall']['max_normal_force'],
            'wall_normal_abs_x_min': (min(normal_x) if normal_x else None),
            'robot_block_contact_steps': mon['robot_block']['n_steps'],
            'robot_wall_contact_steps': mon['robot_wall']['n_steps'],
            'goal_y_range': [goal.y_lo, goal.y_hi], 'final_y': float(st['pos'][1]),
            'yaw_change_rad': yaw_change, 'tilt_deg': tilt, 'final_speed_m_s': speed,
            'rebound_m': rebound,
            'obj_terminal_pos': st['pos'].tolist(), 'obj_terminal_quat': st['quat'].tolist(),
            'obj_terminal_yaw': st['yaw'], 'robot_q_terminal': rs['q'].tolist(),
            'ee_terminal_pos': rs['ee_pos'].tolist(), 'ee_terminal_quat': rs['ee_quat'].tolist(),
            'gripper_width_m': float(rs['fingers'].mean()),
        },
        'contact_summary': mon,
    }


def empty_config() -> dict:
    return asdict(DEFAULT_CONFIG)


def dump(obj, path) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)
