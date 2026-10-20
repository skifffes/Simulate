"""实验 0C 第一阶段：真实接触下的抓取、抬升、移动、放置与释放。

流程严格按电机指令执行：
    张开 -> 预抓取到位 -> 下降到抓取高度 -> 电机闭合夹紧 -> 抬升 10 cm
    -> 移动到放置点 -> 下降到放置高度 -> 张开释放 -> 撤离 -> 静置判定

物体在 episode 中只由夹爪接触和重力移动，没有任何瞬移或固定约束。
成功判定使用物体真实位姿/速度、双指接触力和释放后的静置状态。
"""
import numpy as np
import pybullet as p

from manipulation import (
    FINGER_MAX_FORCE, contact_points, finger_contact_forces, finger_widths,
    non_finger_contacts, object_state, relative_pose, set_gripper,
)

DEFAULT_PARAMS = {
    'block_half': (0.02, 0.02, 0.04),
    'block_mass': 0.12,
    'block_start': (0.55, 0.0, 0.041),
    'block_friction': 1.2,
    # 朝下抓取：夹爪 z 轴指向 -z，两指沿世界 y 轴开合。
    'orientation': tuple(p.getQuaternionFromEuler([np.pi, 0.0, 0.0])),
    'pre_grasp': (0.55, 0.0, 0.18),
    'grasp': (0.55, 0.0, 0.055),
    'lift': (0.55, 0.0, 0.155),
    'place_above': (0.45, 0.18, 0.155),
    'place': (0.45, 0.18, 0.055),
    'retreat': (0.45, 0.18, 0.24),
    'open_width': 0.04,
    'preclose_width': 0.024,
    'close_width': 0.017,
    'grasp_force': FINGER_MAX_FORCE,
    'open_force': FINGER_MAX_FORCE,
    'open_speed': 0.06,
    'close_speed': 0.05,
    'preclose_seconds': 0.6,
    'close_seconds': 0.8,
    'release_seconds': 0.8,
    'settle_seconds': 1.0,
    'lift_height_target': 0.08,
    'slip_tolerance_m': 0.02,
    'place_tolerance_m': 0.02,
    'height_tolerance_m': 0.012,
    'tilt_tolerance_deg': 12.0,
    'rest_linear_speed': 0.01,
    'rest_angular_speed': 0.10,
}

FAIL_TEXT = {
    'ik_joint_limits': 'IK 无满足关节限位的解',
    'joint_limit_during_motion': '运动中关节越过限位',
    'joint_limit_target': '目标关节角越界',
    'collision_during_motion': '运动中出现非预期碰撞',
    'tracking_error': '关节跟踪误差超限',
    'cartesian_pose_error': '末端位姿误差超限',
    'path_step_limit': '路径步数超限',
    'grasp_no_contact': '闭合后双指未同时接触工件',
    'hand_contact': '工件与手掌/臂本体接触',
    'object_not_lifted': '抬升后工件未被带起',
    'object_slipped': '运输中工件相对夹爪滑移超限',
    'place_position_error': '释放后工件偏离放置点',
    'place_height_error': '释放后工件高度异常',
    'object_tipped': '释放后工件倾角过大',
    'object_unstable': '静置后工件未停下',
    'not_released': '释放后夹爪仍与工件接触',
}


class GraspPlaceTask:
    """一次 0C 抓放 episode；observer 只观测，不参与控制。"""

    def __init__(self, sim, controller, block, params=None, on_step=None,
                 record_interval=4, contact_interval=8):
        self.sim, self.ctl, self.block = sim, controller, block
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.on_step = on_step
        self.record_interval = max(1, int(record_interval))
        self.contact_interval = max(1, int(contact_interval))
        self.samples, self.contacts, self.stages = [], [], []
        self.step0 = sim.steps
        self.initial = object_state(sim, block)
        self.grasp_local = None
        self.grasp_contact_record = {}
        self.lift_height = 0.0
        self.slip = 0.0
        self.max_rest_speed = 0.0
        self.hand_contact_samples = 0
        self.failure = None
        self.failure_stage = None
        self.closed_widths = (None, None)
        self._last_contact_sample = -10 ** 9

    # ---------- 记录 ----------
    def now(self):
        return (self.sim.steps - self.step0) * self.sim.dt

    def _sample(self, stage):
        state = object_state(self.sim, self.block)
        (f1, f2, active) = finger_contact_forces(self.sim, self.block)
        widths = finger_widths(self.sim)
        ee = self.sim.pose()[0]
        self.samples.append({
            't': self.now(),
            'stage': stage,
            'ee_x': float(ee[0]),
            'ee_y': float(ee[1]),
            'ee_z': float(ee[2]),
            'obj_x': float(state['position'][0]),
            'obj_y': float(state['position'][1]),
            'obj_z': float(state['position'][2]),
            'obj_lin_speed': state['linear_speed'],
            'obj_ang_speed': state['angular_speed'],
            'tilt_deg': state['tilt_deg'],
            'finger1': widths[0],
            'finger2': widths[1],
            'contact_force_1': f1,
            'contact_force_2': f2,
            'contact_links': len(active),
        })
        if non_finger_contacts(self.sim, self.block):
            self.hand_contact_samples += 1
        if self.sim.steps - self._last_contact_sample >= self.contact_interval:
            self._last_contact_sample = self.sim.steps
            for c in contact_points(self.sim, self.block):
                if c['normal_force'] <= 0.0:
                    continue
                self.contacts.append({
                    't': self.now(), 'stage': stage, 'link': c['link'],
                    'normal_force': c['normal_force'],
                    'px': float(c['position'][0]), 'py': float(c['position'][1]),
                    'pz': float(c['position'][2]),
                })

    def _sample_due(self, stage):
        if self.sim.steps % self.record_interval == 0:
            self._sample(stage)

    # ---------- 阶段原语 ----------
    def _step_stage(self, stage, seconds):
        steps = max(1, int(round(seconds / self.sim.dt)))
        start = self.sim.steps
        for _ in range(steps):
            self.sim.step()
            self._sample_due(stage)
            if self.on_step is not None:
                self.on_step(stage, self.sim.steps - start, steps)
        self._sample(stage)
        self.stages.append({
            'stage': stage, 'success': True, 'reason': 'ok',
            'seconds': (self.sim.steps - start) * self.sim.dt,
        })

    def _move(self, stage, target):
        start = self.sim.steps
        goal = np.asarray(target, dtype=float)

        def observer(k, n, q, seconds):
            self._sample_due(stage)
            if self.on_step is not None and k % 4 == 0:
                self.on_step(stage, k, n + 60)

        result = self.ctl.move_pose(goal, self.params['orientation'], observer=observer)
        self._sample(stage)
        self.stages.append({
            'stage': stage, 'success': bool(result.success), 'reason': result.reason,
            'seconds': (self.sim.steps - start) * self.sim.dt,
            'distance_rad': float(result.distance),
            'position_error_m': getattr(result, 'position_error_m', None),
        })
        return result

    def _fail(self, reason, stage):
        self.failure, self.failure_stage = reason, stage
        return False

    # ---------- 主流程 ----------
    def run(self):
        P = self.params
        set_gripper(self.sim, P['open_width'], P['open_force'], P['open_speed'])
        self._step_stage('open_gripper', 0.5)

        for stage, target in (('approach_pre', P['pre_grasp']), ('approach', P['grasp'])):
            if stage == 'approach':
                # 先闭合到略大于工件宽度，减少下降后闭合时的空行程冲击。
                set_gripper(self.sim, P['preclose_width'], P['grasp_force'], P['close_speed'])
                self._step_stage('preclose', P['preclose_seconds'])
            result = self._move(stage, target)
            if not result.success:
                return self._fail(result.reason, stage)

        set_gripper(self.sim, P['close_width'], P['grasp_force'], P['close_speed'])
        self._step_stage('close', P['close_seconds'])
        self.closed_widths = finger_widths(self.sim)
        f1, f2, active = finger_contact_forces(self.sim, self.block)
        self.grasp_contact_record = {
            'force_1': f1, 'force_2': f2, 'active_links': list(active),
            'widths': list(self.closed_widths),
        }
        if len(active) < 2:
            return self._fail('grasp_no_contact', 'close')
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', 'close')
        self.grasp_local = relative_pose(self.sim, self.block)
        grasped_z = object_state(self.sim, self.block)['position'][2]

        result = self._move('lift', P['lift'])
        if not result.success:
            return self._fail(result.reason, 'lift')
        lifted = object_state(self.sim, self.block)
        self.lift_height = float(lifted['position'][2] - grasped_z)
        local = relative_pose(self.sim, self.block)
        self.slip = float(np.linalg.norm(local[0] - self.grasp_local[0]))
        if self.lift_height < P['lift_height_target']:
            return self._fail('object_not_lifted', 'lift')
        if self.slip > P['slip_tolerance_m']:
            return self._fail('object_slipped', 'lift')
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', 'lift')

        for stage, target in (('transport', P['place_above']), ('lower', P['place'])):
            result = self._move(stage, target)
            if not result.success:
                return self._fail(result.reason, stage)
            local = relative_pose(self.sim, self.block)
            self.slip = max(self.slip, float(np.linalg.norm(local[0] - self.grasp_local[0])))
            if self.slip > P['slip_tolerance_m']:
                return self._fail('object_slipped', stage)
            if non_finger_contacts(self.sim, self.block):
                return self._fail('hand_contact', stage)

        set_gripper(self.sim, P['open_width'], P['open_force'], P['open_speed'])
        self._step_stage('release', P['release_seconds'])
        result = self._move('retreat', P['retreat'])
        if not result.success:
            return self._fail(result.reason, 'retreat')

        settle_start = self.sim.steps
        self._step_stage('settle', P['settle_seconds'])
        for s in self.samples:
            if s['t'] >= (settle_start - self.step0) * self.sim.dt:
                self.max_rest_speed = max(self.max_rest_speed, s['obj_lin_speed'])
        self._finalize()
        return self.success

    def _finalize(self):
        P = self.params
        final = object_state(self.sim, self.block)
        dx = float(final['position'][0] - P['place'][0])
        dy = float(final['position'][1] - P['place'][1])
        self.place_xy_error = float(np.hypot(dx, dy))
        self.place_height_error = float(final['position'][2] - P['block_half'][2])
        self.final_tilt = float(final['tilt_deg'])
        self.final_speed = float(final['linear_speed'])
        self.final_angular_speed = float(final['angular_speed'])
        self.release_contacts = len([c for c in contact_points(self.sim, self.block)
                                     if c['normal_force'] > 0.0])

        checks = [
            ('object_not_lifted', self.lift_height >= P['lift_height_target']),
            ('object_slipped', self.slip <= P['slip_tolerance_m']),
            ('hand_contact', self.hand_contact_samples == 0),
            ('place_position_error', self.place_xy_error <= P['place_tolerance_m']),
            ('place_height_error', abs(self.place_height_error) <= P['height_tolerance_m']),
            ('object_tipped', self.final_tilt <= P['tilt_tolerance_deg']),
            ('object_unstable', self.final_speed <= P['rest_linear_speed']
             and self.final_angular_speed <= P['rest_angular_speed']),
            ('not_released', self.release_contacts == 0),
        ]
        failed = [name for name, ok in checks if not ok]
        self.success = not failed
        self.failure = failed[0] if failed else None
        self.failure_stage = 'settle' if failed else None

    def result(self):
        P = self.params
        return {
            'success': bool(getattr(self, 'success', False)),
            'failure': self.failure,
            'failure_stage': self.failure_stage,
            'failure_text': FAIL_TEXT.get(self.failure, self.failure),
            'initial_object_position': self.initial['position'].tolist(),
            'final_object_position': object_state(self.sim, self.block)['position'].tolist(),
            'final_object_quaternion': object_state(self.sim, self.block)['quaternion'].tolist(),
            'lift_height_m': float(self.lift_height),
            'slip_m': float(self.slip),
            'place_xy_error_m': float(getattr(self, 'place_xy_error', float('nan'))),
            'place_height_error_m': float(getattr(self, 'place_height_error', float('nan'))),
            'final_tilt_deg': float(getattr(self, 'final_tilt', float('nan'))),
            'final_linear_speed': float(getattr(self, 'final_speed', float('nan'))),
            'final_angular_speed': float(getattr(self, 'final_angular_speed', float('nan'))),
            'release_contacts': int(getattr(self, 'release_contacts', -1)),
            'hand_contact_samples': int(self.hand_contact_samples),
            'closed_finger_widths': list(self.closed_widths),
            'grasp_contact': self.grasp_contact_record,
            'targets': {k: list(P[k]) for k in
                        ('pre_grasp', 'grasp', 'lift', 'place_above', 'place', 'retreat')},
            'gripper_widths': {'open': P['open_width'], 'preclose': P['preclose_width'],
                               'close': P['close_width']},
            'params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
            'sim_seconds': float(self.now()),
        }
