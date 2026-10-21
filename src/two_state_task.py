"""实验 0D 阶段二 A：把同一块工件放到交接区的两种确定性终态。

z1：长边沿世界 x；z2：长边沿世界 y。两次执行都从完全相同的初态开始，
只由夹爪接触搬运：z2 在抬升后由腕部绕竖直轴分 15° 步进转动 90°，再放下，
不使用 resetBasePositionAndOrientation，也不给工件任何固定约束。
"""
import numpy as np
import pybullet as p

from manipulation import (
    FINGER_MAX_FORCE, contact_points, finger_contact_forces, finger_widths,
    non_finger_contacts, object_state, relative_pose, set_gripper,
)

HALF = (0.05, 0.02, 0.02)          # 10 x 4 x 4 cm，长边超过夹爪最大开口 8 cm
HANDOVER = (0.45, 0.18)

DEFAULT_PARAMS = {
    'block_half': HALF,
    'block_mass': 0.08,
    'block_yaw_deg': 0.0,
    'grasp_yaw_deg': None,      # None 表示按工件当前朝向抓；复位时显式给工件当前朝向
    'block_start': (0.55, 0.0, 0.021),
    'block_friction': 1.2,
    'pre_grasp': (0.55, 0.0, 0.16),
    'grasp': (0.55, 0.0, 0.03),
    'lift': (0.55, 0.0, 0.13),
    'place_above': (HANDOVER[0], HANDOVER[1], 0.13),
    'place': (HANDOVER[0], HANDOVER[1], 0.03),
    'retreat': (HANDOVER[0], HANDOVER[1], 0.22),
    # 开口必须明显大于工件宽度（4 cm），否则下降过程中手指会先撞到工件侧面，
    # 把工件挤偏后再闭合，造成“抓上了但整体滑移 2 cm”的假成功。
    'open_width': 0.07,
    'preclose_width': 0.06,
    'close_width': 0.017,
    'grasp_force': FINGER_MAX_FORCE,
    'open_force': FINGER_MAX_FORCE,
    'open_speed': 0.06,
    'close_speed': 0.05,
    'preclose_seconds': 0.6,
    'close_seconds': 0.8,
    'release_seconds': 0.8,
    'settle_seconds': 1.2,
    'rotate_step_deg': 10.0,
    'lift_height_target': 0.08,
    'slip_tolerance_m': 0.02,
    'place_tolerance_m': 0.02,
    'height_tolerance_m': 0.012,
    'tilt_tolerance_deg': 12.0,
    'yaw_tolerance_deg': 12.0,
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
    'object_slipped': '运输/转动中工件相对夹爪滑移超限',
    'place_position_error': '释放后工件偏离交接点',
    'place_height_error': '释放后工件高度异常',
    'object_tipped': '释放后工件倾角过大',
    'object_yaw_error': '释放后工件朝向偏离目标终态',
    'object_unstable': '静置后工件未停下',
    'not_released': '释放后夹爪仍与工件接触',
}


def orientation_for(yaw_rad):
    return tuple(p.getQuaternionFromEuler([np.pi, 0.0, float(yaw_rad)]))


def object_yaw_deg(sim, body):
    _, quat = p.getBasePositionAndOrientation(body, physicsClientId=sim.client)
    m = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def yaw_error_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(abs(d))


class HandoverTask:
    """A 任务：抓取->抬升->（可选转动）->放到交接区->释放->静置。"""

    def __init__(self, sim, controller, block, target_yaw_deg, params=None,
                 on_step=None, record_interval=4, contact_interval=8):
        self.sim, self.ctl, self.block = sim, controller, block
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.target_yaw = float(target_yaw_deg)
        self.on_step = on_step
        self.record_interval = max(1, int(record_interval))
        self.contact_interval = max(1, int(contact_interval))
        self.samples, self.contacts, self.stages = [], [], []
        self.step0 = sim.steps
        self.initial = object_state(sim, block)
        self.initial_yaw = object_yaw_deg(sim, block)
        self.grasp_local = None
        self.grasp_contact_record = {}
        self.lift_height = 0.0
        self.slip = 0.0
        self.hand_contact_samples = 0
        self.failure = self.failure_stage = None
        self.success = False
        self.closed_widths = (None, None)
        self._last_contact_sample = -10 ** 9

    # ---------- 记录 ----------
    def now(self):
        return (self.sim.steps - self.step0) * self.sim.dt

    def _sample(self, stage):
        state = object_state(self.sim, self.block)
        f1, f2, active = finger_contact_forces(self.sim, self.block)
        widths = finger_widths(self.sim)
        ee = self.sim.pose()[0]
        self.samples.append({
            't': self.now(), 'stage': stage,
            'ee_x': float(ee[0]), 'ee_y': float(ee[1]), 'ee_z': float(ee[2]),
            'obj_x': float(state['position'][0]), 'obj_y': float(state['position'][1]),
            'obj_z': float(state['position'][2]),
            'obj_yaw_deg': object_yaw_deg(self.sim, self.block),
            'obj_lin_speed': state['linear_speed'],
            'obj_ang_speed': state['angular_speed'],
            'tilt_deg': state['tilt_deg'],
            'finger1': widths[0], 'finger2': widths[1],
            'contact_force_1': f1, 'contact_force_2': f2,
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

    # ---------- 原语 ----------
    def _step_stage(self, stage, seconds):
        steps = max(1, int(round(seconds / self.sim.dt)))
        start = self.sim.steps
        for _ in range(steps):
            self.sim.step()
            self._sample_due(stage)
            if self.on_step is not None:
                self.on_step(stage, self.sim.steps - start, steps)
        self._sample(stage)
        self.stages.append({'stage': stage, 'success': True, 'reason': 'ok',
                            'seconds': (self.sim.steps - start) * self.sim.dt})

    def _move(self, stage, target, yaw_deg=0.0, attempts=3):
        """一次 IK 数值解可能落在局部极值（末端偏几厘米），重解一次即可收敛。

        真实控制器同样会做迭代修正；这里保留重试次数与每次的末端误差记录。
        """
        start = self.sim.steps
        goal = np.asarray(target, dtype=float)
        result = None
        for attempt in range(attempts):
            def observer(k, n, q, seconds):
                self._sample_due(stage)
                if self.on_step is not None and k % 4 == 0:
                    self.on_step(stage, k, n + 60)

            result = self.ctl.move_pose(goal, orientation_for(np.radians(yaw_deg)),
                                        observer=observer)
            self._sample(stage)
            self.stages.append({
                'stage': stage, 'attempt': attempt,
                'success': bool(result.success), 'reason': result.reason,
                'seconds': (self.sim.steps - start) * self.sim.dt,
                'distance_rad': float(result.distance),
                'position_error_m': getattr(result, 'position_error_m', None),
                'orientation_error_rad': getattr(result, 'orientation_error_rad', None),
            })
            if result.success:
                return result
            if result.reason not in ('cartesian_pose_error', 'tracking_error'):
                return result
            if self.sim.collisions():
                return result
            if self._held_violation():
                return result
        return result

    def _held_violation(self):
        if self.grasp_local is None:
            return False
        if non_finger_contacts(self.sim, self.block):
            return True
        local = relative_pose(self.sim, self.block)
        return float(np.linalg.norm(local[0] - self.grasp_local[0])) > \
            self.params['slip_tolerance_m']

    def _fail(self, reason, stage):
        self.failure, self.failure_stage = reason, stage
        self.success = False
        return False

    def _check_held(self, stage):
        local = relative_pose(self.sim, self.block)
        self.slip = max(self.slip, float(np.linalg.norm(local[0] - self.grasp_local[0])))
        if self.slip > self.params['slip_tolerance_m']:
            return self._fail('object_slipped', stage)
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', stage)
        return True

    # ---------- 主流程 ----------
    def _presets(self):
        """抓取点按工件当前位姿推导，因此同一个任务也能用于重新抓取（复位）。"""
        pos = self.initial['position']
        self.grasp_pose = (float(pos[0]), float(pos[1]), float(pos[2]) + 0.009)
        self.pre_grasp_pose = (float(pos[0]), float(pos[1]), float(pos[2]) + 0.14)
        self.lift_pose = (float(pos[0]), float(pos[1]), float(pos[2]) + 0.11)
        # 抓取方向默认取工件当前朝向：z2 因此需要抬升后由腕部真正转 90°，
        # 而不是一开始就把夹爪转到终态朝向（那样转角来自手指挤压，不是受控转动）。
        self.grasp_yaw = self.initial_yaw if self.params['grasp_yaw_deg'] is None \
            else float(self.params['grasp_yaw_deg'])

    def run(self):
        P = self.params
        self._presets()
        set_gripper(self.sim, P['open_width'], P['open_force'], P['open_speed'])
        self._step_stage('open_gripper', 0.5)

        for stage, target in (('approach_pre', self.pre_grasp_pose),
                              ('approach', self.grasp_pose)):
            if stage == 'approach':
                set_gripper(self.sim, P['preclose_width'], P['grasp_force'], P['close_speed'])
                self._step_stage('preclose', P['preclose_seconds'])
            if not self._move(stage, target, yaw_deg=self.grasp_yaw).success:
                return self._fail(self.stages[-1]['reason'], stage)

        set_gripper(self.sim, P['close_width'], P['grasp_force'], P['close_speed'])
        self._step_stage('close', P['close_seconds'])
        self.closed_widths = finger_widths(self.sim)
        f1, f2, active = finger_contact_forces(self.sim, self.block)
        self.grasp_contact_record = {'force_1': f1, 'force_2': f2,
                                     'active_links': list(active),
                                     'widths': list(self.closed_widths)}
        if len(active) < 2:
            return self._fail('grasp_no_contact', 'close')
        if non_finger_contacts(self.sim, self.block):
            return self._fail('hand_contact', 'close')
        self.grasp_local = relative_pose(self.sim, self.block)
        grasped_z = object_state(self.sim, self.block)['position'][2]

        if not self._move('lift', self.lift_pose, yaw_deg=self.grasp_yaw).success:
            return self._fail(self.stages[-1]['reason'], 'lift')
        lifted = object_state(self.sim, self.block)
        self.lift_height = float(lifted['position'][2] - grasped_z)
        if self.lift_height < P['lift_height_target']:
            return self._fail('object_not_lifted', 'lift')
        if not self._check_held('lift'):
            return False

        step = float(P['rotate_step_deg'])
        sign = 1.0 if self.target_yaw >= self.grasp_yaw else -1.0
        angle = self.grasp_yaw
        while abs(self.target_yaw - angle) > 1e-6:
            angle = angle + sign * min(step, abs(self.target_yaw - angle))
            if not self._move('rotate', self.lift_pose, yaw_deg=angle).success:
                return self._fail(self.stages[-1]['reason'], 'rotate')
            if not self._check_held('rotate'):
                return False

        for stage, target in (('transport', P['place_above']), ('lower', P['place'])):
            if not self._move(stage, target, yaw_deg=self.target_yaw).success:
                return self._fail(self.stages[-1]['reason'], stage)
            if not self._check_held(stage):
                return False

        set_gripper(self.sim, P['open_width'], P['open_force'], P['open_speed'])
        self._step_stage('release', P['release_seconds'])
        if not self._move('retreat', P['retreat'], yaw_deg=self.target_yaw).success:
            return self._fail(self.stages[-1]['reason'], 'retreat')

        settle_start = self.sim.steps
        self._step_stage('settle', P['settle_seconds'])
        self.rest_speed = 0.0
        for s in self.samples:
            if s['t'] >= (settle_start - self.step0) * self.sim.dt:
                self.rest_speed = max(self.rest_speed, s['obj_lin_speed'])
        self._finalize()
        return self.success

    def _finalize(self):
        P = self.params
        final = object_state(self.sim, self.block)
        self.final_position = final['position'].copy()
        self.final_quaternion = final['quaternion'].copy()
        self.final_tilt = float(final['tilt_deg'])
        self.final_speed = float(final['linear_speed'])
        self.final_angular_speed = float(final['angular_speed'])
        self.final_yaw = object_yaw_deg(self.sim, self.block)
        self.yaw_error = yaw_error_deg(self.final_yaw, self.target_yaw)
        self.place_xy_error = float(np.hypot(final['position'][0] - HANDOVER[0],
                                             final['position'][1] - HANDOVER[1]))
        self.place_height_error = float(final['position'][2] - HALF[2])
        self.release_contacts = len([c for c in contact_points(self.sim, self.block)
                                     if c['normal_force'] > 0.0])
        checks = [
            ('object_not_lifted', self.lift_height >= P['lift_height_target']),
            ('object_slipped', self.slip <= P['slip_tolerance_m']),
            ('hand_contact', self.hand_contact_samples == 0),
            ('place_position_error', self.place_xy_error <= P['place_tolerance_m']),
            ('place_height_error', abs(self.place_height_error) <= P['height_tolerance_m']),
            ('object_tipped', self.final_tilt <= P['tilt_tolerance_deg']),
            ('object_yaw_error', self.yaw_error <= P['yaw_tolerance_deg']),
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
        final = object_state(self.sim, self.block)
        return {
            'success': bool(self.success),
            'failure': self.failure,
            'failure_stage': self.failure_stage,
            'failure_text': FAIL_TEXT.get(self.failure, self.failure),
            'target_yaw_deg': self.target_yaw,
            'initial_object_position': self.initial['position'].tolist(),
            'final_object_position': final['position'].tolist(),
            'final_object_quaternion': final['quaternion'].tolist(),
            'final_object_yaw_deg': float(getattr(self, 'final_yaw', float('nan'))),
            'yaw_error_deg': float(getattr(self, 'yaw_error', float('nan'))),
            'lift_height_m': float(self.lift_height),
            'slip_m': float(self.slip),
            'place_xy_error_m': float(getattr(self, 'place_xy_error', float('nan'))),
            'place_height_error_m': float(getattr(self, 'place_height_error', float('nan'))),
            'final_tilt_deg': float(getattr(self, 'final_tilt', float('nan'))),
            'final_linear_speed': float(getattr(self, 'final_speed', float('nan'))),
            'release_contacts': int(getattr(self, 'release_contacts', -1)),
            'hand_contact_samples': int(self.hand_contact_samples),
            'closed_finger_widths': list(self.closed_widths),
            'grasp_contact': self.grasp_contact_record,
            'sim_seconds': float(self.now()),
            'stage_trace': self.stages,
        }
