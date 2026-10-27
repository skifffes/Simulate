"""实验 2.4：A→B 接触模式失效归因与多视角诊断。

每个 episode 记录：
  - 统一全局时间索引：physics_step / physics_time / policy_step / video_frame / skill / switch event
  - 240 Hz 物理状态（关节、末端、物体、接触力）CSV
  - 每次策略调用的动作 JSONL
  - 逐物理步接触点 CSV（body/link/名称/力/位置）
  - 逐控制周期的 B 成功判据时间线
  - top / side / contact 三视角 mp4、comparison.mp4、events.mp4
  - event_summary.json / frame_index.csv / metadata.json

组别：
  G0 A0→B 直接切换        G1 A90→B 直接切换
  G2 B 独立 HOME yaw0      G3 B 独立 HOME yaw90
  G4 A0→B 连续退回 HOME    G5 A0→B 局部短退让（预先固定）

用法（服务器）：
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src venv/bin/python \
    scripts/run_experiment_2_4.py --group G0 --episodes 10 --seed-base 40000000 \
    --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
    --out results/experiment_2_4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_handoff_timing import a_success, finger_contacts, obj_state, task_ok  # noqa: E402
from eval_multi_atomic_vla import PolicyRunner  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from robot_controller import RobotController  # noqa: E402
from simulation import HOME  # noqa: E402
from vla.atomic_tasks.config2 import CAMERAS, IMAGE_SIZE, STEPS_PER_ACTION, TASKS  # noqa: E402
from vla.atomic_tasks.scene2 import (  # noqa: E402
    BLOCK_HALF, PLACE, WALL_INNER_X, block_yaw, build_scene,
)
from vla.cameras import CameraRig  # noqa: E402
from vla.lerobot_writer import encode_video, probe_video  # noqa: E402

VIDEO_W, VIDEO_H = 640, 480
FPS = 20
SPA = STEPS_PER_ACTION
BASE_HOME = np.asarray(HOME, dtype=float)

GROUPS = {
    'G0_a0b_direct': {'kind': 'a2b', 'a': 'A0', 'retreat': None},
    'G1_a90b_direct': {'kind': 'a2b', 'a': 'A90', 'retreat': None},
    'G2_b_home_yaw0': {'kind': 'b', 'yaw': 0.0},
    'G3_b_home_yaw90': {'kind': 'b', 'yaw': math.pi / 2},
    'G4_a0b_retreat_home': {'kind': 'a2b', 'a': 'A0', 'retreat': 'home'},
    'G5_a0b_local_retreat': {'kind': 'a2b', 'a': 'A0', 'retreat': 'local'},
}


def _font(size=14):
    from PIL import ImageFont
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT = _font(14)
FONT_SMALL = _font(12)


def render(px, view, proj, w=VIDEO_W, h=VIDEO_H):
    img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                           flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=px)[2]
    return np.asarray(img, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]


def make_view(eye, target, up=(0, 0, 1), fov=48.0, w=VIDEO_W, h=VIDEO_H):
    view = p.computeViewMatrix(list(eye), list(target), list(up))
    proj = p.computeProjectionMatrixFOV(fov, w / h, 0.02, 5.0)
    return view, proj


def link_name(body, link, names, client=0):
    if link < 0:
        return f'{names.get(body, body)}_base'
    if body == names.get('robot'):
        try:
            return p.getJointInfo(body, link, physicsClientId=client)[12].decode()
        except Exception:
            return f'link{link}'
    return f'{names.get(body, body)}_link{link}'


class Episode:
    def __init__(self, args, group, seed, ep_index):
        self.args = args
        self.cfg = GROUPS[group]
        self.group = group
        self.seed = seed
        self.ep_index = ep_index
        self.out = Path(args.out) / group / f'seed_{seed}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.physics_step = 0
        self.policy_step = 0
        self.frame_index = 0
        self.skill = 'init'
        self.q_target = BASE_HOME.copy()
        self.grip_target = 0.035
        self.phys_rows = []
        self.contact_rows = []
        self.policy_rows = []
        self.timeline = []
        self.frames = {'top': [], 'side': [], 'contact': []}
        self.frame_map = []
        self.last_cs = {'left': 0, 'right': 0, 'wall': 0, 'table': 1}
        self.events = {}
        self.max_lift = 0.0
        self.initial_z = None
        self.b_start_z = None
        self.initial_xy = None
        self.switch = None
        self.runner = None
        self.rig256 = None
        self.rig640 = None
        self.side_view = None
        self.side_proj = None
        self.names = {}
        self.started = time.time()

    # ---------------- setup ----------------
    def build(self):
        cfg = self.cfg
        if cfg['kind'] == 'a2b':
            self.sim, self.block, self.wall, self.tray, self.info = build_scene(
                seed=self.seed, jitter=0.008, yaw=0.0)
        else:
            self.sim, self.block, self.wall, self.tray, self.info = build_scene(
                seed=self.seed, start=PLACE, jitter=0.012, yaw=cfg['yaw'])
        self.names = {self.sim.plane: 'table', self.sim.robot: 'robot',
                      self.block: 'block', self.wall: 'wall'}
        for t in self.tray:
            self.names[t] = 'tray'
        self.rig256 = CameraRig(self.sim, CAMERAS, IMAGE_SIZE, supersample=1)
        self.rig640 = CameraRig(self.sim, CAMERAS, (VIDEO_W, VIDEO_H), supersample=1)
        self.side_view, self.side_proj = make_view((0.55, -0.62, 0.30),
                                                   (0.60, 0.16, 0.10), fov=46.0)
        for _ in range(30):
            self.sim.step()
            self.physics_step += 1
        state = obj_state(self.sim, self.block)
        self.initial_z = float(state['position'][2])
        self.initial_xy = np.asarray(state['position'][:2], dtype=float).copy()
        self.runner = PolicyRunner(self.args.policy)

    # ---------------- logging ----------------
    def contact_rows_for_step(self):
        sim = self.sim
        rows = []
        block_contacts = p.getContactPoints(bodyA=self.block, physicsClientId=sim.client)
        for c in block_contacts:
            if c[8] > 0.0 and c[9] <= 0.0:
                continue
            b1, b2, l1, l2 = c[1], c[2], c[3], c[4]
            n1 = link_name(b1, l1, self.names, sim.client)
            n2 = link_name(b2, l2, self.names, sim.client)
            rows.append({
                'physics_step': self.physics_step, 'physics_time_sec': round(self.physics_step / 240.0, 5),
                'policy_step': self.policy_step, 'skill': self.skill,
                'bodyA': b1, 'bodyB': b2, 'linkA': l1, 'linkB': l2,
                'nameA': n1, 'nameB': n2,
                'point_x': round(c[5][0], 5), 'point_y': round(c[5][1], 5),
                'point_z': round(c[5][2], 5),
                'normal_x': round(c[7][0], 5), 'normal_y': round(c[7][1], 5),
                'normal_z': round(c[7][2], 5),
                'normal_force': round(c[9], 4), 'distance': round(c[8], 6),
                'lateral_friction1': round(float(c[12]), 4) if len(c) > 12 and isinstance(c[12], (int, float)) else 0.0,
                'lateral_friction2': round(float(c[13]), 4) if len(c) > 13 and isinstance(c[13], (int, float)) else 0.0,
            })
        return rows

    def summarize_contacts(self, rows):
        s = {'left': 0, 'right': 0, 'palm': 0, 'arm': 0, 'table': 0, 'wall': 0,
             'left_force': 0.0, 'right_force': 0.0, 'wall_force': 0.0, 'total_force': 0.0}
        robot = self.sim.robot
        plane = self.sim.plane
        for r in rows:
            bodies = (r['bodyA'], r['bodyB'])
            if robot in bodies:
                other = r['bodyB'] if r['bodyA'] == robot else r['bodyA']
                robot_link = r['linkA'] if r['bodyA'] == robot else r['linkB']
                if other == self.block:
                    if robot_link == 9:
                        s['left'] = 1
                        s['left_force'] += r['normal_force']
                    elif robot_link == 10:
                        s['right'] = 1
                        s['right_force'] += r['normal_force']
                    elif robot_link == 8:
                        s['palm'] = 1
                    else:
                        s['arm'] = 1
            if plane in bodies:
                s['table'] = 1
            if self.wall in bodies:
                s['wall'] = 1
                s['wall_force'] += r['normal_force']
            s['total_force'] += r['normal_force']
        return s

    def log_physics(self):
        sim = self.sim
        states = p.getJointStates(sim.robot, list(range(7)), physicsClientId=sim.client)
        q = [float(x[0]) for x in states]
        dq = [float(x[1]) for x in states]
        fingers = p.getJointStates(sim.robot, [9, 10], physicsClientId=sim.client)
        grip = float(np.mean([f[0] for f in fingers]))
        ee = p.getLinkState(sim.robot, sim.ee, computeForwardKinematics=True,
                            computeLinkVelocity=True, physicsClientId=sim.client)
        pos, quat = p.getBasePositionAndOrientation(self.block, physicsClientId=sim.client)
        lin, ang = p.getBaseVelocity(self.block, physicsClientId=sim.client)
        euler = p.getEulerFromQuaternion(quat)
        tilt = abs(math.degrees(euler[0])) + abs(math.degrees(euler[1]))
        yaw = block_yaw(self.block, sim.client)
        depth = BLOCK_HALF[0] * abs(math.cos(yaw)) + BLOCK_HALF[1] * abs(math.sin(yaw))
        gap = float(WALL_INNER_X - (pos[0] + depth))
        lift = float(pos[2] - self.initial_z)
        b_lift = float(pos[2] - self.b_start_z) if self.b_start_z is not None else 0.0
        if self.skill == 'B':
            self.max_lift = max(self.max_lift, b_lift)
        rows = self.contact_rows_for_step()
        self.contact_rows.extend(rows)
        cs = self.summarize_contacts(rows)
        row = {
            'physics_step': self.physics_step, 'physics_time_sec': round(self.physics_step / 240.0, 5),
            'policy_step': self.policy_step, 'skill': self.skill,
            **{f'q{i + 1}': round(q[i], 5) for i in range(7)},
            **{f'dq{i + 1}': round(dq[i], 5) for i in range(7)},
            **{f'q_target{i + 1}': round(float(self.q_target[i]), 5) for i in range(7)},
            'grip_actual': round(grip, 5), 'grip_target': round(float(self.grip_target), 5),
            'ee_x': round(float(ee[4][0]), 5), 'ee_y': round(float(ee[4][1]), 5),
            'ee_z': round(float(ee[4][2]), 5),
            'ee_qx': round(float(ee[5][0]), 5), 'ee_qy': round(float(ee[5][1]), 5),
            'ee_qz': round(float(ee[5][2]), 5), 'ee_qw': round(float(ee[5][3]), 5),
            'ee_vx': round(float(ee[6][0]), 5), 'ee_vy': round(float(ee[6][1]), 5),
            'ee_vz': round(float(ee[6][2]), 5),
            'ee_wx': round(float(ee[7][0]), 5), 'ee_wy': round(float(ee[7][1]), 5),
            'ee_wz': round(float(ee[7][2]), 5),
            'obj_x': round(float(pos[0]), 5), 'obj_y': round(float(pos[1]), 5),
            'obj_z': round(float(pos[2]), 5),
            'obj_qx': round(float(quat[0]), 5), 'obj_qy': round(float(quat[1]), 5),
            'obj_qz': round(float(quat[2]), 5), 'obj_qw': round(float(quat[3]), 5),
            'obj_vx': round(float(lin[0]), 5), 'obj_vy': round(float(lin[1]), 5),
            'obj_vz': round(float(lin[2]), 5),
            'obj_wx': round(float(ang[0]), 5), 'obj_wy': round(float(ang[1]), 5),
            'obj_wz': round(float(ang[2]), 5),
            'obj_tilt_deg': round(tilt, 3), 'obj_yaw_rad': round(float(yaw), 5),
            'obj_wall_gap_m': round(gap, 5), 'obj_lift_m': round(lift, 5),
            'b_lift_m': round(b_lift, 5),
            'obj_dx_m': round(float(pos[0] - self.initial_xy[0]), 5),
            'obj_dy_m': round(float(pos[1] - self.initial_xy[1]), 5),
            'table_contact': cs['table'], 'wall_contact': cs['wall'],
            'left_finger_contact': cs['left'], 'right_finger_contact': cs['right'],
            'palm_contact': cs['palm'], 'arm_contact': cs['arm'],
            'left_force': round(cs['left_force'], 3), 'right_force': round(cs['right_force'], 3),
            'wall_force': round(cs['wall_force'], 3), 'total_force': round(cs['total_force'], 3),
        }
        self.phys_rows.append(row)
        self.last_cs = cs
        self._update_events(row)

    def _update_events(self, row):
        if self.skill != 'B':
            return
        t = row['physics_time_sec']

        def put(name, **extra):
            if name not in self.events:
                self.events[name] = {'physics_step': self.physics_step, 'physics_time_sec': t,
                                     'policy_step': self.policy_step,
                                     'frame_index': self.frame_index,
                                     'obj_x': row['obj_x'], 'obj_z': row['obj_z'],
                                     'ee_z': row['ee_z'], 'lift': row['b_lift_m'], **extra}

        if row['left_finger_contact'] or row['right_finger_contact'] \
                or row['palm_contact'] or row['arm_contact']:
            put('first_contact')
        if row['left_finger_contact'] and row['right_finger_contact']:
            put('first_both_fingers')
        if row['wall_contact']:
            put('first_wall_contact')
        if row['b_lift_m'] > 0.005 and not row['table_contact']:
            put('first_table_loss')
        if row['b_lift_m'] > 0.03:
            put('first_lift')
        if self.max_lift == row['b_lift_m']:
            self.events['max_lift'] = {'physics_step': self.physics_step, 'physics_time_sec': t,
                                       'policy_step': self.policy_step,
                                       'frame_index': self.frame_index,
                                       'lift': row['b_lift_m']}

    # ---------------- rendering ----------------
    def overlay(self, img, extra=''):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        st = obj_state(self.sim, self.block)
        ee = p.getLinkState(self.sim.robot, self.sim.ee, computeForwardKinematics=True,
                            physicsClientId=self.sim.client)[4]
        grip = float(np.mean(finger_widths(self.sim)))
        lift_now = st['position'][2] - (self.b_start_z if self.b_start_z is not None else self.initial_z)
        lines = [
            f'{self.group}  seed {self.seed}  ep {self.ep_index}',
            f't={self.physics_step / 240.0:6.2f}s  step {self.physics_step}  policy {self.policy_step}  skill {self.skill}',
            f'obj x={st["position"][0]:.3f} z={st["position"][2]:.3f}  ee z={ee[2]:.3f}  grip={grip:.3f}',
            f'L={self.last_cs["left"]} R={self.last_cs["right"]} wall={self.last_cs["wall"]} '
            f'table={self.last_cs["table"]} lift={lift_now:.3f} {extra}',
        ]
        y = 4
        for line in lines:
            d.rectangle([0, y - 1, 630, y + 16], fill=(0, 0, 0))
            d.text((4, y), line, font=FONT_SMALL, fill=(255, 255, 255))
            y += 17
        if 'first_lift' in self.events and abs(self.physics_step - self.events['first_lift']['physics_step']) < 120:
            d.text((10, 100), 'LIFT EVENT', font=FONT, fill=(255, 60, 60))
        if 'first_wall_contact' in self.events and abs(self.physics_step - self.events['first_wall_contact']['physics_step']) < 120:
            d.text((10, 125), 'WALL CONTACT', font=FONT, fill=(255, 200, 0))
        if self.switch and abs(self.physics_step - self.switch['physics_step']) < 120:
            d.text((10, 150), f'SKILL SWITCH: {self.switch["from"]} -> {self.skill}', font=FONT, fill=(0, 255, 255))
        return np.asarray(im)

    def capture(self, extra=''):
        pol = self.rig256.capture()
        top = render(self.sim.client, self.rig640._static['top'][0], self.rig640._static['top'][1])
        side = render(self.sim.client, self.side_view, self.side_proj)
        wv, wp = self.rig640._wrist_matrices(CAMERAS['wrist'])
        contact = render(self.sim.client, wv, wp)
        self.frames['top'].append(self.overlay(top, extra))
        self.frames['side'].append(self.overlay(side, extra))
        self.frames['contact'].append(self.overlay(contact, extra))
        self.frame_idx = self.frame_index
        self.frame_index += 1
        self.frame_map.append({
            'frame_index': self.frame_idx, 'physics_step': self.physics_step,
            'physics_time_sec': round(self.physics_step / 240.0, 5),
            'policy_step': self.policy_step, 'skill': self.skill,
        })
        return pol

    # ---------------- control ----------------
    def policy_tick(self, skill, task, max_lift_check=None):
        frames = self.capture()
        state = np.concatenate([self.sim.q(), np.asarray(finger_widths(self.sim))])
        t0 = time.perf_counter()
        action = self.runner.act(frames, state, task)
        latency = (time.perf_counter() - t0) * 1000
        target = np.clip(action[:7], np.asarray(self.sim.lower) + 1e-4,
                         np.asarray(self.sim.upper) - 1e-4)
        width = float(np.clip(action[7], 0.0, 0.04))
        self.q_target = target
        self.grip_target = width
        self.sim.command(target)
        set_gripper(self.sim, width, 20.0, 0.05)
        self.policy_rows.append({
            'policy_step': self.policy_step, 'physics_step': self.physics_step,
            'physics_time_sec': round(self.physics_step / 240.0, 5), 'skill': skill,
            'task': task, 'state': [round(float(v), 5) for v in state],
            'action': [round(float(v), 5) for v in action],
            'latency_ms': round(latency, 3),
            'frame_index': self.frame_idx,
        })
        for _ in range(SPA):
            self.sim.step()
            self.physics_step += 1
            self.log_physics()
        self.policy_step += 1
        return action

    def step_hold(self, ticks, skill):
        for _ in range(ticks):
            for _ in range(SPA):
                self.sim.command(self.q_target)
                set_gripper(self.sim, self.grip_target, 20.0, 0.05)
                self.sim.step()
                self.physics_step += 1
                self.log_physics()
            self.policy_step += 1

    # ---------------- phases ----------------
    def run_a(self):
        a = self.cfg['a']
        task = TASKS[a]
        self.runner.reset()
        np.random.seed(self.seed)
        self.runner.torch.manual_seed(self.seed)
        self.runner.torch.cuda.manual_seed_all(self.seed)
        self.skill = a
        pos0, _ = p.getBasePositionAndOrientation(self.block, physicsClientId=self.sim.client)
        initial_z = float(pos0[2])
        max_z = initial_z
        for tick in range(self.args.max_ticks_a):
            self.policy_tick(a, task)
            pos, _ = p.getBasePositionAndOrientation(self.block, physicsClientId=self.sim.client)
            max_z = max(max_z, float(pos[2]))
            if tick > 20 and a_success(a, self.sim, self.block, initial_z, max_z):
                self.switch = {'from': a, 'to': 'B', 'policy_step': self.policy_step,
                               'physics_step': self.physics_step,
                               'physics_time_sec': round(self.physics_step / 240.0, 5),
                               'frame_index': self.frame_index}
                return True
        return False

    def retreat_home(self):
        self.skill = 'retreat'
        q0 = self.sim.q().copy()
        delta = BASE_HOME - q0
        n = max(1, int(np.ceil(float(np.linalg.norm(delta)) / 0.002)))
        self.capture('RETREAT HOME')
        for k in range(1, n + 1):
            self.q_target = q0 + delta * (k / n)
            self.sim.command(self.q_target)
            set_gripper(self.sim, 0.04, 20.0, 0.06)
            self.sim.step()
            self.physics_step += 1
            self.log_physics()
            if k % 12 == 0:
                self.policy_step += 1
                self.capture('RETREAT HOME')

    def retreat_local(self):
        self.skill = 'retreat'
        ctl = RobotController(self.sim)
        ee, quat = self.sim.pose()
        target = np.asarray(ee, dtype=float) + np.array([-0.04, 0.0, 0.03])
        q_ik, ok, _ = ctl.solve(target, quat)
        mode = 'ik_offset'
        if not ok:
            target = np.asarray(ee, dtype=float) + np.array([0.0, 0.0, 0.04])
            q_ik, ok, _ = ctl.solve(target, quat)
            mode = 'lift_only'
        if not ok:
            return 'ik_failed'
        q0 = self.sim.q().copy()
        delta = np.asarray(q_ik, dtype=float) - q0
        n = max(1, int(np.ceil(float(np.linalg.norm(delta)) / 0.002)))
        self.capture(f'LOCAL RETREAT {mode}')
        for k in range(1, n + 1):
            self.q_target = q0 + delta * (k / n)
            self.sim.command(self.q_target)
            set_gripper(self.sim, 0.04, 20.0, 0.06)
            self.sim.step()
            self.physics_step += 1
            self.log_physics()
            if k % 12 == 0:
                self.policy_step += 1
                self.capture(f'LOCAL RETREAT {mode}')
        return mode

    def run_b(self):
        task = TASKS['B']
        self.skill = 'B'
        self.runner.reset()
        st = obj_state(self.sim, self.block)
        self.b_start_z = float(st['position'][2])
        self.max_lift = 0.0
        y0, yaw0 = float(st['position'][1]), st['yaw']
        success = False
        first_success = None
        for tick in range(self.args.max_ticks_b):
            self.policy_tick('B', task)
            st = obj_state(self.sim, self.block)
            conds = self.b_conditions(st, y0, yaw0)
            self.timeline.append({
                'policy_step': self.policy_step, 'physics_step': self.physics_step,
                'physics_time_sec': round(self.physics_step / 240.0, 5),
                'frame_index': self.frame_index - 1, 'skill': 'B',
                'success_pred': int(conds['predicate']),
                **{k: int(v) for k, v in conds.items() if isinstance(v, bool)},
                'gap_m': round(conds['gap_m'], 5), 'lateral_dy_m': round(conds['dy'], 5),
                'dyaw_deg': round(math.degrees(conds['dyaw']), 2),
                'speed': round(conds['speed'], 5), 'tilt_deg': round(conds['tilt'], 2),
                'obj_z': round(float(st['position'][2]), 5),
            })
            if conds['predicate'] and 'first_success_predicate' not in self.events:
                self.events['first_success_predicate'] = {
                    'policy_step': self.policy_step, 'physics_step': self.physics_step,
                    'physics_time_sec': round(self.physics_step / 240.0, 5),
                    'frame_index': self.frame_index - 1,
                    'gap_m': float(conds['gap_m']), 'lift': float(st['position'][2] - self.initial_z),
                }
            if tick > 15 and task_ok('B', st, y0, yaw0):
                success = True
                first_success = tick
                break
        # 1.0 s settle with last command (v2.2)
        self.step_hold(int(round(1.0 / (SPA / 240.0))), 'settle')
        st = obj_state(self.sim, self.block)
        stable = st['speed'] < 0.01
        conds = self.b_conditions(st, y0, yaw0)
        success = bool(success and conds['predicate'] and stable)
        return success, first_success

    def b_conditions(self, st, y0, yaw0):
        depth = BLOCK_HALF[0] * abs(math.cos(st['yaw'])) + BLOCK_HALF[1] * abs(math.sin(st['yaw']))
        gap = float(WALL_INNER_X - (st['position'][0] + depth))
        dy = float(st['position'][1] - y0)
        dyaw = abs(math.atan2(math.sin(st['yaw'] - yaw0), math.cos(st['yaw'] - yaw0)))
        c = {
            'gap_m': gap, 'dy': dy, 'dyaw': dyaw, 'speed': st['speed'], 'tilt': st['tilt'],
            'cond_gap': bool(0.0 <= gap <= 0.02),
            'cond_lateral': bool(abs(dy) <= 0.02),
            'cond_yaw': bool(dyaw <= math.radians(15)),
            'cond_speed': bool(st['speed'] < 0.01),
            'cond_tilt': bool(st['tilt'] <= 12),
        }
        c['predicate'] = bool(c['cond_gap'] and c['cond_lateral'] and c['cond_yaw']
                              and c['cond_speed'] and c['cond_tilt'])
        return c

    # ---------------- episode ----------------
    def run(self):
        self.build()
        try:
            if self.cfg['kind'] == 'b':
                success, first = self.run_b()
                a_ok = True
            else:
                a_ok = self.run_a()
                if not a_ok:
                    success, first = False, None
                else:
                    if self.cfg.get('retreat') == 'home':
                        self.retreat_home()
                    elif self.cfg.get('retreat') == 'local':
                        self.events['local_retreat_mode'] = self.retreat_local()
                    success, first = self.run_b()
        finally:
            self.finish(a_ok if 'a_ok' in locals() else False, success if 'success' in locals() else False,
                        first if 'first' in locals() else None)
            p.disconnect(self.sim.client)

    def finish(self, a_ok, success, first_success):
        out = self.out
        # CSV logs
        if self.phys_rows:
            with (out / 'physics_trace.csv').open('w', newline='', encoding='utf-8-sig') as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.phys_rows[0]))
                w.writeheader()
                w.writerows(self.phys_rows)
        if self.contact_rows:
            with (out / 'contact_trace.csv').open('w', newline='', encoding='utf-8-sig') as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.contact_rows[0]))
                w.writeheader()
                w.writerows(self.contact_rows)
        with (out / 'policy_trace.jsonl').open('w', encoding='utf-8') as fh:
            for row in self.policy_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        if self.timeline:
            with (out / 'success_timeline.csv').open('w', newline='', encoding='utf-8-sig') as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.timeline[0]))
                w.writeheader()
                w.writerows(self.timeline)
        # videos
        videos = {}
        for name, frames in self.frames.items():
            if not frames:
                continue
            arr = np.stack(frames)
            path = out / f'{name}.mp4'
            encode_video(arr, path, FPS)
            videos[name] = str(path)
            from PIL import Image
            for tag, idx in (('start', 0), ('mid', len(arr) // 2), ('end', len(arr) - 1)):
                Image.fromarray(arr[idx]).save(out / f'{name}_{tag}.png')
            if name == 'top':
                keys = list(self.frames)
                comp = np.concatenate([np.stack(self.frames[k]) for k in keys], axis=1)
                encode_video(comp, out / 'comparison.mp4', FPS)
        # events clip: side view around event frames
        side = self.frames.get('side')
        if side:
            segs = []
            for name in ('max_lift', 'first_wall_contact', 'first_lift', 'first_both_fingers'):
                ev = self.events.get(name)
                if not ev:
                    continue
                idx = min(max(0, ev['frame_index']), len(side) - 1)
                seg = side[max(0, idx - 40):min(len(side), idx + 40)]
                segs.extend(seg)
                segs.extend([np.full_like(seg[0], 255)] * 8)
            if segs:
                encode_video(np.stack(segs), out / 'events.mp4', FPS)
        # frame index
        with (out / 'frame_index.csv').open('w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=['frame_index', 'physics_step', 'physics_time_sec',
                                               'policy_step', 'skill'])
            w.writeheader()
            w.writerows(self.frame_map)
        # metadata / events / summary
        probe = {}
        for path in sorted(out.glob('*.mp4')):
            try:
                probe[path.name] = probe_video(path)
            except Exception as e:
                probe[path.name] = {'error': str(e)}
        event_summary = {
            'group': self.group, 'seed': self.seed, 'episode': self.ep_index,
            'a_success': bool(a_ok), 'success': bool(success),
            'first_success_tick': first_success,
            'events': self.events, 'max_lift_m': round(self.max_lift, 4),
            'switch': self.switch,
            'ticks_total': self.physics_step, 'wall_seconds': round(time.time() - self.started, 2),
            'videos': probe,
        }
        (out / 'event_summary.json').write_text(json.dumps(event_summary, ensure_ascii=False, indent=2),
                                                encoding='utf-8')
        metadata = {
            'group': self.group, 'seed': self.seed, 'episode': self.ep_index,
            'policy': self.args.policy, 'checkpoint_sha256': self.args.checkpoint_sha,
            'repo_commit': self.args.commit,
            'pybullet': p.getAPIVersion(), 'control_hz': 20, 'physics_hz': 240,
            'video_fps': FPS, 'video_resolution': [VIDEO_W, VIDEO_H],
            'policy_input_size': list(IMAGE_SIZE),
            'diagnostic_cameras_are_policy_input': False,
            'camera_views': {'top': 'fixed', 'side': 'diagnostic',
                             'contact': 'wrist (high-res diagnostic)'},
            'success_judge': 'v2.2 (1.0 s settle, gap<=2cm, |dy|<=2cm, dyaw<=15deg, tilt<=12deg)',
            'initial_z': self.initial_z,
            'initial_xy': self.initial_xy.tolist(),
            'local_retreat_distance_m': 0.05 if self.group == 'G5_a0b_local_retreat' else None,
        }
        (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
        (out / 'summary.json').write_text(json.dumps({
            'group': self.group, 'seed': self.seed, 'a_success': bool(a_ok),
            'success': bool(success), 'max_lift_m': self.max_lift,
            'wall_contact': bool(self.events.get('first_wall_contact')),
            'both_fingers': bool(self.events.get('first_both_fingers')),
            'table_loss': bool(self.events.get('first_table_loss')),
            'first_lift': self.events.get('first_lift'),
            'first_success_tick': first_success,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'group': self.group, 'seed': self.seed, 'success': success,
                          'a_success': a_ok, 'max_lift': round(self.max_lift, 3),
                          'wall': bool(self.events.get('first_wall_contact'))},
                         ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=list(GROUPS), required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--seed-base', type=int, default=40000000)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--checkpoint-sha', default='')
    ap.add_argument('--commit', default='fcf7665')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_4')
    ap.add_argument('--max-ticks-a', type=int, default=260)
    ap.add_argument('--max-ticks-b', type=int, default=260)
    args = ap.parse_args()
    for i in range(args.episodes):
        seed = args.seed_base + i
        ep = Episode(args, args.group, seed, i)
        if (ep.out / 'summary.json').exists():
            print('skip done', ep.out, flush=True)
            continue
        ep.run()


if __name__ == '__main__':
    main()
