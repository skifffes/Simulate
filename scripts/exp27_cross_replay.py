"""实验 2.7 实验 2：P14/P40 机械臂动作序列 × 14/40 mm 初始夹爪的交叉重放。

Open-loop Action Replay Diagnostic：只重放已记录的七关节目标；所有条件统一从 tick0
把夹爪目标设为 0.014 m，初始实际开度分别为 14/40 mm。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_handoff_timing import final_ok, obj_state, task_ok  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from run_experiment_2_4 import (  # noqa: E402
    CAMERAS, IMAGE_SIZE, PLACE, SPA, TASKS, VIDEO_H, VIDEO_W, Episode, make_view,
)
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402


class CrossEpisode(Episode):
    def __init__(self, args, snap, actions, init_width, label):
        super().__init__(args, 'G2_b_home_yaw0', snap['seed'], 0)
        self.snap = snap
        self.actions = actions
        self.init_width = init_width
        self.label = label
        self.out = Path(args.out) / label
        self.out.mkdir(parents=True, exist_ok=True)

    def build(self):
        self.sim, self.block, self.wall, self.tray, self.info = build_scene(
            seed=self.seed, start=PLACE, jitter=0.012, yaw=0.0)
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
        p.restoreState(fileName=str(ROOT / self.snap['bullet']), physicsClientId=self.sim.client)
        for j in (9, 10):
            p.resetJointState(self.sim.robot, j, float(self.init_width),
                              physicsClientId=self.sim.client)
        self.sim.command(self.sim.q())
        set_gripper(self.sim, float(self.init_width), 20.0, 0.05)
        for _ in range(10):
            self.sim.step()
            self.physics_step += 1
        st = obj_state(self.sim, self.block)
        self.initial_z = float(st['position'][2])
        self.initial_xy = np.asarray(st['position'][:2], dtype=float).copy()
        self.b_start_z = float(st['position'][2])
        self.gripper_init = {'requested_m': float(self.init_width),
                             'actual_m': float(np.mean(finger_widths(self.sim)))}

    def run(self):
        self.build()
        self.skill = 'B'
        st0 = obj_state(self.sim, self.block)
        y0, yaw0 = float(st0['position'][1]), st0['yaw']
        last_q = self.sim.q().copy()
        success = False
        first = None
        for i, a in enumerate(self.actions):
            self.capture('CROSS REPLAY')
            q = np.clip(np.asarray(a[:7], float), np.asarray(self.sim.lower) + 1e-4,
                        np.asarray(self.sim.upper) - 1e-4)
            last_q = q
            self.q_target = q
            self.sim.command(q)
            # unified gripper protocol: close to 0.014 from tick 0
            self.grip_target = 0.014
            set_gripper(self.sim, 0.014, 20.0, 0.05)
            self.policy_rows.append({'policy_step': self.policy_step,
                                     'physics_step': self.physics_step,
                                     'physics_time_sec': round(self.physics_step / 240.0, 5),
                                     'skill': 'B', 'source': 'open_loop_replay',
                                     'action': [round(float(v), 5) for v in a]})
            for _ in range(SPA):
                self.sim.step()
                self.physics_step += 1
                self.log_physics()
            self.policy_step += 1
            st = obj_state(self.sim, self.block)
            if i > 2 and task_ok('B', st, y0, yaw0) and first is None:
                first = i
        for _ in range(int(round(1.0 / (SPA / 240.0)))):
            self.sim.command(last_q)
            set_gripper(self.sim, 0.014, 20.0, 0.05)
            for _ in range(SPA):
                self.sim.step()
                self.physics_step += 1
                self.log_physics()
            self.policy_step += 1
        st = obj_state(self.sim, self.block)
        stable = st['speed'] < 0.01
        success = bool(final_ok('B', st, True, stable, y0, yaw0))
        self.finish(True, success, first)
        data = {'condition': self.label, 'source': str(self.snap.get('source')),
                'snapshot': self.snap['bullet'], 'init_width_m': self.init_width,
                'gripper_init': self.gripper_init, 'actions_source': str(self.actions_source)}
        data['success'] = success
        for name in ('metadata.json', 'event_summary.json'):
            path = self.out / name
            if path.exists():
                d = json.loads(path.read_text(encoding='utf-8'))
                d.update(data)
                path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
        if p.isConnected(self.sim.client):
            p.disconnect(self.sim.client)


def load_actions(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return [np.asarray(r['action'], float) for r in rows if r.get('skill') == 'B']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--p14-actions', type=Path, required=True)
    ap.add_argument('--p40-actions', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--policy', default='')
    ap.add_argument('--checkpoint-sha', default='34a6e308')
    ap.add_argument('--commit', default='ca743ef')
    ap.add_argument('--max-ticks-a', type=int, default=0)
    ap.add_argument('--max-ticks-b', type=int, default=260)
    args = ap.parse_args()
    snap = json.loads(args.snapshot.read_text())
    sources = {'P14': load_actions(args.p14_actions), 'P40': load_actions(args.p40_actions)}
    for src in ('P14', 'P40'):
        for width in (0.014, 0.040):
            label = f'{src}_g{int(width*1000):03d}'
            ep = CrossEpisode(args, snap, sources[src], width, label)
            ep.actions_source = args.p14_actions if src == 'P14' else args.p40_actions
            if (ep.out / 'summary.json').exists():
                print('skip', label, flush=True); continue
            print('run', label, len(sources[src]), flush=True)
            ep.run()


if __name__ == '__main__':
    main()
