"""实验 2.6：D2 B 中间状态独立启动（A）与夹爪变体（B）。

对每个由 replay 重建的 Bullet 快照运行：
  expert  : 原始 D2 专家剩余动作（open-loop）
  frozen  : 冻结 D2 从同一快照重新启动
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_handoff_timing import final_ok, obj_state, task_ok  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from run_experiment_2_4 import (  # noqa: E402
    CAMERAS, IMAGE_SIZE, PLACE, SPA, TASKS, VIDEO_H, VIDEO_W, Episode,
    make_view, PolicyRunner,
)
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.cameras import CameraRig  # noqa: E402

SHARD = ROOT / 'data/_shards_e23/B_D2'


class MidstateEpisode(Episode):
    condition = 'expert'

    def __init__(self, args, row, condition):
        super().__init__(args, 'G2_b_home_yaw0', row['seed'], row['episode'])
        self.row = row
        self.condition = condition
        self.gripper_init = None
        tag = f'ep{row["episode"]:03d}_f{row["frame"]:03d}'
        if getattr(args, 'gripper_width', None) is not None:
            tag += f'_g{int(round(args.gripper_width * 1000)):03d}'
        self.out = Path(args.out) / condition / tag
        self.out.mkdir(parents=True, exist_ok=True)

    def build(self):
        # no PolicyRunner for expert-only episodes
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
        p.restoreState(fileName=str(ROOT / self.row['bullet']), physicsClientId=self.sim.client)
        q = self.sim.q()
        self.sim.command(q)
        set_gripper(self.sim, float(finger_widths(self.sim)[0]), 20.0, 0.05)
        for _ in range(2):
            self.sim.step()
            self.physics_step += 1
        st = obj_state(self.sim, self.block)
        self.initial_z = float(st['position'][2])
        self.initial_xy = np.asarray(st['position'][:2], dtype=float).copy()
        self.b_start_z = float(st['position'][2])
        gw = getattr(self.args, 'gripper_width', None)
        if gw is not None:
            before = obj_state(self.sim, self.block)
            for j in (9, 10):
                p.resetJointState(self.sim.robot, j, float(gw), physicsClientId=self.sim.client)
            self.sim.command(self.sim.q())
            set_gripper(self.sim, float(gw), 20.0, 0.05)
            for _ in range(10):
                self.sim.step()
                self.physics_step += 1
            after = obj_state(self.sim, self.block)
            self.gripper_init = {
                'requested_m': float(gw),
                'actual_m': float(np.mean(finger_widths(self.sim))),
                'obj_move_mm': float(np.linalg.norm(after['position'] - before['position']) * 1000),
                'obj_tilt_change_deg': float(after['tilt'] - before['tilt']),
            }
            self.initial_z = float(after['position'][2])
            self.initial_xy = np.asarray(after['position'][:2], dtype=float).copy()
            self.b_start_z = float(after['position'][2])
        self.runner = PolicyRunner(self.args.policy) if self.condition == 'frozen' else None

    def actions_for_episode(self):
        f = SHARD / f'data/chunk-000/episode_{self.row["episode"]:06d}.parquet'
        d = pq.read_table(f, columns=['action']).to_pydict()
        return np.asarray(d['action'], float)

    def run_expert(self):
        self.skill = 'B'
        st0 = obj_state(self.sim, self.block)
        y0, yaw0 = float(st0['position'][1]), st0['yaw']
        actions = self.actions_for_episode()[self.row['frame']:]
        success = False
        first = None
        last_q = self.sim.q().copy()
        last_w = float(finger_widths(self.sim)[0])
        for i, a in enumerate(actions):
            self.capture('EXPERT')
            q = np.clip(a[:7], np.asarray(self.sim.lower) + 1e-4,
                        np.asarray(self.sim.upper) - 1e-4)
            w = float(np.clip(a[7], 0.0, 0.04))
            last_q, last_w = q, w
            self.sim.command(q)
            set_gripper(self.sim, w, 20.0, 0.05)
            self.policy_rows.append({'policy_step': self.policy_step,
                                     'physics_step': self.physics_step,
                                     'physics_time_sec': round(self.physics_step / 240.0, 5),
                                     'skill': 'B', 'source': 'expert',
                                     'action': [round(float(v), 5) for v in a]})
            for _ in range(SPA):
                self.sim.step()
                self.physics_step += 1
                self.log_physics()
            self.policy_step += 1
            st = obj_state(self.sim, self.block)
            if i > 2 and task_ok('B', st, y0, yaw0):
                success = True
                first = i
                break
        for _ in range(int(round(1.0 / (SPA / 240.0)))):
            self.sim.command(last_q)
            set_gripper(self.sim, last_w, 20.0, 0.05)
            for _ in range(SPA):
                self.sim.step()
                self.physics_step += 1
                self.log_physics()
            self.policy_step += 1
        st = obj_state(self.sim, self.block)
        stable = st['speed'] < 0.01
        success = bool(success and final_ok('B', st, True, stable, y0, yaw0))
        return success, first

    def run(self):
        self.build()
        success, first = (False, None)
        try:
            if self.condition == 'expert':
                success, first = self.run_expert()
            else:
                success, first = self.run_b()
        finally:
            self.finish(True, success, first)
            data = {'condition': self.condition, 'episode': self.row['episode'],
                    'frame': self.row['frame'], 'group': self.row['group'],
                    'mode': self.row['mode'], 'seed': self.row['seed'],
                    'gripper_init': self.gripper_init}
            for name in ('metadata.json', 'event_summary.json'):
                path = self.out / name
                if path.exists():
                    d = json.loads(path.read_text(encoding='utf-8'))
                    d.update(data)
                    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
            if p.isConnected(self.sim.client):
                p.disconnect(self.sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', type=Path,
                    default=ROOT / 'results/experiment_2_6/A_midstate_launch/reconstructed/reconstruction_report.json')
    ap.add_argument('--states', default='all')
    ap.add_argument('--condition', choices=['expert', 'frozen'], required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--checkpoint-sha', default='34a6e308')
    ap.add_argument('--commit', default='03f227d')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_6/A_midstate_launch')
    ap.add_argument('--max-ticks-a', type=int, default=0)
    ap.add_argument('--max-ticks-b', type=int, default=260)
    ap.add_argument('--gripper-width', type=float, default=None)
    args = ap.parse_args()
    rows = json.loads(args.report.read_text())
    if args.states != 'all':
        keys = set(args.states.split(','))
        rows = [r for r in rows if f"ep{r['episode']:03d}_f{r['frame']:03d}" in keys]
    for r in rows:
        ep = MidstateEpisode(args, r, args.condition)
        if (ep.out / 'summary.json').exists():
            print('skip', ep.out, flush=True)
            continue
        ep.run()


if __name__ == '__main__':
    main()
