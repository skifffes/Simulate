"""实验 2.5：G1 接触前置条件（C1/C2/C3/C4）与 G5 clearance-first（R1）干预。

基于实验 2.4 的 Episode（同一记录/视频系统），每组用相同的 A policy 先完成前序技能，
再执行对应连接干预，最后交给同一个冻结 B VLA。

用法：
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src venv/bin/python \
    scripts/run_experiment_2_5.py --condition C1 --episodes 3 --seed-base 40000000 \
    --policy outputs/smolvla_e23_D2/checkpoints/020000/pretrained_model \
    --out results/experiment_2_5
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

from run_experiment_2_4 import Episode  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from robot_controller import RobotController  # noqa: E402
from vla.atomic_tasks.scene2 import grasp_quat  # noqa: E402


class Episode25(Episode):
    condition = 'C0'

    def __init__(self, args, group, seed, index):
        super().__init__(args, group, seed, index)
        base = Path(args.out) / ('G1' if args.condition.startswith('C') else 'G5') / args.condition
        self.out = base / f'seed_{seed}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.condition = args.condition
        self.gate_until = 0
        self.gate_min = 0.035
        self.connector = {'phases': []}

    # 带夹爪 gate 的策略 tick
    def policy_tick(self, skill, task, max_lift_check=None):
        frames = self.capture()
        state = np.concatenate([self.sim.q(), np.asarray(finger_widths(self.sim))])
        import time as _t
        t0 = _t.perf_counter()
        action = self.runner.act(frames, state, task)
        latency = (_t.perf_counter() - t0) * 1000
        target = np.clip(action[:7], np.asarray(self.sim.lower) + 1e-4,
                         np.asarray(self.sim.upper) - 1e-4)
        original = float(np.clip(action[7], 0.0, 0.04))
        gate = self.physics_step < self.gate_until
        width = max(original, self.gate_min) if gate else original
        self.q_target = target
        self.grip_target = width
        self.sim.command(target)
        set_gripper(self.sim, width, 20.0, 0.05)
        row = {
            'policy_step': self.policy_step, 'physics_step': self.physics_step,
            'physics_time_sec': round(self.physics_step / 240.0, 5), 'skill': skill,
            'task': task, 'state': [round(float(v), 5) for v in state],
            'action': [round(float(v), 5) for v in action],
            'original_gripper_target': round(original, 5),
            'executed_gripper_target': round(width, 5),
            'gripper_gate_active': int(gate),
            'latency_ms': round(latency, 3), 'frame_index': self.frame_idx,
        }
        self.policy_rows.append(row)
        for _ in range(12):
            self.sim.step()
            self.physics_step += 1
            self.log_physics()
        self.policy_step += 1

    def joint_path(self, q_goal, skill, tag):
        self.skill = skill
        q0 = self.sim.q().copy()
        delta = np.asarray(q_goal, dtype=float) - q0
        n = max(1, int(math.ceil(float(np.linalg.norm(delta)) / 0.002)))
        done = 0
        for k in range(1, n + 1):
            self.q_target = q0 + delta * (k / n)
            self.sim.command(self.q_target)
            set_gripper(self.sim, 0.04, 20.0, 0.06)
            self.sim.step()
            self.physics_step += 1
            self.log_physics()
            if k % 24 == 0:
                self.policy_step += 1
                self.capture(f'{tag} {k}/{n}')
            done += 1
        self.connector['phases'].append({'tag': tag, 'ticks': done,
                                         'start_q': [float(v) for v in q0],
                                         'goal_q': [float(v) for v in q_goal]})

    def pose_move(self, target, quat, skill, tag, lift_first=None):
        ctl = RobotController(self.sim)
        q_try = np.asarray(self.sim.q(), dtype=float)
        q_goal, ok, _ = ctl.solve(np.asarray(target, dtype=float), np.asarray(quat, dtype=float),
                                  rest=q_try)
        if not ok:
            self.connector['phases'].append({'tag': tag, 'error': 'ik_infeasible',
                                             'target': list(map(float, target))})
            return False
        self.joint_path(q_goal, skill, tag)
        return True

    def safe_clearance(self):
        ee, quat = self.sim.pose()
        self.pose_move(np.asarray(ee) + [0, 0, 0.08], quat, 'connect', 'lift_clear')
        ee, quat = self.sim.pose()
        self.pose_move(np.asarray(ee) + [-0.06, 0, 0.0], quat, 'connect', 'back_clear')
        self.connector['mode'] = 'safe_clearance'
        self.connector['end_ee'] = [float(v) for v in self.sim.pose()[0]]

    def prepush(self):
        st = p.getBasePositionAndOrientation(self.block, physicsClientId=self.sim.client)[0]
        target = np.array([st[0] - 0.10, st[1], 0.07])
        self.pose_move(target, grasp_quat(0.0), 'connect', 'prepush')
        self.connector['mode'] = 'prepush'
        self.connector['end_ee'] = [float(v) for v in self.sim.pose()[0]]

    def clearance_first(self):
        ee, quat = self.sim.pose()
        self.pose_move(np.asarray(ee) + [0, 0, 0.08], quat, 'connect', 'lift_clear')
        ee, quat = self.sim.pose()
        self.pose_move(np.asarray(ee) + [-0.02, 0, 0.0], quat, 'connect', 'short_back')
        self.connector['mode'] = 'clearance_first'
        self.connector['end_ee'] = [float(v) for v in self.sim.pose()[0]]

    def run(self):
        self.build()
        a_ok = False
        success = False
        first = None
        try:
            a_ok = self.run_a()
            if a_ok:
                if self.condition == 'C1':
                    self.gate_until = self.physics_step + int(0.5 * 240)
                    self.gate_min = 0.035
                elif self.condition == 'C2':
                    self.safe_clearance()
                elif self.condition == 'C3':
                    self.prepush()
                elif self.condition == 'C4':
                    self.retreat_home()
                elif self.condition == 'R1':
                    self.clearance_first()
                if self.condition in ('C1', 'C2', 'C3', 'C4', 'R1'):
                    success, first = self.run_b()
        finally:
            self.finish(a_ok, success, first)
            for name in ('metadata.json', 'event_summary.json'):
                pth = self.out / name
                if pth.exists():
                    data = json.loads(pth.read_text(encoding='utf-8'))
                    data['condition'] = self.condition
                    data['connector'] = self.connector
                    data['gate'] = {'until_physics_step': self.gate_until,
                                    'min_width': self.gate_min} if self.condition == 'C1' else None
                    pth.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            if p.isConnected(self.sim.client):
                p.disconnect(self.sim.client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=['C1', 'C2', 'C3', 'C4', 'R1'], required=True)
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seed-base', type=int, default=40000000)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--checkpoint-sha', default='34a6e308')
    ap.add_argument('--commit', default='ada1d3e')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_5')
    ap.add_argument('--max-ticks-a', type=int, default=260)
    ap.add_argument('--max-ticks-b', type=int, default=260)
    args = ap.parse_args()
    group = 'G1_a90b_direct' if args.condition.startswith('C') else 'G5_a0b_local_retreat'
    for i in range(args.episodes):
        seed = args.seed_base + i
        ep = Episode25(args, group, seed, i)
        if (ep.out / 'summary.json').exists():
            print('skip', ep.out, flush=True)
            continue
        ep.run()


if __name__ == '__main__':
    main()
