"""从真实快照启动 B90：冻结 D2 或闭环 B 专家（C3 终态 / 方式 B 扩展）。"""
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
from run_experiment_2_6 import MidstateEpisode  # noqa: E402
from run_experiment_2_4 import SPA  # noqa: E402
from vla.atomic_tasks.recording import RecordingExpert  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402


class LoggedExpert(RecordingExpert):
    def __init__(self, sim, block, camera, episode, steps_per_action=12):
        super().__init__(sim, block, camera, steps_per_action)
        self.episode = episode

    def _hold(self, ticks: int = 1):
        for _ in range(ticks):
            self.episode.skill = 'B'
            self.episode.capture('EXPERT')
            self._record()
            act = np.concatenate([np.asarray(self._cmd, float), [float(self._grip_cmd)]])
            self.episode.policy_rows.append({
                'policy_step': self.episode.policy_step,
                'physics_step': self.episode.physics_step,
                'physics_time_sec': round(self.episode.physics_step / 240.0, 5),
                'skill': 'B', 'source': 'expert_closed_loop',
                'action': [round(float(v), 5) for v in act]})
            for _ in range(self.spa):
                self.sim.command(self._cmd)
                set_gripper(self.sim, self._grip_cmd, 20.0, 0.05)
                self.sim.step()
                self.episode.physics_step += 1
                self.episode.log_physics()
            self.episode.policy_step += 1
        return None


class SnapshotEpisode(MidstateEpisode):
    def __init__(self, args, snap, condition):
        row = {'episode': snap.get('episode', 0), 'frame': snap.get('frame', 0),
               'seed': snap['seed'], 'bullet': snap['bullet'],
               'group': snap.get('group', 'yaw90'), 'mode': snap.get('mode', 'c3')}
        super().__init__(args, row, condition)
        self.snap = snap
        tag = snap.get('tag', f"seed_{snap['seed']}")
        if getattr(args, 'gripper_width', None) is not None:
            tag += f'_g{int(round(args.gripper_width * 1000)):03d}'
        self.out = Path(args.out) / condition / tag
        self.out.mkdir(parents=True, exist_ok=True)
        self.gripper_init = None

    def run_expert_closed(self):
        self.skill = 'B'
        st0 = obj_state(self.sim, self.block)
        y0, yaw0 = float(st0['position'][1]), st0['yaw']
        exp = LoggedExpert(self.sim, self.block, self.rig256, self, SPA)
        res = exp.skill_b()
        self.q_target = np.asarray(exp._cmd, float)
        self.grip_target = float(exp._grip_cmd)
        for _ in range(int(round(1.0 / (SPA / 240.0)))):
            self.sim.command(self.q_target)
            set_gripper(self.sim, self.grip_target, 20.0, 0.05)
            for _ in range(SPA):
                self.sim.step()
                self.physics_step += 1
                self.log_physics()
            self.policy_step += 1
        st = obj_state(self.sim, self.block)
        stable = st['speed'] < 0.01
        success = bool(res.get('success') and final_ok('B', st, True, stable, y0, yaw0))
        return success, None

    def run(self):
        self.build()
        success, first = (False, None)
        try:
            if self.condition == 'expert':
                success, first = self.run_expert_closed()
            else:
                success, first = self.run_b()
        finally:
            self.finish(True, success, first)
            data = {'condition': self.condition, 'source': self.snap.get('source'),
                    'gripper_init': self.gripper_init,
                    'source_meta': self.snap}
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
    ap.add_argument('--snap', type=Path, required=True)
    ap.add_argument('--condition', choices=['expert', 'frozen'], required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gripper-width', type=float, default=None)
    ap.add_argument('--checkpoint-sha', default='34a6e308')
    ap.add_argument('--commit', default='a0d6377')
    ap.add_argument('--max-ticks-b', type=int, default=260)
    args = ap.parse_args()
    snap = json.loads(args.snap.read_text())
    ep = SnapshotEpisode(args, snap, args.condition)
    if (ep.out / 'summary.json').exists():
        print('skip', ep.out, flush=True)
        return
    ep.run()


if __name__ == '__main__':
    main()
