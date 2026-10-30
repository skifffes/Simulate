"""实验 2.9 B：状态感知专家 E1（不强制经过固定 0.07 m waypoint）。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from eval_handoff_timing import final_ok, obj_state  # noqa: E402
from manipulation import finger_widths, set_gripper  # noqa: E402
from run_b90_snapshot import SnapshotEpisode, LoggedExpert  # noqa: E402
from run_experiment_2_4 import SPA  # noqa: E402
from vla.atomic_tasks.experts import PUSH_Z, PUSH_WIDTH  # noqa: E402
from vla.atomic_tasks.scene2 import BLOCK_HALF, WALL_INNER_X, block_yaw  # noqa: E402


class StateAwareLoggedExpert(LoggedExpert):
    def skill_b(self):
        state = self._obj()
        pos = np.asarray(state['position'], float)
        yaw0 = self._yaw()
        depth = BLOCK_HALF[0] * abs(np.cos(yaw0)) + BLOCK_HALF[1] * abs(np.sin(yaw0))
        gap0 = float(WALL_INNER_X - (pos[0] + depth))
        self.metrics['front_gap_m'] = gap0
        self.metrics['e1_start_ee_z'] = float(
            p.getLinkState(self.sim.robot, 11, computeForwardKinematics=True,
                           physicsClientId=self.sim.client)[4][2])
        self._grip(PUSH_WIDTH)
        self._settle(0.3)
        ee = np.asarray(p.getLinkState(self.sim.robot, 11, computeForwardKinematics=True,
                                       physicsClientId=self.sim.client)[4], float)
        # Low start: approach horizontally at the current safe height, then lower.
        # High start: use the canonical 0.07 m approach height.
        if ee[2] <= 0.10:
            approach_z = max(PUSH_Z + 0.02, min(float(ee[2]), 0.08))
        else:
            approach_z = 0.07
        self.metrics['e1_approach_z'] = float(approach_z)
        if not self._move((pos[0] - 0.10, pos[1], approach_z), yaw=0.0, tag='b_approach').success:
            return self._result('B', False, 'b_approach')
        if approach_z > PUSH_Z + 0.001:
            if not self._move((pos[0] - 0.10, pos[1], PUSH_Z), yaw=0.0, tag='b_lower').success:
                return self._result('B', False, 'b_lower')
        finger_front = 0.0145
        rear0 = float(pos[0] - depth)
        contact_x = rear0 - finger_front - 0.010
        if not self._move((contact_x, pos[1], PUSH_Z), yaw=0.0, tag='b_contact').success:
            return self._result('B', False, 'b_contact')
        self._settle(0.3)
        block_target = float(WALL_INNER_X - depth - 0.004)
        push_x = float(block_target - depth - finger_front - 0.001)
        contact_x = float(contact_x)
        for i in range(1, 7):
            x = contact_x + (push_x - contact_x) * i / 6
            if not self._move((x, pos[1], PUSH_Z), yaw=0.0, tag=f'b_push_{i}').success:
                return self._result('B', False, f'b_push_{i}')
            self._settle(0.25)
        self._settle(0.5)
        if not self._move((push_x - 0.02, pos[1], 0.08), yaw=0.0, tag='b_retreat').success:
            return self._result('B', False, 'b_retreat')
        final_state = self._obj(); final = self._final()
        dy = float(final_state['position'][1] - pos[1])
        depth1 = BLOCK_HALF[0] * abs(np.cos(final['yaw_rad'])) + BLOCK_HALF[1] * abs(np.sin(final['yaw_rad']))
        gap1 = float(WALL_INNER_X - (final_state['position'][0] + depth1))
        self.metrics.update({'front_gap_end_m': gap1, 'pushed_x_m': float(final_state['position'][0] - pos[0]),
                             'lateral_dy_m': dy, 'yaw_change_rad': final['yaw_rad'] - yaw0})
        good = (abs(gap1) <= 0.02 and abs(dy) <= 0.02 and
                abs((final['yaw_rad'] - yaw0 + np.pi) % (2*np.pi) - np.pi) <= np.radians(15) and
                final['tilt_deg'] <= 12 and final['rest'])
        return self._result('B', good, 'ok' if good else 'push_check', extra={'gap_m': gap1, **final})


class E1Episode(SnapshotEpisode):
    def run(self):
        self.build()
        success, first = False, None
        try:
            success, first = self.run_expert_closed()
        finally:
            self.finish(True, success, first)
            data = {'condition': self.condition, 'source': self.snap.get('source'),
                    'gripper_init': self.gripper_init, 'source_meta': self.snap,
                    'expert': 'state_aware_e1'}
            for name in ('metadata.json', 'event_summary.json'):
                path = self.out / name
                if path.exists():
                    d = json.loads(path.read_text(encoding='utf-8')); d.update(data)
                    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
            if p.isConnected(self.sim.client):
                p.disconnect(self.sim.client)

    def run_expert_closed(self):
        self.skill = 'B'
        st0 = obj_state(self.sim, self.block); y0, yaw0 = float(st0['position'][1]), st0['yaw']
        exp = StateAwareLoggedExpert(self.sim, self.block, self.rig256, self, SPA)
        res = exp.skill_b()
        self.q_target = np.asarray(exp._cmd, float); self.grip_target = float(exp._grip_cmd)
        for _ in range(int(round(1.0 / (SPA / 240.0)))):
            self.sim.command(self.q_target); set_gripper(self.sim, self.grip_target, 20.0, 0.05)
            for _ in range(SPA):
                self.sim.step(); self.physics_step += 1; self.log_physics()
            self.policy_step += 1
        st = obj_state(self.sim, self.block); stable = st['speed'] < 0.01
        success = bool(res.get('success') and final_ok('B', st, True, stable, y0, yaw0))
        return success, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snap', type=Path, required=True)
    ap.add_argument('--condition', default='expert_e1')
    ap.add_argument('--policy', default='')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gripper-width', type=float, required=True)
    ap.add_argument('--checkpoint-sha', default='34a6e308')
    ap.add_argument('--commit', default='8440348')
    args = ap.parse_args()
    snap = json.loads(args.snap.read_text())
    ep = E1Episode(args, snap, args.condition)
    ep.run()


if __name__ == '__main__':
    main()
