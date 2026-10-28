"""Reconstruct D2 B mid-trajectory snapshots by replaying dataset actions.

For each selected (episode, frame), rebuild scene, set the first observation,
replay the recorded absolute joint/gripper actions, and save a Bullet snapshot.
Reports replay-vs-recorded observation error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from manipulation import set_gripper  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402
from vla.atomic_tasks.starts import set_robot_state  # noqa: E402

SHARD = ROOT / 'data/_shards_e23/B_D2'
STEPS = 12


def load_meta():
    rows = [json.loads(x) for x in (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text().splitlines() if x.strip()]
    return [r for r in rows if r['success'] and r['reason'] == 'ok']


def episode_data(ep: int):
    f = SHARD / f'data/chunk-000/episode_{ep:06d}.parquet'
    d = pq.read_table(f, columns=['observation.state', 'action', 'frame_index', 'timestamp']).to_pydict()
    return (np.asarray(d['observation.state'], float), np.asarray(d['action'], float),
            np.asarray(d['frame_index']), np.asarray(d['timestamp']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default='66:39,66:45,22:39,22:30,12:39,38:39,74:39,98:39')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/experiment_2_6/A_midstate_launch/reconstructed')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    wanted = {}
    for s in args.states.split(','):
        e, f = map(int, s.split(':'))
        wanted.setdefault(e, []).append(f)
    report = []
    for ep, frames in wanted.items():
        st, ac, fi, ts = episode_data(ep)
        rec = next(r for r in meta if r['index'] == ep)
        # verify start match
        dist0 = float(np.linalg.norm(st[0, :7] - np.asarray(rec['start_q'])))
        sim, block, wall, tray, info = build_scene(start=tuple(rec['scene']['block_start']),
                                                   yaw=float(rec['scene']['block_yaw']))
        try:
            q0 = st[0, :7]
            set_robot_state(sim, q0, finger=float(0.5 * (st[0, 7] + st[0, 8])))
            sim.command(q0)
            set_gripper(sim, float(0.5 * (st[0, 7] + st[0, 8])), 20.0, 0.05)
            errors = {}
            for t in range(len(ac) - 1):
                a = ac[t]
                sim.command(a[:7])
                set_gripper(sim, float(a[7]), 20.0, 0.05)
                for _ in range(STEPS):
                    sim.step()
                if t + 1 in frames:
                    qerr = float(np.linalg.norm(sim.q() - st[t + 1, :7]))
                    gw = np.asarray(__import__('manipulation').finger_widths(sim))
                    gerr = float(abs(gw.mean() - 0.5 * (st[t + 1, 7] + st[t + 1, 8])))
                    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
                    bullet = args.out / f'ep{ep:03d}_f{t+1:03d}.bullet'
                    p.saveBullet(str(bullet), physicsClientId=sim.client)
                    row = {'episode': ep, 'frame': int(t + 1), 'seed': rec['spec']['seed'],
                           'group': rec['spec']['group'], 'mode': rec['mode'],
                           'start_q_dist': dist0, 'q_error_rad': qerr,
                           'gripper_error_m': gerr, 'gripper_obs_m': float(gw.mean()),
                           'obj_xy': [float(pos[0]), float(pos[1])],
                           'obj_z': float(pos[2]), 'obj_yaw_rad': float(rec['scene']['block_yaw']),
                           'bullet': str(bullet), 'length': int(len(ac))}
                    report.append(row)
                    print(json.dumps(row), flush=True)
        finally:
            p.disconnect(sim.client)
    with (args.out / 'reconstruction_report.json').open('w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
