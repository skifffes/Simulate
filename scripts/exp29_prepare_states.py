"""实验 2.9 A：生成物理一致的 Low/High B90 起点（新物体种子）。"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pybullet as p

ROOT = Path('/scratch/lijh/simulate_vla')
sys.path.insert(0, str(ROOT / 'src'))
from vla.atomic_tasks.scene2 import build_scene, PLACE, grasp_quat, block_yaw
from vla.atomic_tasks.starts import (home_array, ik_solve, set_robot_state,
                                     ee_position, robot_contacts)
from manipulation import finger_widths, set_gripper

OUT = ROOT / 'results/experiment_2_9/ab'
STATES = OUT / 'states'; STATES.mkdir(parents=True, exist_ok=True)
DATA = OUT / 'data'; DATA.mkdir(parents=True, exist_ok=True)
WIDTHS = [0.014, 0.025, 0.035, 0.040]


def check_width(sim, block, width):
    before, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    for j in (9, 10):
        p.resetJointState(sim.robot, j, width, physicsClientId=sim.client)
    sim.command(sim.q())
    set_gripper(sim, width, 20.0, 0.05)
    for _ in range(10):
        sim.step()
    after, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    move = float(np.linalg.norm(np.asarray(after) - np.asarray(before)) * 1000)
    contacts = [c for c in p.getContactPoints(bodyA=sim.robot, bodyB=block,
                                               physicsClientId=sim.client)]
    actual = float(np.mean(finger_widths(sim)))
    return {'width_m': width, 'actual_m': actual, 'obj_move_mm': move,
            'block_contacts': len(contacts), 'valid': bool(move < 0.5 and not contacts
                                                           and abs(actual - width) < 0.001)}


def main():
    rows = []
    seed0 = 92000000
    for kind in ('low', 'high'):
        accepted = 0
        for k in range(10):
            seed = seed0 + (0 if kind == 'low' else 5000) + k * 11
            rng = np.random.default_rng(seed)
            yaw = math.pi / 2 + float(rng.uniform(-0.08, 0.08))
            sim, block, wall, tray, info = build_scene(seed=seed, start=PLACE,
                                                       jitter=0.012, yaw=yaw)
            try:
                bp, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
                bp = np.asarray(bp, float)
                if kind == 'low':
                    target = np.array([bp[0] - 0.12, bp[1] + float(rng.uniform(-0.04, 0.04)),
                                       float(rng.uniform(0.16, 0.21))])
                else:
                    target = np.array([bp[0] - 0.20, bp[1] + float(rng.uniform(-0.05, 0.05)),
                                       float(rng.uniform(0.32, 0.40))])
                q, ok = ik_solve(sim, target, grasp_quat(0.0), home_array())
                if not ok:
                    print('reject', kind, k, 'ik'); continue
                set_robot_state(sim, q, finger=0.035)
                sim.command(q)
                set_gripper(sim, 0.035, 20.0, 0.05)
                for _ in range(20):
                    sim.step()
                ee = ee_position(sim)
                err = float(np.linalg.norm(ee - target))
                obj0, _ = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
                contacts = [c for c in p.getContactPoints(bodyA=sim.robot, bodyB=block,
                                                           physicsClientId=sim.client)]
                if err > 0.025 or contacts:
                    print('reject', kind, k, 'err', round(err, 4), 'contacts', len(contacts)); continue
                checks = []
                for w in WIDTHS:
                    for j in (9, 10):
                        p.resetJointState(sim.robot, j, 0.035, physicsClientId=sim.client)
                    set_robot_state(sim, q, finger=0.035)
                    sim.command(q)
                    set_gripper(sim, 0.035, 20.0, 0.05)
                    for _ in range(5):
                        sim.step()
                    checks.append(check_width(sim, block, w))
                if not all(c['valid'] for c in checks):
                    print('reject', kind, k, 'width', [(c['width_m'], c['valid'], round(c['obj_move_mm'], 2)) for c in checks]); continue
                # restore q/35 and save
                set_robot_state(sim, q, finger=0.035)
                sim.command(q)
                set_gripper(sim, 0.035, 20.0, 0.05)
                for _ in range(10):
                    sim.step()
                sid = f'{kind}_{accepted:02d}'
                bullet = STATES / f'{sid}.bullet'
                p.saveBullet(str(bullet), physicsClientId=sim.client)
                obj, oq = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
                row = {'state_id': sid, 'kind': kind, 'seed': seed, 'mode': 'new_low_high',
                       'q': ';'.join(f'{v:.7f}' for v in q),
                       'ee_target_x': float(target[0]), 'ee_target_y': float(target[1]),
                       'ee_target_z': float(target[2]), 'ik_error_m': err,
                       'ee_x': float(ee[0]), 'ee_y': float(ee[1]), 'ee_z': float(ee[2]),
                       'obj_x': float(obj[0]), 'obj_y': float(obj[1]), 'obj_z': float(obj[2]),
                       'obj_yaw_rad': float(block_yaw(block, sim.client)),
                       'distance_xy_m': float(np.linalg.norm(ee[:2] - bp[:2])),
                       'ee_z_above_obj_m': float(ee[2] - bp[2]),
                       'bullet': str(bullet), 'valid': True}
                rows.append(row)
                accepted += 1
                print('ACCEPT', sid, kind, 'seed', seed, 'z', round(ee[2], 3), 'dist', round(row['distance_xy_m'], 3), flush=True)
                if accepted >= 5:
                    break
            finally:
                p.disconnect(sim.client)
        if accepted < 3:
            raise SystemExit(f'only {accepted} {kind} states')
    import csv
    with (DATA / 'state_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('total', len(rows))


if __name__ == '__main__':
    main()
