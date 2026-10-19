"""里程碑 2：A 的两种终态接 B/C，并与标准化恢复对照。"""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulation import Simulation, HOME
from robot_controller import RobotController
from skills import Z1, Z2, B, C, skill_success

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
N = 10
SEED = 20261019


def run_once(implementation, successor, canonical=False):
    sim = Simulation(gui=False)
    ctl = RobotController(sim)
    try:
        a = Z1 if implementation == 'z1' else Z2
        a_result = ctl.move_pose(a.position, a.orientation)
        a_success = bool(a_result.success and skill_success(sim, a))
        a_q = sim.q().tolist()
        a_pose, a_orn = sim.pose()
        if canonical:
            recovery = ctl.move(HOME)
        else:
            recovery = None
        target = B if successor == 'B' else C
        successor_result = ctl.move_pose(target.position, target.orientation)
        return {
            'implementation': implementation, 'successor': successor, 'canonical': canonical,
            'a_success': a_success,
            'a_reason': a_result.reason, 'a_seconds': a_result.seconds, 'a_distance': a_result.distance,
            'a_q': json.dumps(a_q), 'a_position_m': json.dumps(a_pose.tolist()),
            'a_orientation_xyzw': json.dumps(a_orn.tolist()),
            'recovery_success': None if recovery is None else recovery.success,
            'recovery_reason': None if recovery is None else recovery.reason,
            'recovery_seconds': None if recovery is None else recovery.seconds,
            'recovery_distance': None if recovery is None else recovery.distance,
            'ik_solvable': successor_result.ik, 'collision': successor_result.collision,
            'joint_limit': successor_result.limit, 'reached': successor_result.reached,
            'successor_success': successor_result.success,
            'successor_reason': successor_result.reason, 'successor_seconds': successor_result.seconds,
            'successor_distance': successor_result.distance, 'joint_tracking_error_rad': successor_result.error,
            'position_error_actual_m': getattr(successor_result, 'position_error_m', None),
            'orientation_error_rad': getattr(successor_result, 'orientation_error_rad', None),
        }
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=N)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    np.random.seed(args.seed)
    OUT.mkdir(exist_ok=True)
    rows = []
    for canonical in (False, True):
        for implementation in ('z1', 'z2'):
            for successor in ('B', 'C'):
                for trial in range(args.trials):
                    row = run_once(implementation, successor, canonical=canonical)
                    prefix = 'reset:' if canonical else 'direct:'
                    row.update(trial=trial, seed=args.seed, condition=f'{prefix}{implementation}->{successor}')
                    rows.append(row)
    fields = list(rows[0])
    with (OUT / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = []
    for condition in sorted({r['condition'] for r in rows}):
        group = [r for r in rows if r['condition'] == condition]
        summary.append({'condition': condition, 'n': len(group),
                        'a_success_rate': np.mean([r['a_success'] for r in group]),
                        'ik_rate': np.mean([r['ik_solvable'] for r in group]),
                        'reached_rate': np.mean([r['reached'] for r in group]),
                        'success_rate': np.mean([r['successor_success'] for r in group]),
                        'mean_successor_seconds': np.mean([r['successor_seconds'] for r in group]),
                        'recovery_success_rate': None if not group[0]['canonical'] else float(np.mean([r['recovery_success'] for r in group])),
                        'chain_success_rate': float(np.mean([r['a_success'] and r['successor_success'] and (not r['canonical'] or r['recovery_success']) for r in group])),
                        'mean_recovery_seconds': float(np.mean([r['recovery_seconds'] or 0 for r in group])),
                        'mean_transition_seconds': float(np.mean([(r['recovery_seconds'] or 0) + r['successor_seconds'] for r in group])),
                        'mean_transition_distance_rad': float(np.mean([(r['recovery_distance'] or 0) + r['successor_distance'] for r in group]))})
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (OUT / 'experiment_config.json').write_text(json.dumps({'seed': args.seed, 'trials': args.trials, 'fixed_timestep': 1/240,
        'home_joint_rad': HOME, 'targets': {t.name: {'position_m': t.position, 'orientation_xyzw': t.orientation} for t in (Z1, Z2, B, C)},
        'controller': {'max_interpolation_steps': 1800, 'joint_step_norm_rad': 0.004, 'settle_steps': 60,
                       'joint_tracking_tolerance_rad': 0.06, 'position_tolerance_m': 0.025, 'orientation_tolerance_rad': 0.15},
        'note': 'Deterministic repeated runs; not independent random physics trials.'}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
