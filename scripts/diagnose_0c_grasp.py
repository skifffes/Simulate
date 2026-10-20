"""实验 0C 负对照：扫描工件摩擦系数与夹爪夹持力。

目的不是调参，而是确认：
  1) 成功依赖真实摩擦与夹持力，而不是任何隐藏约束；
  2) 失败判定器确实能在低摩擦/低夹持力下报出失败原因。

用法:
    python scripts/diagnose_0c_grasp.py
"""
from pathlib import Path
import csv
import json
import sys

import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_experiment_0c import build_scene
from robot_controller import RobotController
from grasp_task import DEFAULT_PARAMS, GraspPlaceTask

OUT = ROOT / 'results' / 'experiment_0c_diagnose'
FRICTIONS = (0.05, 0.2, 0.6, 1.2)
FORCES = (0.5, 2.0, 8.0, 20.0)


def trial(friction, force):
    params = {key: (tuple(value) if isinstance(value, tuple) else value)
              for key, value in DEFAULT_PARAMS.items()}
    params['block_friction'] = friction
    params['grasp_force'] = force
    sim, block = build_scene(params)
    task = GraspPlaceTask(sim, RobotController(sim), block, params=params)
    task.run()
    result = task.result()
    sim.close()
    return {
        'friction': friction,
        'grasp_force_n': force,
        'success': result['success'],
        'failure': result['failure'] or '',
        'failure_stage': result['failure_stage'] or '',
        'meas_force_1': round(result['grasp_contact']['force_1'], 3),
        'meas_force_2': round(result['grasp_contact']['force_2'], 3),
        'lift_height_m': round(result['lift_height_m'], 4),
        'slip_m': round(result['slip_m'], 4),
        'place_xy_error_m': round(result['place_xy_error_m'], 4),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for friction in FRICTIONS:
        for force in FORCES:
            row = trial(friction, force)
            rows.append(row)
            print(f"mu={friction:<4} force={force:<5} success={str(row['success']):<5} "
                  f"failure={row['failure'] or '-':<20} lift={row['lift_height_m']:.3f} "
                  f"slip={row['slip_m']:.3f}")
    with (OUT / 'grid.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'frictions': list(FRICTIONS),
        'forces': list(FORCES),
        'success_count': sum(1 for r in rows if r['success']),
        'total': len(rows),
        'failures': sorted({r['failure'] for r in rows if r['failure']}),
        'note': '负对照：只改变工件摩擦与夹爪电机力上限，其余流程完全一致。',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding='utf-8')
    print(f"{summary['success_count']}/{summary['total']} succeed -> {OUT}")


if __name__ == '__main__':
    main()
