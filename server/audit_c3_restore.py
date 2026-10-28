import csv, json, sys
from pathlib import Path
import numpy as np
import pybullet as p
ROOT = Path('/scratch/lijh/simulate_vla')
sys.path.insert(0, str(ROOT / 'src'))
from vla.atomic_tasks.scene2 import build_scene
from manipulation import finger_widths

base = ROOT / 'results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot'
rows = []
for d in sorted(base.glob('seed_*')):
    meta = json.loads((d / 'c3_terminal.json').read_text())
    sim, block, wall, tray, info = build_scene()
    try:
        p.restoreState(fileName=str(ROOT / meta['bullet']), physicsClientId=sim.client)
        q = np.asarray(sim.q(), float)
        f = np.asarray(finger_widths(sim))
        pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        rows.append({'seed': meta['seed'], 'q_error_rad': float(np.linalg.norm(q - np.asarray(meta['q']))),
                     'gripper_error_m': float(np.max(np.abs(f - np.asarray(meta['fingers'])))),
                     'obj_error_mm': float(np.linalg.norm(np.asarray(pos) - np.asarray(meta['obj'])) * 1000),
                     'restored_q': q.tolist(), 'saved_q': meta['q']})
    finally:
        p.disconnect(sim.client)
with (ROOT / 'results/experiment_2_6/C3_direct_B90/state_restore_audit.csv').open('w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=['seed', 'q_error_rad', 'gripper_error_m', 'obj_error_mm'])
    w.writeheader(); w.writerows([{k: r[k] for k in ('seed', 'q_error_rad', 'gripper_error_m', 'obj_error_mm')} for r in rows])
print(json.dumps([{k: round(r[k], 9) for k in ('seed', 'q_error_rad', 'gripper_error_m', 'obj_error_mm')} for r in rows], indent=1))
