import csv, json, sys
from pathlib import Path
import numpy as np
import pybullet as p
ROOT = Path('/scratch/lijh/simulate_vla')
sys.path.insert(0, str(ROOT / 'src'))
from vla.atomic_tasks.scene2 import build_scene
from manipulation import finger_widths

OUT = ROOT / 'results/experiment_2_8'
DATA = OUT / 'data'; DATA.mkdir(parents=True, exist_ok=True)
rep = json.loads((OUT / 'states/reconstructed/reconstruction_report.json').read_text())
rows = []
for r in rep:
    sim, block, wall, tray, info = build_scene()
    try:
        p.restoreState(fileName=str(ROOT / r['bullet']), physicsClientId=sim.client)
        q = np.asarray(sim.q(), float); f = np.asarray(finger_widths(sim))
        ee, quat = sim.pose()
        pos, oq = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        # object yaw from quaternion
        yaw = float(np.arctan2(2 * (oq[3] * oq[2] + oq[0] * oq[1]),
                               1 - 2 * (oq[1] ** 2 + oq[2] ** 2)))
        prog = r['frame'] / max(1, r['length'] - 1)
        stage = ('early' if prog < 0.15 else 'approach' if prog < 0.3
                 else 'precontact' if prog < 0.4 else 'push')
        sid = f"ep{r['episode']:03d}_f{r['frame']:03d}"
        rows.append({'state_id': sid, 'episode_id': r['episode'], 'frame_id': r['frame'],
                     'episode_progress': round(prog, 4), 'yaw': r['group'], 'mode': r['mode'],
                     'stage_progress_label': stage,
                     'q': ';'.join(f'{v:.6f}' for v in q),
                     'grip_obs_m': float(f.mean()), 'grip_left_m': float(f[0]),
                     'grip_right_m': float(f[1]),
                     'ee_x': float(ee[0]), 'ee_y': float(ee[1]), 'ee_z': float(ee[2]),
                     'obj_x': float(pos[0]), 'obj_y': float(pos[1]), 'obj_z': float(pos[2]),
                     'obj_yaw_rad': yaw, 'q_replay_error_rad': r['q_error_rad'],
                     'bullet': r['bullet'], 'valid': r['q_error_rad'] < 1e-4})
    finally:
        p.disconnect(sim.client)
with (DATA / 'state_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), 'states')
