import csv, json
from pathlib import Path
import numpy as np
ROOT = Path('/scratch/lijh/simulate_vla')
OUT = ROOT / 'results/experiment_2_8'; DATA = OUT / 'data'
states = list(csv.DictReader((DATA / 'state_manifest.csv').open(encoding='utf-8-sig')))
tr = list(csv.DictReader((DATA / 'expert_dynamics.csv').open(encoding='utf-8-sig')))
Q = np.array([[float(r[f'q{j}']) for j in range(1, 8)] for r in tr])
G = np.array([float(r['gripper_per_finger_mm']) for r in tr])
V = np.array([float(r['vz_mm_s']) for r in tr])
E = np.array([int(r['episode_id']) for r in tr])
EZ = np.array([float(r['ee_z']) for r in tr])
neigh = []
cov = []
for s in states:
    q = np.array([float(x) for x in s['q'].split(';')])
    d = np.linalg.norm(Q - q, axis=1)
    order = np.argsort(d)
    for rank, k in enumerate(order[:20], 1):
        if d[k] > 0.75:
            break
        neigh.append({'state_id': s['state_id'], 'rank': rank,
                      'train_episode': int(E[k]),
                      'train_frame': int(tr[k]['frame_id']),
                      'joint_distance_rad': float(d[k]),
                      'ee_distance_mm': abs(float(s['ee_z']) - EZ[k]) * 1000,
                      'grip_obs_mm': float(G[k]), 'expert_vz_mm_s': float(V[k]),
                      'progress': float(tr[k]['progress'])})
    for rad in (0.15, 0.30, 0.50, 0.75):
        m = d <= rad
        if not m.any():
            cov.append({'state_id': s['state_id'], 'radius_rad': rad, 'n_frames': 0,
                        'episodes': 0, 'has_common_support': False})
            continue
        near14 = int(((G[m] >= 11) & (G[m] <= 17)).sum())
        near35 = int(((G[m] >= 32) & (G[m] <= 41)).sum())
        cov.append({'state_id': s['state_id'], 'radius_rad': rad, 'n_frames': int(m.sum()),
                    'episodes': int(len(set(E[m].tolist()))),
                    'grip_min_mm': float(G[m].min()), 'grip_max_mm': float(G[m].max()),
                    'grip_p05_mm': float(np.percentile(G[m], 5)),
                    'grip_p95_mm': float(np.percentile(G[m], 95)),
                    'near14_frames': near14, 'near35_frames': near35,
                    'has_common_support': bool(near14 >= 5 and near35 >= 5),
                    'vz_mean_mm_s': float(V[m].mean()), 'vz_std_mm_s': float(V[m].std()),
                    'nearest_joint_rad': float(d.min()),
                    'nearest_grip_mm': float(G[int(np.argmin(d))])})
with (DATA / 'coverage_neighbors.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(neigh[0].keys())); w.writeheader(); w.writerows(neigh)
with (DATA / 'coverage_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(cov[0].keys())); w.writeheader(); w.writerows(cov)
print(len(cov), 'coverage rows')
