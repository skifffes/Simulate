import csv, json
from pathlib import Path
import numpy as np
ROOT = Path('/scratch/lijh/simulate_vla')
OUT = ROOT / 'results/experiment_2_8'; DATA = OUT / 'data'; POL = OUT / 'policy'
# merge sensitivity
rows = []
for p in sorted(POL.glob('sensitivity_*.csv')):
    rows += list(csv.DictReader(p.open(encoding='utf-8-sig')))
with (DATA / 'policy_sensitivity.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# merge predictions (large, server-only)
pred_files = sorted(POL.glob('predictions_*.csv'))
if pred_files:
    with (DATA / 'policy_predictions.csv').open('w', newline='', encoding='utf-8-sig') as out:
        first = True
        for p in pred_files:
            with p.open(encoding='utf-8-sig') as f:
                r = csv.reader(f); hdr = next(r)
                if first:
                    w = csv.writer(out); w.writerow(hdr); first = False
                for row in r: w.writerow(row)
states = {r['state_id']: r for r in csv.DictReader((DATA / 'state_manifest.csv').open(encoding='utf-8-sig'))}
cov = list(csv.DictReader((DATA / 'coverage_summary.csv').open(encoding='utf-8-sig')))
sens = {(r['state_id'], r['intervention'], int(r['chunk_step'])): r for r in rows}
out_rows = []
for sid, s in states.items():
    c3 = next((r for r in cov if r['state_id'] == sid and abs(float(r['radius_rad']) - 0.30) < 1e-6), {})
    c75 = next((r for r in cov if r['state_id'] == sid and abs(float(r['radius_rad']) - 0.75) < 1e-6), {})
    r = sens.get((sid, '14-40mm', 9))
    if not r:
        continue
    out_rows.append({'state_id': sid, 'episode_id': s['episode_id'], 'frame_id': s['frame_id'],
                     'stage': s['stage_progress_label'], 'progress': s['episode_progress'],
                     'grip_obs_mm': float(s['grip_obs_m']) * 1000, 'ee_z_m': s['ee_z'],
                     'sensitivity_mm_per_mm': r['sensitivity_mm_per_mm'],
                     'mean_delta_z_mm': r['mean_delta_z_mm'],
                     'std_delta_z_mm': r['std_delta_z_mm'],
                     'positive_fraction': r['positive_fraction'],
                     'coverage_0.30_frames': c3.get('n_frames', 0),
                     'coverage_0.30_episodes': c3.get('episodes', 0),
                     'coverage_0.30_grip_max_mm': c3.get('grip_max_mm', ''),
                     'coverage_0.30_near35': c3.get('near35_frames', 0),
                     'coverage_0.75_frames': c75.get('n_frames', 0),
                     'coverage_0.75_episodes': c75.get('episodes', 0),
                     'coverage_0.75_grip_max_mm': c75.get('grip_max_mm', ''),
                     'coverage_0.75_near35': c75.get('near35_frames', 0),
                     'has_common_support': c75.get('has_common_support', False)})
with (DATA / 'cross_state_analysis.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
dz = np.array([float(r['mean_delta_z_mm']) for r in out_rows])
print('states', len(out_rows), 'mean', float(dz.mean()), 'min', float(dz.min()),
      'max', float(dz.max()), 'positive', int((dz > 1).sum()), 'negative', int((dz < -1).sum()),
      'nearzero', int((abs(dz) <= 1).sum()))
print(json.dumps({'n_states': len(out_rows), 'mean_delta_z_mm': float(dz.mean()),
                  'min_delta_z_mm': float(dz.min()), 'max_delta_z_mm': float(dz.max()),
                  'positive_states': int((dz > 1).sum()), 'negative_states': int((dz < -1).sum()),
                  'near_zero_states': int((abs(dz) <= 1).sum())}, indent=2))
