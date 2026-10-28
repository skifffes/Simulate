import csv, json
from pathlib import Path

ROOT = Path('/scratch/lijh/simulate_vla')
BASE = ROOT / 'results/experiment_2_6'
audit = {r['seed']: r for r in csv.DictReader((BASE / 'C3_direct_B90/state_restore_audit.csv').open(encoding='utf-8-sig'))}
c3rows = []
for d in sorted((BASE / 'C3_direct_B90/states/G1/C3_snapshot').glob('seed_*')):
    m = json.loads((d / 'c3_terminal.json').read_text())
    a = audit.get(str(m['seed']), {})
    c3rows.append({'seed': m['seed'], 'q1..q7': ';'.join(f'{v:.6f}' for v in m['q']),
                   'grip_actual_m': m['grip_actual'], 'ee_x': m['ee'][0], 'ee_y': m['ee'][1],
                   'ee_z': m['ee'][2], 'obj_x': m['obj'][0], 'obj_y': m['obj'][1],
                   'obj_z': m['obj'][2], 'bullet': m['bullet'],
                   'q_restore_error_rad': a.get('q_error_rad'),
                   'obj_restore_error_mm': a.get('obj_error_mm')})
with (BASE / 'C3_direct_B90/state_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(c3rows[0].keys())); w.writeheader(); w.writerows(c3rows)

brows = []
for p in sorted((BASE / 'B90_gripper_extension/states').glob('ep*.json')):
    s = json.loads(p.read_text())
    if s.get('group') != 'yaw90':
        continue
    brows.append({'state': s['tag'], 'episode': s['episode'], 'frame': s['frame'],
                  'group': s['group'], 'mode': s['mode'], 'seed': s['seed'],
                  'q_replay_error_rad': s.get('q_error_rad'), 'bullet': s['bullet']})
with (BASE / 'B90_gripper_extension/state_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)

# video manifest
rows = []
for group, conds in [('C3_direct_B90', ['frozen', 'expert']),
                     ('B90_gripper_extension', ['frozen', 'expert'])]:
    for cond in conds:
        for d in sorted((BASE / group / cond).glob('*')):
            if not (d / 'summary.json').exists():
                continue
            s = json.loads((d / 'summary.json').read_text())
            if d.name.startswith('ep0') and 'seed' not in d.name and group == 'C3_direct_B90':
                continue
            rows.append({'group': group, 'condition': cond, 'state': d.name,
                         'success': s['success'],
                         'side': str(d / 'side.mp4'), 'top': str(d / 'top.mp4'),
                         'contact': str(d / 'contact.mp4'),
                         'comparison': str(d / 'comparison.mp4'),
                         'events': str(d / 'events.mp4')})
with (BASE / 'b90_extension_video_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), 'videos')
