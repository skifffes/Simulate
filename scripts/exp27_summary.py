import csv, json, math
from pathlib import Path
ROOT = Path('/scratch/lijh/simulate_vla')
BASE = ROOT / 'results/experiment_2_7'

# cross replay stats
rows = []
for d in sorted((BASE / 'crossed_action_replay').iterdir()):
    if not (d / 'summary.json').exists():
        continue
    s = json.loads((d / 'summary.json').read_text())
    ph = list(csv.DictReader((d / 'physics_trace.csv').open(encoding='utf-8-sig')))[-1]
    first = None
    for r in csv.DictReader((d / 'contact_trace.csv').open(encoding='utf-8-sig')):
        if r.get('bodyA') == '1' or r.get('bodyB') == '1':
            first = r; break
    rows.append({'condition': d.name, 'success': s['success'],
                 'max_lift_m': s['max_lift_m'], 'wall_contact': s['wall_contact'],
                 'final_gap_m': float(ph['obj_wall_gap_m']),
                 'final_dy_m': float(ph['obj_dy_m']),
                 'final_tilt_deg': float(ph['obj_tilt_deg']),
                 'first_contact_link': (first['nameA'] if first and first['bodyA'] == '1' else first['nameB']) if first else None,
                 'first_contact_time': float(first['physics_time_sec']) if first else None})
with (BASE / 'crossed_action_replay/contact_statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# C3 actual EE height, 14 vs 40
c3 = [('c3_14mm', ROOT / 'results/experiment_2_6/C3_seed40000000_frozen_14mm_x10/run00/frozen/seed_40000000_g014/physics_trace.csv'),
      ('c3_40mm', ROOT / 'results/experiment_2_6/C3_direct_B90/frozen/seed_40000000/physics_trace.csv')]
with (BASE / 'c3_ee_height.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['condition', 'tick', 'time_s', 'ee_z', 'obj_tilt', 'left', 'right'])
    for name, path in c3:
        for r in list(csv.DictReader(path.open(encoding='utf-8-sig')))[:80]:
            w.writerow([name, r['physics_step'], r['physics_time_sec'], r['ee_z'],
                        r['obj_tilt_deg'], r['left_finger_contact'], r['right_finger_contact']])
print(json.dumps(rows, indent=1))
