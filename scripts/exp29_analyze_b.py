import csv, glob, json, statistics
from pathlib import Path
ROOT = Path('/scratch/lijh/simulate_vla')
BASE = ROOT / 'results/experiment_2_9/ab'
OUT = BASE / 'data'; OUT.mkdir(parents=True, exist_ok=True)

def parse_tag(tag):
    p = tag.split('_g')
    sid = p[0]; rest = p[1]
    g, rep = rest.split('_r')
    return sid, int(g) / 1000.0, int(rep)

rows = []
for f in sorted((BASE / 'runs').glob('*/*/*/*/summary.json')):
    d = f.parent
    rel = d.relative_to(BASE / 'runs')
    kind = rel.parts[0]
    tag = rel.parts[1]
    sid, width, rep = parse_tag(tag)
    s = json.loads(f.read_text())
    ph = list(csv.DictReader((d / 'physics_trace.csv').open(encoding='utf-8-sig')))
    last = ph[-1]
    first = None
    for c in csv.DictReader((d / 'contact_trace.csv').open(encoding='utf-8-sig')):
        if c.get('bodyA') == '1' or c.get('bodyB') == '1':
            first = c; break
    ee = ''
    if first:
        t = float(first['physics_time_sec'])
        rr = min(ph, key=lambda x: abs(float(x['physics_time_sec']) - t))
        ee = float(rr['ee_z'])
    rows.append({'condition': kind, 'state_id': sid, 'width_m': width, 'rep': rep,
                 'success': bool(s['success']), 'max_lift_m': float(s['max_lift_m']),
                 'wall_contact': bool(s['wall_contact']),
                 'final_gap_m': float(last['obj_wall_gap_m']),
                 'final_dy_m': float(last['obj_dy_m']),
                 'final_tilt_deg': float(last['obj_tilt_deg']),
                 'first_link': (first['nameA'] if first and first['bodyA'] == '1' else first['nameB']) if first else '',
                 'first_time_s': float(first['physics_time_sec']) if first else '',
                 'normal_z': float(first['normal_z']) if first else '',
                 'ee_z_at_contact': ee, 'run_dir': str(d)})
with (OUT / 'all_rollouts.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with (OUT / 'expert_rollouts.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows([r for r in rows if r['condition'] in ('E0', 'E1')])
with (OUT / 'frozen_policy_rollouts.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows([r for r in rows if r['condition'] == 'D2'])
with (OUT / 'contact_analysis.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows([r for r in rows if r['first_link']])

comp = []
for sid in sorted(set(r['state_id'] for r in rows)):
    for w in sorted(set(r['width_m'] for r in rows)):
        sub = [r for r in rows if r['state_id'] == sid and r['width_m'] == w]
        e0 = next((r for r in sub if r['condition'] == 'E0'), None)
        e1 = next((r for r in sub if r['condition'] == 'E1'), None)
        d2 = [r for r in sub if r['condition'] == 'D2']
        comp.append({'state_id': sid, 'width_m': w,
                     'E0_success': int(bool(e0 and e0['success'])),
                     'E1_success': int(bool(e1 and e1['success'])),
                     'D2_success_n': sum(r['success'] for r in d2), 'D2_n': len(d2),
                     'D2_success_rate': sum(r['success'] for r in d2) / max(1, len(d2)),
                     'E1_lift_m': e1['max_lift_m'] if e1 else '',
                     'D2_lift_mean_m': statistics.mean([r['max_lift_m'] for r in d2]) if d2 else '',
                     'E1_first_link': (e1['first_link'] if e1 else ''),
                     'D2_first_links': ';'.join(r['first_link'] for r in d2),
                     'E1_final_gap_m': e1['final_gap_m'] if e1 else '',
                     'D2_final_gaps_m': ';'.join(f"{r['final_gap_m']:.4f}" for r in d2)})
with (OUT / 'expert_policy_comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(comp[0].keys())); w.writeheader(); w.writerows(comp)
with (OUT / 'failure_cases.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows([r for r in rows if not r['success']])
print(len(rows), len(comp), sum(not r['success'] for r in rows))
