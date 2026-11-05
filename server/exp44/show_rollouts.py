import glob
import json
import os

L = '/scratch/lijh/exp40/repo/results/experiment_4_4/rollouts'
rows = []
for d in sorted(glob.glob(L + '/*')):
    if not os.path.isdir(d):
        continue
    tag = os.path.basename(d)
    for f in sorted(glob.glob(d + '/*.json')):
        if f.endswith('_summary.json'):
            continue
        j = json.load(open(f))
        m = j.get('metrics') or {}
        rows.append({
            'tag': tag, 'skill': j.get('skill'), 'ep': j.get('episode'),
            'success': j.get('strict_success'), 'reason': j.get('failure_reason'),
            'adv': m.get('advance_x_m'), 'gap': m.get('terminal_gap_m'),
            'wall': m.get('wall_contact_steps'), 'rb': m.get('robot_block_contact_steps'),
            'force': m.get('wall_contact_force_max_n'), 'tilt': m.get('tilt_deg'),
            'lift': m.get('lift_m'), 'pos_err': m.get('position_error_m'),
            'n_frames': j.get('n_infer'),
        })
print(len(rows), 'rollouts')
for r in rows:
    def f(v, nd=3):
        return '—' if v is None else round(v, nd)
    print(f"{r['tag']:24s} {r['skill']} ep{r['ep']} ok={r['success']} {str(r['reason'])[:22]:22s} "
          f"adv={f(r['adv'])} gap={f(r['gap'],4)} wall={r['wall']} rb={r['rb']} "
          f"force={f(r['force'],1)} tilt={f(r['tilt'],2)} lift={f(r['lift'])}")
json.dump(rows, open('/tmp/roll_rows.json', 'w'), ensure_ascii=False, indent=1)
