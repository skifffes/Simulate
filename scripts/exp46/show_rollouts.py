import glob
import json

for step in ('1500', '3000'):
    print('== step', step)
    for f in sorted(glob.glob(f'/scratch/lijh/exp40/repo/results/experiment_4_6/rollouts/step{step}/off*/*.json')):
        if f.endswith('_summary.json'):
            continue
        j = json.load(open(f))
        m = j.get('metrics') or {}
        print(' ', j['skill'], 'ep', j['episode'], 'off', f.split('off')[-1].split('/')[0],
              'ok', j['strict_success'], str(j['failure_reason'])[:22],
              'adv', round(m.get('advance_x_m') or 0, 3),
              'gap', round(m.get('terminal_gap_m') or 0, 4),
              'wall', m.get('wall_contact_steps'),
              'lift', round(m.get('lift_m') or 0, 3),
              'poserr', round(m.get('position_error_m') or 0, 4),
              'twofin', m.get('two_finger_overlap_steps'))
