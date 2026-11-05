import glob
import json
import time

rows = []
for tag in sorted(glob.glob('/scratch/lijh/exp43/a90*') + glob.glob('/scratch/lijh/exp43/b90*')):
    fs = sorted(glob.glob(f'{tag}/records/*.json'))
    if not fs:
        continue
    ok = 0
    stored = 0
    reasons = {}
    for f in fs:
        r = json.load(open(f))
        if r.get('strict_success'):
            ok += 1
            if r.get('stored_episode_index') is not None:
                stored += 1
        else:
            reasons[r.get('failure_reason')] = reasons.get(r.get('failure_reason'), 0) + 1
    rows.append((tag.split('/')[-1], len(fs), ok, stored, reasons))
print('time', time.strftime('%H:%M:%S'))
for tag, n, ok, stored, reasons in rows:
    print(f'{tag:8s} attempts={n:3d} success={ok:3d} stored={stored:3d} {reasons}')
print('TOTAL stored:', sum(r[3] for r in rows))
