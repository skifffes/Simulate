import glob
import json
import sys

for sub in ('a90', 'b90'):
    fs = sorted(glob.glob(f'/scratch/lijh/exp43/{sub}/records/*.json'))
    ok = 0
    reasons = {}
    for f in fs:
        r = json.load(open(f))
        if r.get('strict_success'):
            ok += 1
        else:
            reasons[r.get('failure_reason')] = reasons.get(r.get('failure_reason'), 0) + 1
    print(sub, 'attempts', len(fs), 'success', ok,
          'stored', sum(1 for f in fs if json.load(open(f)).get('stored_episode_index') is not None),
          'reasons', reasons)
