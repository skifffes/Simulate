import glob
import json
import os
import subprocess
import time

print('time', time.strftime('%H:%M:%S'))
for m in ('ae', 'lora'):
    base = f'/scratch/lijh/exp40/repo/results/experiment_4_5/rollouts/{m}_step12000'
    n = len(glob.glob(base + '/off*/*.json'))
    ok = 0
    for f in glob.glob(base + '/off*/*.json'):
        if f.endswith('_summary.json'):
            continue
        try:
            if json.load(open(f)).get('strict_success'):
                ok += 1
        except Exception:
            pass
    print(m, 'json files', n, 'success', ok)
for f in ('/scratch/lijh/exp45/paired_ae_run.log', '/scratch/lijh/exp45/paired_lora_run.log'):
    try:
        lines = open(f, errors='ignore').read().splitlines()
        print(os.path.basename(f), 'lines', len(lines), '| last:', lines[-1] if lines else '')
    except FileNotFoundError:
        print(f, 'missing')
print(subprocess.run(['df', '-h', '/scratch'], capture_output=True, text=True).stdout)
