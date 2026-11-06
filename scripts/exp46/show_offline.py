import json
import glob
from pathlib import Path

for f in sorted(glob.glob('/scratch/lijh/exp40/repo/results/experiment_4_6/offline/step*_stats.json')):
    d = json.load(open(f))
    o = d['overall']
    print(Path(f).name.replace('_stats.json', ''),
          'MAE', round(o['joint_mae_mean'], 4),
          '| A90', round(d['by_skill']['A90']['joint_mae_mean'], 4),
          '| B90', round(d['by_skill']['B90']['joint_mae_mean'], 4),
          '| grip', round(o['grip_err_mean'], 4),
          '| seedstd', round(o['seed_mae_std_mean'], 4),
          '| cos_med', round((d['by_skill']['A90'].get('cos_median') or 0), 3))
for f in sorted(glob.glob('/scratch/lijh/exp40/repo/results/experiment_4_6/offline/step*_chunk.json')):
    print(Path(f).name, json.load(open(f)))
