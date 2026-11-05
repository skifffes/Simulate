import json
from pathlib import Path

import numpy as np
import pandas as pd

SRV = Path('/scratch/lijh/exp43')
MAN = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion/train_manifest_v2.json')
man = json.loads(MAN.read_text(encoding='utf-8'))
e = next(x for x in man['episodes'] if x['skill'] == 'A90')
df = pd.read_parquet(SRV / e['parquet'])
st = np.stack(df['observation.state'].to_numpy()).astype(float)
ac = np.stack(df['action'].to_numpy()).astype(float)
gcmd = np.clip(ac[:, 7] / 0.04, 0, 1)
print('episode', e['global_episode_id'], 'frames', len(df))
print('gripper cmd unique (first 5):', np.round(np.unique(ac[:5, 7]), 4))
print('gripper cmd min/max:', round(float(ac[:, 7].min()), 4), round(float(ac[:, 7].max()), 4))
print('gcmd<0.6 count:', int((gcmd < 0.6).sum()), 'first idx:', int(np.where(gcmd < 0.6)[0][0]) if (gcmd < 0.6).any() else None)
print('q z proxy (joint4/6):', np.round(st[:3, 3], 3).tolist(), '->', np.round(st[-3:, 3], 3).tolist())
print('lift_m metric:', (e.get('metrics') or {}).get('lift_m'))
