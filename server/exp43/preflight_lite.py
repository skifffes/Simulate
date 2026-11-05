import glob
import json

import numpy as np
import pandas as pd


def check(ddir: str, skill: str):
    eps = sorted(glob.glob(f'/scratch/lijh/exp43/{ddir}/{skill}/data/**/*.parquet',
                           recursive=True))
    if not eps:
        return {'error': 'no parquet', 'dataset': f'{ddir}/{skill}'}
    df = pd.read_parquet(eps[0])
    cols = list(df.columns)
    st = np.stack(df['observation.state'].to_numpy()) if 'observation.state' in cols else None
    ac = np.stack(df['action'].to_numpy()) if 'action' in cols else None
    out = {
        'dataset': f'{ddir}/{skill}', 'n_episodes': len(eps),
        'columns': cols, 'rows': len(df),
        'state_shape': list(st.shape) if st is not None else None,
        'action_shape': list(ac.shape) if ac is not None else None,
        'has_nan': bool((np.isnan(st).any() if st is not None else False)
                         or (np.isnan(ac).any() if ac is not None else False)),
        'n_videos': len(glob.glob(f'/scratch/lijh/exp43/{ddir}/{skill}/videos/**/*.mp4',
                                  recursive=True)),
    }
    if st is not None and ac is not None and ac.shape[1] > 7:
        out['dq_absmax'] = float(np.abs(ac[:, :7] - st[:, :7]).max())
        out['grip_min'] = float(ac[:, 7].min())
        out['grip_max'] = float(ac[:, 7].max())
    if 'task' in cols:
        out['task'] = str(df['task'].iloc[0])[:80]
    if 'timestamp' in cols:
        out['ts_monotonic'] = bool(np.all(np.diff(df['timestamp'].to_numpy()) > 0))
    return out


print(json.dumps([check('data_a90a', 'A90'), check('data_b90a', 'B90')],
                 ensure_ascii=False, indent=2))
