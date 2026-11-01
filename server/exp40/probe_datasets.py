"""探查数据集元信息：fps、episode 列表、任务文本、以及首帧 observation.state 与
`b_training_start_states.csv` 的 q 是否一致（用于判定 episode 索引映射是否正确）。

用法：
    python server/exp40/probe_datasets.py
输出到 stdout，并把结果写入 /scratch/lijh/exp40/dataset_probe.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path('/scratch/lijh/simulate_vla/data')
CSV = pathlib.Path('/scratch/lijh/exp40/repo/results/experiment_3/data/b_training_start_states.csv')
OUT = pathlib.Path('/scratch/lijh/exp40/dataset_probe.json')

report = {}
for name in ('lerobot_panda_e23_D2', 'lerobot_panda_multi', 'lerobot_panda_pick_place'):
    base = DATA / name
    if not base.exists():
        report[name] = {'exists': False}
        continue
    info = json.loads((base / 'meta' / 'info.json').read_text(encoding='utf-8'))
    episodes = [json.loads(l) for l in (base / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    tasks = [json.loads(l) for l in (base / 'meta' / 'tasks.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    entry = {'exists': True, 'fps': info.get('fps'),
             'total_episodes': info.get('total_episodes'),
             'total_frames': info.get('total_frames'),
             'chunks_size': info.get('chunks_size'),
             'data_path': info.get('data_path'),
             'features': list(info.get('features', {}).keys()),
             'tasks': [t['task'] for t in tasks][:8],
             'episode_keys': sorted(episodes[0].keys()) if episodes else [],
             'first_episodes': [{k: episodes[i].get(k) for k in ('episode_index', 'length', 'tasks')}
                                for i in range(min(3, len(episodes)))],
             }
    # 首帧 observation.state 采样（前 5 个 episode）
    try:
        df = pd.read_parquet(base / 'data' / 'chunk-000' / 'episode_000000.parquet',
                             columns=['observation.state', 'action'])
        entry['ep0_first_state'] = np.round(np.stack(df['observation.state'].to_numpy())[0], 6).tolist()
        entry['ep0_first_action'] = np.round(np.stack(df['action'].to_numpy())[0], 6).tolist()
        entry['ep0_n_frames'] = int(len(df))
    except Exception as exc:  # pragma: no cover
        entry['ep0_error'] = f'{type(exc).__name__}: {exc}'
    report[name] = entry

# B90 起始状态 CSV 的 episode_id -> q 映射核对
if CSV.exists():
    import csv as _csv
    rows = list(_csv.DictReader(open(CSV, encoding='utf-8')))
    report['b_start_states'] = {
        'n': len(rows),
        'episode_ids': [int(r['episode_id']) for r in rows][:8],
        'first_rows': [{'episode_id': int(r['episode_id']),
                        'q': [round(float(v), 6) for v in r['q'].split(';')],
                        'obj': [round(float(r['obj_x']), 4), round(float(r['obj_y']), 4),
                                round(float(r['obj_z']), 4)]} for r in rows[:3]],
    }

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
