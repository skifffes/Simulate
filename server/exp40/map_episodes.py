"""把数据集 episode 与实验 3 的 B90 起始状态 CSV 对应起来，并给出 A90/B90 的可用 episode 清单。

关键点（上一轮的根因）：
    `lerobot_panda_e23_D2` 里有 **480 个 episode，混合了 A0/A90/B90/C90 四种任务**
    （`meta/episodes.jsonl` 的 `tasks` 字段逐条给出）。上一轮把 `episode_index=64` 直接当成
    B90 的 `episode_id=64`，而索引 64 其实是别的任务 → 恢复的机器人起点与回放的动作不属于同一集，
    方块当然推不动。本脚本用**首帧关节 q 与 CSV 的 q 逐条比对**来确定真正的对应关系。

用法：
    python server/exp40/map_episodes.py
输出：/scratch/lijh/exp40/episode_mapping.json
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path('/scratch/lijh/simulate_vla/data/lerobot_panda_e23_D2')
CSV = pathlib.Path('/scratch/lijh/exp40/repo/results/experiment_3/data/b_training_start_states.csv')
OUT = pathlib.Path('/scratch/lijh/exp40/episode_mapping.json')

TASK_KEYS = {'A0': 'keeping its long side aligned with Y',
             'A90': 'with its long side aligned with X',
             'B90': 'push the orange block along the positive X direction to the wall',
             'C90': 'place it into the tray'}


def first_state(base: pathlib.Path, idx: int) -> np.ndarray | None:
    p = base / 'data' / f'chunk-{idx // 1000:03d}' / f'episode_{idx:06d}.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=['observation.state'])
    return np.stack(df['observation.state'].to_numpy()).astype(float)


def main() -> int:
    base = DATA
    eps = [json.loads(l) for l in (base / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    info = json.loads((base / 'meta' / 'info.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
    csv_q = np.array([[float(v) for v in r['q'].split(';')] for r in rows])
    csv_ids = [int(r['episode_id']) for r in rows]

    per_task = {}
    for name, key in TASK_KEYS.items():
        idxs = [int(e['episode_index']) for e in eps if any(key in t for t in e.get('tasks', []))]
        per_task[name] = idxs

    mapping = {}
    exact = []
    for name in ('B90', 'A90'):
        candidates = []
        for idx in per_task[name]:
            st = first_state(base, idx)
            if st is None:
                continue
            q0 = st[0, :7]
            if name == 'B90':
                d = np.linalg.norm(csv_q - q0, axis=1)
                j = int(np.argmin(d))
                candidates.append({'episode_index': idx, 'matched_episode_id': csv_ids[j],
                                   'q_distance': float(d[j]), 'n_frames': int(st.shape[0])})
                if float(d[j]) < 1e-6:
                    exact.append({'episode_index': idx, 'episode_id': csv_ids[j],
                                  'n_frames': int(st.shape[0]),
                                  'obj': [float(rows[j]['obj_x']), float(rows[j]['obj_y']),
                                          float(rows[j]['obj_z'])],
                                  'obj_quat': [float(v) for v in rows[j]['obj_quat'].split(';')]})
            else:
                candidates.append({'episode_index': idx, 'n_frames': int(st.shape[0]),
                                   'first_q': np.round(q0, 5).tolist(),
                                   'first_obj_note': 'A90 起始物体位姿未记录在数据集内（只有关节/夹爪）'})
        mapping[name] = {'n_episodes': len(per_task[name]), 'candidates': candidates}

    out = {'dataset': str(base), 'fps': info.get('fps'),
           'total_episodes': info.get('total_episodes'),
           'tasks_in_dataset': sorted({t for e in eps for t in e.get('tasks', [])}),
           'episodes_per_task': {k: len(v) for k, v in per_task.items()},
           'b90_exact_matches': exact,
           'b90_n_exact': len(exact),
           'b90_n_candidates': len(mapping['B90']['candidates']),
           'note': ('B90 episode_index 与 b_training_start_states.csv 的 episode_id 不是同一编号；'
                    '用首帧 q 的 L2 距离配对，< 1e-6 视为同一集（这些 episode 的机器人起点与 CSV 完全一致，'
                    '因此可以用对应 CSV 行的物体位姿恢复场景）。'),
           'b90_mapping_sample': mapping['B90']['candidates'][:12],
           'a90_mapping_sample': mapping['A90']['candidates'][:6],
           'full': mapping}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('fps', out['fps'], 'episodes_per_task', out['episodes_per_task'])
    print('b90 exact matches:', out['b90_n_exact'], 'of', out['b90_n_candidates'])
    for r in out['b90_exact_matches'][:10]:
        print('  ', r)
    print('A90 sample:')
    for r in out['a90_mapping_sample']:
        print('  ', r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
