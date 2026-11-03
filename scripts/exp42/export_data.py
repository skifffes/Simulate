"""实验 4.2 数据导出：把 4.1 的 LeRobot 数据展开成训练可直接随机访问的数组。

产出（`/scratch/lijh/exp42/data/`）：
    {skill}/ep{idx:03d}_top.npy       uint8 (T,256,256,3)  外部相机
    {skill}/ep{idx:03d}_wrist.npy     uint8 (T,256,256,3)  腕部相机
    meta.jsonl                        每个 episode 一行：state/action 路径、任务文本、split、goal/config
    arrays_output {skill}/ep{idx:03d}_sa.npz   state(9)/action(8) 原始物理量

split 直接取自实验 4.1 冻结的 `split_manifest.json`（exp4.1-split-v1），不重新划分。
episode ↔ record 的对应关系用采集时写入的 `stored_episode_index`，不用首帧 q 最近邻。

用法（simulate_vla venv）：
    python scripts/exp42/export_data.py --data data/exp41 --out /scratch/lijh/exp42/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def split_of(rec: dict, split: dict) -> str:
    if rec['skill'] == 'A90':
        g = rec['goal_id']
        sp = split['A90']
        if g in sp['train']:
            return 'train'
        if g in sp['val']:
            return 'val'
        return 'test'
    c = rec.get('config_id') or ''
    sp = split['B90']
    if c in sp['train']:
        return 'train'
    if c in sp['val']:
        return 'val'
    return 'test'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/exp41')
    ap.add_argument('--results', default='results/experiment_4_1')
    ap.add_argument('--out', default='/scratch/lijh/exp42/data')
    ap.add_argument('--max-frames', type=int, default=0)
    args = ap.parse_args(argv)
    import pandas as pd

    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    out = Path(args.out)
    res = ROOT / args.results
    split = json.loads((res / 'dataset' / 'split_manifest.json').read_text(encoding='utf-8'))
    recs = {}
    for f in sorted((res / 'dataset' / 'records').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('stored_episode_index') is not None:
            recs[(r['skill'], int(r['stored_episode_index']))] = r
    print(f'records with episode index: {len(recs)}')

    meta = []
    for skill in ('A90', 'B90'):
        root = data / skill
        (out / skill).mkdir(parents=True, exist_ok=True)
        eps = [json.loads(l) for l in (root / 'meta' / 'episodes.jsonl')
               .read_text(encoding='utf-8').splitlines() if l.strip()]
        tasks = {}
        for line in (root / 'meta' / 'tasks.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                t = json.loads(line)
                tasks[t['task_index']] = t['task']
        for ep in eps:
            idx = int(ep['episode_index'])
            rec = recs.get((skill, idx))
            if rec is None:
                print(f'  skip {skill} ep{idx}: no record')
                continue
            parq = root / 'data' / f'chunk-{idx // 1000:03d}' / f'episode_{idx:06d}.parquet'
            df = pd.read_parquet(parq, columns=['observation.state', 'action', 'task_index'])
            S = np.stack(df['observation.state'].to_numpy()).astype(np.float32)
            A = np.stack(df['action'].to_numpy()).astype(np.float32)
            n = S.shape[0] if args.max_frames <= 0 else min(args.max_frames, S.shape[0])
            S, A = S[:n], A[:n]
            np.savez(out / skill / f'ep{idx:03d}_sa.npz', state=S, action=A)
            import imageio.v3 as iio

            for cam, key in (('top', 'observation.images.top'), ('wrist', 'observation.images.wrist')):
                vid = root / 'videos' / f'chunk-{idx // 1000:03d}' / key / f'episode_{idx:06d}.mp4'
                arr = iio.imread(vid, index=None)
                arr = np.asarray(arr, dtype=np.uint8)[:n]
                if arr.shape[0] != n:
                    print(f'  WARN {skill} ep{idx} {cam}: video {arr.shape[0]} vs parquet {n}')
                    n2 = min(arr.shape[0], n)
                    arr = arr[:n2]
                    S, A = S[:n2], A[:n2]
                    np.savez(out / skill / f'ep{idx:03d}_sa.npz', state=S, action=A)
                np.save(out / skill / f'ep{idx:03d}_{cam}.npy', arr)
            meta.append({
                'skill': skill, 'episode_index': idx, 'n_frames': int(S.shape[0]),
                'task': tasks.get(int(df['task_index'].iloc[0])),
                'prompt': rec['prompt'], 'goal_id': rec['goal_id'],
                'config_id': rec.get('config_id'), 'split': split_of(rec, split),
                'attempt_id': rec['attempt_id'],
                'state_path': f'{skill}/ep{idx:03d}_sa.npz',
                'top_path': f'{skill}/ep{idx:03d}_top.npy',
                'wrist_path': f'{skill}/ep{idx:03d}_wrist.npy',
                'goal_point': rec.get('goal_point'),
            })
            print(f'  {skill} ep{idx:3d} frames={S.shape[0]:4d} split={meta[-1]["split"]:5s} '
                  f'goal={rec["goal_id"]}', flush=True)
    (out / 'meta.jsonl').write_text(
        '\n'.join(json.dumps(m, ensure_ascii=False) for m in meta) + '\n', encoding='utf-8')
    summ = {}
    for m in meta:
        key = (m['skill'], m['split'])
        d = summ.setdefault(key, {'episodes': 0, 'frames': 0})
        d['episodes'] += 1
        d['frames'] += m['n_frames']
    print(json.dumps({f'{k[0]}/{k[1]}': v for k, v in summ.items()}, ensure_ascii=False, indent=2))
    (out / 'export_summary.json').write_text(json.dumps(
        {'n_episodes': len(meta),
         'by_skill_split': {f'{k[0]}/{k[1]}': v for k, v in summ.items()},
         'split_version': split.get('version')},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
