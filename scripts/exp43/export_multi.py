"""多源 OpenPI 导出：旧 61 条 train + 新 253 条（train_manifest_v2）→ 314 条 train。

格式与 `/scratch/lijh/exp42/data` 完全一致：
  <skill>/ep{idx:03d}_sa.npz   (state (T,9), action (T,8))
  <skill>/ep{idx:03d}_top.npy   (T,256,256,3) uint8
  <skill>/ep{idx:03d}_wrist.npy
  meta.jsonl                  每行一个 episode（skill/episode_index/n_frames/prompt/split/...）

新 episode 使用 **1000+ 的全局唯一 episode_index**，并通过符号链接复用旧的图像/状态文件，
因此不会覆盖旧数据，也不会发生 ep000 冲突。val/test 只复制旧 meta 的对应行。

用法（服务器）：
  python scripts/exp43/export_multi.py --out /scratch/lijh/exp43/export
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRV = Path('/scratch/lijh/exp43')
OLD = Path('/scratch/lijh/exp42/data')
MAN = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion/'
           'train_manifest_v2.json')


def ep_id_to_paths(ep_id: int):
    return (f'ep{ep_id:03d}_sa.npz', f'ep{ep_id:03d}_top.npy', f'ep{ep_id:03d}_wrist.npy')


def decode_videos(paths):
    frames = []
    for v in paths:
        cap = cv2.VideoCapture(str(SRV / v))
        buf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        frames.append(np.asarray(buf, dtype=np.uint8))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/scratch/lijh/exp43/export')
    ap.add_argument('--limit', type=int, default=0, help='>0 时只导出前 N 条新数据（调试）')
    args = ap.parse_args()
    out = Path(args.out)
    for skill in ('A90', 'B90'):
        (out / skill).mkdir(parents=True, exist_ok=True)

    meta_rows = []
    # 1) 旧数据：只复制 train 行，文件用符号链接复用
    n_old = 0
    for line in (OLD / 'meta.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get('split') != 'train':
            continue
        for key in ('state_path', 'top_path', 'wrist_path'):
            dst = out / m[key]
            src = OLD / m[key]
            if not dst.exists():
                os.symlink(src, dst)
        meta_rows.append(m)
        n_old += 1

    # 2) 新数据：写新文件（episode_index 从 1000 起，全局唯一）
    man = json.loads(MAN.read_text(encoding='utf-8'))
    eps = man['episodes']
    if args.limit > 0:
        eps = eps[:args.limit]
    offsets = {'A90': 1000, 'B90': 1000}
    n_new = 0
    for e in eps:
        skill = e['skill']
        ep_id = offsets[skill]
        offsets[skill] += 1
        df = pd.read_parquet(SRV / e['parquet'])
        st = np.stack(df['observation.state'].to_numpy()).astype(np.float32)
        ac = np.stack(df['action'].to_numpy()).astype(np.float32)
        top, wrist = decode_videos(e['videos'])
        n = min(len(st), len(ac), len(top), len(wrist))
        if n < 10:
            print('skip short', e['global_episode_id'], n)
            continue
        sa_name, top_name, wrist_name = ep_id_to_paths(ep_id)
        np.savez_compressed(out / skill / sa_name, state=st[:n], action=ac[:n])
        np.save(out / skill / top_name, top[:n])
        np.save(out / skill / wrist_name, wrist[:n])
        prompt = str(df['task'].iloc[0]) if 'task' in df.columns else ''
        meta_rows.append({
            'skill': skill, 'episode_index': ep_id, 'n_frames': int(n),
            'task': prompt, 'prompt': prompt,
            'goal_id': e.get('goal_id'), 'config_id': e.get('config_id'),
            'split': 'train', 'source': f"exp43:{e['source_tag']}",
            'source_episode_index': e['source_episode_index'], 'seed': e['seed'],
            'global_episode_id': e['global_episode_id'],
            'state_path': f'{skill}/{sa_name}', 'top_path': f'{skill}/{top_name}',
            'wrist_path': f'{skill}/{wrist_name}',
        })
        n_new += 1
        if n_new % 25 == 0:
            print('exported', n_new, 'new episodes', flush=True)

    # 3) 旧 val/test 行也保留（指向旧文件，保证 split 完整）
    for line in (OLD / 'meta.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get('split') == 'train':
            continue
        for key in ('state_path', 'top_path', 'wrist_path'):
            dst = out / m[key]
            src = OLD / m[key]
            if not dst.exists():
                os.symlink(src, dst)
        meta_rows.append(m)

    (out / 'meta.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in meta_rows) + '\n', encoding='utf-8')
    summary = {
        'out': str(out), 'n_old_train': n_old, 'n_new': n_new,
        'n_total': len(meta_rows),
        'by_split': {s: sum(1 for r in meta_rows if r.get('split') == s)
                      for s in ('train', 'val', 'test')},
        'by_skill_train': {s: sum(1 for r in meta_rows
                                   if r.get('split') == 'train' and r['skill'] == s)
                            for s in ('A90', 'B90')},
    }
    (out / 'export_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
