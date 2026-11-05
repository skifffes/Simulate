"""为一个原子技能生成"原始分布"评测用的 picks（整段数据按 stride 采样）。

与 `expert_state_picks.py`（少量阶段化状态）不同，这里覆盖**整条 episode 的所有阶段**，
用于回答"模型在该技能的原始数据分布上表现如何"，而不是挑出来的几个状态。

用法：
  python scripts/exp42/make_dist_picks.py --skill B90 --split train --stride 20 \
      --out /tmp/dist_B90_train.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path('/scratch/lijh/exp42/data')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skill', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--max-frames', type=int, default=600)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    rows = [m for m in meta if m['skill'] == args.skill and m['split'] == args.split]
    items = []
    for m in rows:
        import numpy as np

        z = np.load(DATA / m['state_path'])
        T = int(z['state'].shape[0])
        for t in range(0, T, args.stride):
            items.append({'skill': args.skill, 'episode': m['episode_index'],
                          'tick': t, 'frame': t, 'phase': f'dist_{args.split}',
                          'robot_block': 0, 'block_wall': 0})
    if len(items) > args.max_frames:
        step = max(1, len(items) // args.max_frames)
        items = items[::step][:args.max_frames]
    out = {'items': items,
           'note': f'{args.skill}/{args.split} 全阶段采样 stride={args.stride}',
           'n_episodes': len(rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{args.skill}/{args.split}: episodes={len(rows)} frames={len(items)} -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
