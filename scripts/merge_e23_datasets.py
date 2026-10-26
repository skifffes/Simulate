"""实验 2.3：把 A0/A90/C 与新的 B 数据合并成 480 集完整数据集变体。

用法（服务器）：
    PYTHONPATH=src venv/bin/python scripts/merge_e23_datasets.py --variant D1
    PYTHONPATH=src venv/bin/python scripts/merge_e23_datasets.py --variant D2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from vla.atomic_tasks.config2 import FPS, ROBOT_TYPE, feature_dict  # noqa: E402
from vla.lerobot_writer import merge_multi_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['D1', 'D2'], required=True)
    ap.add_argument('--base', type=Path, default=ROOT / 'data')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    base = args.base
    exp2 = base / '_shards_exp2'
    e23 = base / '_shards_e23'
    if args.variant == 'D1':
        shards = [exp2 / 'A0', exp2 / 'A90', e23 / 'B_D1_yaw0', e23 / 'B_D1_yaw90', exp2 / 'C']
    else:
        shards = [exp2 / 'A0', exp2 / 'A90', e23 / 'B_D2', exp2 / 'C']
    missing = [str(p) for p in shards if not (p / 'meta/info.json').exists()]
    if missing:
        raise SystemExit(f'缺少 shard: {missing}')
    out = args.out or (base / f'lerobot_panda_e23_{args.variant}')
    info = merge_multi_task(shards, out, feature_dict(), FPS, ROBOT_TYPE)
    print(json.dumps({'variant': args.variant, 'out': str(out),
                      'episodes': info['total_episodes'], 'frames': info['total_frames'],
                      'tasks': info['total_tasks']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
