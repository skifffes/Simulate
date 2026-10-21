"""修复 LeRobot v2.1 数据集 `meta/episodes_stats.jsonl` 中图像统计量的形状。

早期版本的写入器按 NHWC 保留维度，得到 (1,1,1,C)；
lerobot 0.3.3 的 `aggregate_stats` 要求图像特征的 min/max/mean/std 为 (C,1,1)，
否则加载数据集时会抛 `ValueError: Shape of 'min' must be (3,1,1)`。

用法：
    .conda312\\python.exe scripts\\repair_dataset_stats.py --root data\\lerobot_panda_pick_place
    .conda312\\python.exe scripts\\repair_dataset_stats.py --root data\\lerobot_panda_pick_place --dry-run
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

STATS_KEYS = ('min', 'max', 'mean', 'std')


def _fix_image_stats(stats):
    changed = False
    for feature, values in stats.items():
        if 'images' not in feature:
            continue
        for key in STATS_KEYS:
            array = np.asarray(values[key], dtype=np.float64)
            if array.shape == (1, 1, 1, array.size):
                values[key] = np.squeeze(array, axis=0).transpose(2, 0, 1).tolist()
                changed = True
            elif array.shape == (1, array.size):
                values[key] = array.reshape(array.size, 1, 1).tolist()
                changed = True
            elif array.shape != (3, 1, 1):
                raise ValueError(f'{feature}.{key} 形状异常: {array.shape}')
    return changed


def repair(root, dry_run=False):
    root = Path(root)
    stats_file = root / 'meta' / 'episodes_stats.jsonl'
    if not stats_file.exists():
        raise SystemExit(f'找不到 {stats_file}')
    rows = [json.loads(line) for line in stats_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    changed = sum(_fix_image_stats(row['stats']) for row in rows)
    print(f'{stats_file}: {len(rows)} 条记录，修复 {changed} 条')
    if dry_run or not changed:
        return changed
    backup = stats_file.with_suffix('.jsonl.bak')
    if not backup.exists():
        shutil.copy2(stats_file, backup)
    stats_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                          encoding='utf-8')
    print(f'已写回，备份在 {backup}')
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/lerobot_panda_pick_place')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repair(args.root, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
