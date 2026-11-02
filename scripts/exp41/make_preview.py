"""抽取 4 张可视化样本（A90/B90 × 外部/腕部 × 末帧），存入 dataset/preview。

Git 只保存这些小图；完整视频留在服务器 data/exp41/。

用法：python scripts/exp41/make_preview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp41 import scene41  # noqa: E402
from exp41.run41 import run_attempt  # noqa: E402


def main() -> int:
    from PIL import Image

    out = ROOT / 'results' / 'experiment_4_1' / 'dataset' / 'preview'
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ('A90', scene41.A_GOALS[4], None, None),
        ('B90', next(g for g in scene41.B_GOALS if g.goal_id == 'B_wall_y_mid'),
         next(o for o in scene41.B_OBJECT_STARTS if o['start_id'] == 'B_obj_y021'),
         scene41.B_ROBOT_STARTS[0]),
    ]
    for skill, goal, obj, robot in jobs:
        att = run_attempt(skill, seed=410004 if skill == 'A90' else 421100, goal=goal,
                          obj_start=obj, robot_start=robot, record_frames=True)
        frames = att.frames
        for cam, arr in frames.items():
            n = arr.shape[0]
            for tag, idx in (('first', 0), ('final', n - 1), ('mid', n // 2)):
                Image.fromarray(np.asarray(arr[idx], dtype=np.uint8)).save(
                    out / f'{skill}_{cam}_{tag}.png')
        print(skill, 'frames', {k: v.shape for k, v in frames.items()},
              'success', att.record['strict_success'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
