"""实验 4.6：从 export_v3 里取少量**完整成功 episode** 构造过拟合训练集。

选用（4 条，均通过 exp4.1-strict-v1.2）：
  A90: ep0, ep5      （train split；ep5 在 4.3 闭环里 AE 曾抬起 0.10 m）
  B90: ep0, ep2      （4.4 闭环里推进最多的两条训练初态）
输出：/scratch/lijh/exp46/export_overfit，meta.jsonl 只含这些 episode 的 train 行，
文件用符号链接复用，不复制数据。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

SRC = Path('/scratch/lijh/exp43/export_v3')
DST = Path('/scratch/lijh/exp46/export_overfit')
L = Path('/scratch/lijh/exp40/repo/results/experiment_4_6')
CHOSEN = {('A90', 5), ('A90', 0), ('B90', 0), ('B90', 2)}


def main() -> int:
    rows = [json.loads(l) for l in (SRC / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    keep = [r for r in rows if r.get('split') == 'train'
            and (r['skill'], int(r['episode_index'])) in CHOSEN]
    assert len(keep) == len(CHOSEN), f'expected {len(CHOSEN)} episodes, got {len(keep)}'
    for skill in ('A90', 'B90'):
        (DST / skill).mkdir(parents=True, exist_ok=True)
    for r in keep:
        for key in ('state_path', 'top_path', 'wrist_path'):
            dst = DST / r[key]
            if not dst.exists():
                os.symlink(SRC / r[key], dst)
    (DST / 'meta.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in keep) + '\n', encoding='utf-8')
    L.mkdir(parents=True, exist_ok=True)
    summary = {'root': str(DST), 'n_episodes': len(keep),
               'by_skill': dict(Counter(r['skill'] for r in keep)),
               'episodes': [{'skill': r['skill'], 'episode': r['episode_index'],
                             'config': r.get('config_id'), 'n_frames': r.get('n_frames'),
                             'prompt': r.get('prompt')} for r in keep]}
    (L / 'overfit_subset.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
