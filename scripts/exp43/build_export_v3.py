"""生成**训练就绪数据根** `export_v3`：meta.jsonl 只含 311 条（v3），文件用符号链接复用。

之后 `train_lora.py` 只需 `EXP42_DATA=/scratch/lijh/exp43/export_v3` 即可，
不会误用被剔除的 3 条旧 B90。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SRC = Path('/scratch/lijh/exp43/export')
L = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion')
DST = Path('/scratch/lijh/exp43/export_v3')


def main() -> int:
    v3 = json.loads((L / 'train_manifest_v3.json').read_text(encoding='utf-8'))
    keep = {(e['skill'], int(e['episode_index'])) for e in v3['episodes']}
    rows = []
    for line in (SRC / 'meta.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get('split') == 'train':
            if (m['skill'], int(m['episode_index'])) not in keep:
                continue
        rows.append(m)
    for skill in ('A90', 'B90'):
        (DST / skill).mkdir(parents=True, exist_ok=True)
    for m in rows:
        for key in ('state_path', 'top_path', 'wrist_path'):
            dst = DST / m[key]
            if not dst.exists():
                os.symlink(SRC / m[key], dst)
    (DST / 'meta.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
    counts = {s: sum(1 for r in rows if r.get('split') == 'train' and r['skill'] == s)
              for s in ('A90', 'B90')}
    summary = {'root': str(DST), 'n_total': len(rows), 'train_by_skill': counts,
               'splits': {s: sum(1 for r in rows if r.get('split') == s)
                          for s in ('train', 'val', 'test')}}
    (L / 'export_v3_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
