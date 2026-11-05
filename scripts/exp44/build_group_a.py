"""实验 4.4：从 export_v3 构造 Group A（58 条旧合格示范）训练数据根。

- 只取 `source == 'exp41'`（旧数据）且 split=train 的行；
- 旧 B90 的 3 条 v1.2 失败项（ep8/20/29）已在 export_v3 中剔除，因此这里自然是 30+28=58；
- val/test 与 export_v3 完全一致（同一批行，符号链接），保证冻结划分不变。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

SRC = Path('/scratch/lijh/exp43/export_v3')
DST = Path('/scratch/lijh/exp44_export_A58')
L = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')


def main() -> int:
    rows = [json.loads(l) for l in (SRC / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    keep = [r for r in rows
            if r.get('split') != 'train' or r.get('source', 'exp41') == 'exp41']
    for skill in ('A90', 'B90'):
        (DST / skill).mkdir(parents=True, exist_ok=True)
    for r in keep:
        for key in ('state_path', 'top_path', 'wrist_path'):
            dst = DST / r[key]
            if not dst.exists():
                os.symlink(SRC / r[key], dst)
    (DST / 'meta.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in keep) + '\n', encoding='utf-8')
    train = [r for r in keep if r.get('split') == 'train']
    summary = {'root': str(DST),
               'train_by_skill': dict(Counter(r['skill'] for r in train)),
               'n_train': len(train),
               'splits': dict(Counter(r.get('split') for r in keep))}
    (L / 'data_manifest_A58.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
