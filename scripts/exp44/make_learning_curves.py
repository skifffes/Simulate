"""实验 4.4：从训练日志与 benchmark 生成 learning_curves.csv 与训练摘要。"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')
LOGS = {'A': Path('/scratch/lijh/exp44/logs_A.txt'),
        'B': Path('/scratch/lijh/exp44/logs_B.txt')}
CK = {'A': Path('/scratch/lijh/exp44/A/ckpts'), 'B': Path('/scratch/lijh/exp44/B/ckpts')}


def main() -> int:
    rows = []
    summary = {}
    for g, f in LOGS.items():
        if not f.exists():
            continue
        txt = f.read_text(errors='ignore')
        steps = re.findall(r'step (\d+)/(\d+) loss=([\d.eE+-]+) grad=([\d.eE+-]+) ([\d.]+) step/s',
                           txt)
        for s, total, loss, grad, sps in steps:
            rows.append({'group': g, 'step': int(s), 'total': int(total),
                         'loss': float(loss), 'grad_norm': float(grad), 'step_per_s': float(sps)})
        bench = CK[g] / 'manifest.json'
        summary[g] = {'n_logged_steps': len(steps),
                      'last_step': int(steps[-1][0]) if steps else None,
                      'last_loss': float(steps[-1][2]) if steps else None,
                      'last_sps': float(steps[-1][4]) if steps else None,
                      'ckpt_steps': sorted(int(p.stem.replace('lora_step', ''))
                                           for p in CK[g].glob('lora_step*.npz')),
                      'opt_steps': sorted(int(p.stem.replace('opt_step', ''))
                                          for p in CK[g].glob('opt_step*.npz'))}
        if bench.exists():
            summary[g]['manifest'] = json.loads(bench.read_text(encoding='utf-8'))
    if rows:
        (L / 'learning_curves.csv').write_text('', encoding='utf-8')
        with open(L / 'learning_curves.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (L / 'training_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    print(json.dumps({g: {k: v for k, v in s.items() if k != 'manifest'}
                      for g, s in summary.items()}, ensure_ascii=False, indent=2))
    print('rows', len(rows), '->', L / 'learning_curves.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
