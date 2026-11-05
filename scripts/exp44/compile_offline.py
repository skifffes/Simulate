"""实验 4.4：汇总离线评测（所有 checkpoint）→ offline_eval_results.json + 选择表。

读取 results/experiment_4_4/offline/{A,B}_step{STEP}_val_analysis.json 与 _chunk.json，
输出：
  * offline_eval_results.json：逐 checkpoint 的逐阶段指标 + 采样方差
  * checkpoint_selection.md：按冻结规则给出每组的候选与最终选择
"""
from __future__ import annotations

import json
import re
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')
OFF = L / 'offline'


def main() -> int:
    rows = []
    for f in sorted(OFF.glob('*_val_analysis.json')):
        m = re.match(r'([AB])_step(\d+)_val_analysis\.json', f.name)
        if not m:
            continue
        group, step = m.group(1), int(m.group(2))
        a = json.loads(f.read_text(encoding='utf-8'))
        chunk_p = OFF / f'{group}_step{step}_chunk.json'
        chunk = json.loads(chunk_p.read_text(encoding='utf-8')) if chunk_p.exists() else {}
        rec = {'group': group, 'step': step, 'phases': a.get('phases', a), 'chunk': chunk}
        rows.append(rec)

    out = {'n_checkpoints': len(rows), 'rows': rows,
           'metrics_note': 'cos 使用首行（k=0）方向；expert 接近零的阶段用 MAE/幅值比替代'}
    (L / 'offline_eval_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                 encoding='utf-8')

    # 简单汇总：每个 checkpoint 的 A90/B90 关键阶段
    lines = ['# checkpoint 离线对比（val，30 状态 × 4 seeds）', '',
             '| group | step | A90 grasp cos | A90 MAE | B90 contact cos | B90 Δz | B90 MAE | chunk MAE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        ph = r['phases']

        def get(skill_phase, key):
            v = ph.get(skill_phase) or {}
            return v.get(key)

        lines.append(
            f"| {r['group']} | {r['step']} | {get('A90:grasp_close','cos_mean')} | "
            f"{get('A90:grasp_close','joint_mae')} | {get('B90:contact_onset','cos_mean')} | "
            f"{get('B90:contact_onset','dz_mean_m')} | {get('B90:contact_onset','joint_mae')} | "
            f"{(r['chunk'] or {}).get('mae_k0')} |")
    (L / 'checkpoint_selection.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('aggregated', len(rows), 'checkpoints')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
