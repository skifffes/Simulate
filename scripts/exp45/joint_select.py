"""实验 4.5：**联合** checkpoint 选择（同一模型必须同时具备 A90/B90 能力）。

规则（在看到 test 之前冻结）：
  * A90 用抓取/保持阶段（gripper_closed）的 joint MAE（越小越好）——专家运动接近 0 时不用 cos；
  * B90 用主动推动阶段（motion_high）的方向余弦（越大越好）；
  * 组内各自转成排名（rank），joint_score = rank_A90 + rank_B90（越小越好）；
  * 并列时取 seed 方差更小者；
  * LoRA 组（4.4 Group B）与 Action Expert 组（4.5）用**同一规则**各选一个 checkpoint，
    不允许把两个独立 checkpoint 拼成一个策略。
输出：results/experiment_4_5/checkpoint_selection.md + selection.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

L45 = Path('/scratch/lijh/exp40/repo/results/experiment_4_5')
L44 = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')


def load_set(pattern: str):
    rows = []
    for p in (sorted((L45 / 'offline').glob(pattern))
              + sorted((L44 / 'offline').glob(pattern))):
        m = re.search(r'step(\d+)_val_stats', p.name)
        if not m:
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        closed = d['strata'].get('A90:gripper_closed') or {}
        hi_b = d['strata'].get('B90:motion_high') or {}
        rows.append({
            'file': str(p), 'step': int(m.group(1)),
            'A90_mae': closed.get('joint_mae_mean'),
            'A90_grip': closed.get('grip_err_mean'),
            'A90_dz': closed.get('dz_model_mean_m'),
            'A90_cos_overall': (d['by_skill'].get('A90') or {}).get('cos_median'),
            'B90_cos': hi_b.get('cos_median'),
            'B90_mae': hi_b.get('joint_mae_mean'),
            'B90_dz': (d['by_skill'].get('B90') or {}).get('dz_model_mean_m'),
            'seedstd': d['overall']['seed_mae_std_mean'],
            'overall_mae': d['overall']['joint_mae_mean'],
        })
    return rows


def rank_asc(vals):
    order = sorted(range(len(vals)), key=lambda i: (vals[i] if vals[i] is not None else 1e9))
    ranks = [0] * len(vals)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def pick(rows, label):
    a = [r['A90_mae'] for r in rows]
    b = [-(r['B90_cos'] if r['B90_cos'] is not None else -1) for r in rows]
    ra, rb = rank_asc(a), rank_asc(b)
    for i, r in enumerate(rows):
        r['rank_A90'] = ra[i]
        r['rank_B90'] = rb[i]
        r['joint_score'] = ra[i] + rb[i]
    best = sorted(rows, key=lambda r: (r['joint_score'], r['seedstd']))[0]
    print(f'{label}: joint pick step {best["step"]} '
          f'(A90 MAE {best["A90_mae"]:.4f}, B90 cos {best["B90_cos"]:.3f}, '
          f'joint {best["joint_score"]})')
    return best


def fmt(v, nd=4):
    return '—' if v is None else round(v, nd)


def main() -> int:
    ae = load_set('ae_step*_val_stats.json')
    lora = load_set('B_step*_val_stats.json')
    lines = ['# 实验 4.5 checkpoint 联合选择（val-only，规则先冻结）', '',
             '判据：A90 抓取阶段 joint MAE（专家运动≈0，不用 cos）+ B90 主动推动阶段 cos；'
             '组内排名求和，最小者当选；并列取 seed 方差更小。', '']
    out = {}
    for label, rows in (('Action Expert (4.5)', ae), ('LoRA (4.4 Group B)', lora)):
        if not rows:
            continue
        best = pick(rows, label)
        lines += [f'## {label}', '',
                  '| step | A90 抓取 MAE | A90 grip | A90 Δz | B90 推动 cos | B90 MAE | B90 Δz '
                  '| seed std | joint |', '| ---: | ---: | ---: | ---: | ---: | ---: | ---: '
                  '| ---: | ---: |']
        for r in sorted(rows, key=lambda x: x['step']):
            lines.append(f"| {r['step']} | {fmt(r['A90_mae'])} | {fmt(r['A90_grip'], 3)} | "
                         f"{fmt(r['A90_dz'])} | {fmt(r['B90_cos'], 3)} | {fmt(r['B90_mae'])} | "
                         f"{fmt(r['B90_dz'])} | {fmt(r['seedstd'])} | {r['joint_score']} |")
        lines += ['', f"**选中 step {best['step']}**：A90 MAE {best['A90_mae']:.4f}、"
                      f"grip {best['A90_grip']:.3f}、Δz {best['A90_dz']:+.4f}；"
                      f"B90 cos {best['B90_cos']:.3f}、MAE {best['B90_mae']:.4f}、"
                      f"Δz {best['B90_dz']:+.4f}。", '']
        out[label] = {'step': best['step'], 'metrics': {k: best[k] for k in
                                                       ('A90_mae', 'A90_grip', 'A90_dz',
                                                        'B90_cos', 'B90_mae', 'B90_dz',
                                                        'seedstd', 'joint_score')}}
    (L45 / 'checkpoint_selection.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (L45 / 'selection.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                        encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float)[:800])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
