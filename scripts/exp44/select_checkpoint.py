"""实验 4.4：按**冻结规则**从 val 统计里选 checkpoint。

规则（与 training_plan.md §5 一致，并遵守"专家动作接近零时不用余弦做主判据"）：
  * A90 抓取/保持阶段专家动作幅度接近 0 → 主判据用 **joint MAE、夹爪开度误差、|Δz|**，
    cos 只作为参考（`A90` 总体 cos 在 motion_high 分层里有意义）。
  * B90 主动推动阶段（motion_high）专家有明确运动 → 主判据用 **cos**、次级 MAE 与 |Δz|。
  * 两个候选 cos 差 < 5% 时取 seed 方差更小者。
输出 checkpoint_selection.md + selection.json。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')
OFF = L / 'offline'


def load(group: str, step: int):
    p = OFF / f'{group}_step{step}_val_stats.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def main() -> int:
    steps = sorted({int(m.group(1)) for p in OFF.glob('*_step*_val_stats.json')
                    if (m := re.search(r'_step(\d+)_val_stats', p.stem))})
    table = {}
    for g in ('A', 'B'):
        for s in steps:
            d = load(g, s)
            if not d:
                continue
            closed = d['strata'].get('A90:gripper_closed') or {}
            hi_b = d['strata'].get('B90:motion_high') or {}
            table.setdefault(g, []).append({
                'step': s,
                'A90_grasp_mae': closed.get('joint_mae_mean'),
                'A90_grasp_grip': closed.get('grip_err_mean'),
                'A90_grasp_dz': closed.get('dz_model_mean_m'),
                'A90_grasp_seedstd': closed.get('seed_mae_std_mean'),
                'A90_overall_cos': (d['by_skill'].get('A90') or {}).get('cos_median'),
                'B90_cos': hi_b.get('cos_median'),
                'B90_mae': hi_b.get('joint_mae_mean'),
                'B90_dz': (d['by_skill'].get('B90') or {}).get('dz_model_mean_m'),
                'B90_seedstd': (d['by_skill'].get('B90') or {}).get('seed_mae_std_mean'),
                'overall_mae': d['overall']['joint_mae_mean'],
            })

    lines = ['# checkpoint 选择（冻结规则，val-only）', '',
             '判据（在看到 test 之前冻结）：', '',
             '- **A90**：抓取/保持阶段专家运动接近 0，**不用 cos 做主判据**；按 joint MAE → '
             '夹爪开度误差 → |Δz| 排序（越小越好），总体 cos 仅作参考。',
             '- **B90**：主动推动阶段（motion_high）专家有明确运动；按 cos 排序，'
             '次级看 MAE 与 |Δz|；cos 差 < 0.05 时取 seed 方差更小者。',
             '']
    out = {}
    for g in ('A', 'B'):
        rows = table.get(g, [])
        if not rows:
            continue
        lines.append(f'## Group {g}')
        lines.append('')
        lines.append('| step | A90 grasp MAE | A90 grip err | A90 Δz | A90 cos(整体) | '
                     'B90 cos | B90 MAE | B90 Δz | B90 seedstd |')
        lines.append('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for r in sorted(rows, key=lambda x: x['step']):
            def f(v, nd=4):
                return '—' if v is None else round(v, nd)
            lines.append(f"| {r['step']} | {f(r['A90_grasp_mae'])} | {f(r['A90_grasp_grip'], 3)} | "
                         f"{f(r['A90_grasp_dz'])} | {f(r['A90_overall_cos'], 3)} | "
                         f"{f(r['B90_cos'], 3)} | {f(r['B90_mae'])} | {f(r['B90_dz'])} | "
                         f"{f(r['B90_seedstd'])} |")
        valid_a = [r for r in rows if r['A90_grasp_mae'] is not None]
        a_pick = min(valid_a, key=lambda r: (r['A90_grasp_mae'],
                                             r['A90_grasp_grip'] or 9,
                                             abs(r['A90_grasp_dz'] or 9))) if valid_a else None
        valid_b = [r for r in rows if r['B90_cos'] is not None]
        b_pick = None
        if valid_b:
            top = max(valid_b, key=lambda r: r['B90_cos'])
            close = [r for r in valid_b if r['B90_cos'] >= top['B90_cos'] - 0.05]
            close.sort(key=lambda r: (r['B90_seedstd'] or 9, r['B90_mae'] or 9))
            b_pick = close[0]
        out[g] = {'A90_checkpoint': a_pick, 'B90_checkpoint': b_pick}
        lines.append('')
        if a_pick:
            lines.append(f"**A90 选 step {a_pick['step']}**：grasp MAE "
                         f"{a_pick['A90_grasp_mae']:.4f}、grip {a_pick['A90_grasp_grip']:.3f}、"
                         f"Δz {a_pick['A90_grasp_dz']:+.4f}、整体 cos "
                         f"{a_pick['A90_overall_cos']:.3f}")
        if b_pick:
            lines.append(f"**B90 选 step {b_pick['step']}**：cos {b_pick['B90_cos']:.3f}、"
                         f"MAE {b_pick['B90_mae']:.4f}、Δz {b_pick['B90_dz']:+.4f}、"
                         f"seedstd {b_pick['B90_seedstd']:.4f}")
        lines.append('')
    (L / 'checkpoint_selection.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (L / 'selection.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                      encoding='utf-8')
    for g, v in out.items():
        a, b = v['A90_checkpoint'], v['B90_checkpoint']
        print(g, 'A90 step', a['step'] if a else None, 'B90 step', b['step'] if b else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
