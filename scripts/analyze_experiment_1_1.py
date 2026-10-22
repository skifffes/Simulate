"""实验 1.1 汇总分析：把各评测目录的 eval.json 合并成表、算置信区间、做配对检验、出图。

只读 results/experiment_1_1/<group>/eval.json，写出：
    episodes_all.csv        逐 episode 合并表（含组别）
    summary.csv             每组成功率 + Wilson 95% 区间
    paired_tests.csv        同种子配对比较（McNemar 精确检验）
    failures_primary.csv    主要失败原因分布
    summary.md              人读的汇总
    success_rate.png        成功率条形图（带 95% 区间）
    failure_primary.png     失败原因分布图

用法：
    python scripts/analyze_experiment_1_1.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/experiment_1_1'

# (显示名, 目录列表, 类别)  类别用于配色：strict / assisted / expert / ood
GROUPS = [
    ('VLA strict · chunk50 · IID-20', ['strict_chunk50_iid20'], 'strict'),
    ('VLA strict · n10 · IID-20', ['strict_n10_iid20'], 'strict'),
    ('VLA assisted · chunk50 · IID-20', ['assisted_chunk50_iid20'], 'assisted'),
    ('VLA strict · n10 · ss2 · IID-20', ['strict_n10_ss2_iid20'], 'render'),
    ('VLA strict · chunk50 · IID-60', ['strict_chunk50_iid20', 'strict_chunk50_iid40new'], 'strict'),
    ('VLA strict · n10 · IID-60', ['strict_n10_iid20', 'strict_n10_iid40new'], 'strict'),
    ('Expert · IID-20', ['expert_iid20'], 'expert'),
    ('VLA strict · OOD shift-x 3cm', ['ood_shiftx_chunk50'], 'ood'),
    ('VLA strict · OOD yaw 0.25rad', ['ood_yaw025_chunk50'], 'ood'),
    ('VLA strict · OOD yaw 0.5rad (超几何极限)', ['ood_yaw_chunk50'], 'inf'),
    ('VLA strict · OOD friction 0.4', ['ood_friction_chunk50'], 'ood'),
    ('Expert · OOD shift-x 3cm', ['expert_ood_shiftx'], 'expert'),
    ('Expert · OOD yaw 0.25rad', ['expert_ood_yaw025'], 'expert'),
    ('Expert · OOD yaw 0.5rad (超几何极限)', ['expert_ood_yaw'], 'expert'),
    ('Expert · OOD friction 0.4', ['expert_ood_friction'], 'expert'),
]

COLORS = {
    'strict': (52, 110, 199),
    'assisted': (222, 143, 32),
    'render': (128, 92, 176),
    'expert': (60, 150, 90),
    'ood': (190, 70, 70),
    'inf': (120, 120, 120),
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(b: int, c: int) -> float:
    """精确 McNemar：b/c 为两种方法不一致的配对计数，双尾。"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_group(dirs):
    rows, seen = [], set()
    for d in dirs:
        path = BASE / d / 'eval.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for ep in data.get('episodes', []):
            seed = int(ep['seed'])
            if seed in seen:
                continue
            seen.add(seed)
            ep = dict(ep)
            ep['group_dir'] = d
            rows.append(ep)
    rows.sort(key=lambda r: r['seed'])
    return rows


def fmt_pct(k, n):
    return f'{100.0 * k / n:.0f}%' if n else '-'


def classify(ep):
    """按任务级失败原因重新分类。

    1.1 早期版本把“闭爪指令后 20 tick 内无双指接触”当作空抓，但策略常在
    空中先闭爪再下降，导致成功抓取也被误标。这里以任务级 failure 为准，
    只有“没抬起且抬升 <3 cm”才归为空抓。
    """
    if ep.get('success'):
        return 'completed'
    fail = ep.get('failure') or ''
    lift = float(ep.get('lift_height_m') or 0.0)
    if fail == 'object_not_lifted':
        return 'no_grasp_contact' if lift < 0.03 else 'lift_failed'
    if fail == 'object_slipped':
        return 'dropped_in_transport'
    return fail or 'other'


def main():
    summaries, episodes = [], []
    by_name = {}
    for name, dirs, cat in GROUPS:
        rows = load_group(dirs)
        if not rows:
            continue
        by_name[name] = rows
        n = len(rows)
        k = sum(1 for r in rows if r.get('success'))
        lo, hi = wilson(k, n)
        lift = [float(r.get('lift_height_m', 0.0)) for r in rows]
        xy = [float(r.get('place_xy_error_m', float('nan'))) for r in rows]
        xy = [v for v in xy if v == v]
        prim = Counter(classify(r) for r in rows)
        stage = Counter(r.get('failure_stage_event') or ('completed' if r.get('success') else 'unknown')
                        for r in rows)
        summaries.append({
            'group': name,
            'category': cat,
            'episodes': n,
            'success': k,
            'success_rate': round(k / n, 4),
            'wilson_lo': round(lo, 4),
            'wilson_hi': round(hi, 4),
            'autonomous_release': sum(1 for r in rows if r.get('autonomous_release')),
            'mean_lift_m': round(sum(lift) / n, 4),
            'mean_xy_error_m': round(sum(xy) / len(xy), 4) if xy else None,
            'primary_failures': dict(prim),
            'stage_failures': dict(stage),
        })
        for r in rows:
            episodes.append({
                'group': name,
                'seed': r['seed'],
                'success': int(bool(r.get('success'))),
                'failure': r.get('failure') or '',
                'failure_primary': classify(r),
                'failure_stage_event': r.get('failure_stage_event') or '',
                'failure_reasons': r.get('failure_reasons') or '',
                'lift_height_m': r.get('lift_height_m'),
                'place_xy_error_m': r.get('place_xy_error_m'),
                'slip_m': r.get('slip_m'),
                'autonomous_release': int(bool(r.get('autonomous_release'))),
                'close_offset_xy_m': r.get('close_offset_xy_m'),
                'close_dual_contact': r.get('close_dual_contact'),
                'ticks': r.get('ticks'),
                'timeout': int(bool(r.get('timeout'))),
                'saturated_ticks': r.get('saturated_ticks'),
                'counterfactual_assisted_success': int(bool(r.get('counterfactual_assisted_success'))),
                'block_x': (r.get('scene') or {}).get('block_start', [None])[0],
                'block_y': (r.get('scene') or {}).get('block_start', [None, None])[1],
                'block_yaw': (r.get('scene') or {}).get('block_yaw'),
                'block_friction': (r.get('scene') or {}).get('block_friction'),
                'mode': r.get('mode'),
                'supersample': r.get('supersample'),
                'n_action_steps': r.get('n_action_steps'),
            })

    BASE.mkdir(parents=True, exist_ok=True)
    if episodes:
        with (BASE / 'episodes_all.csv').open('w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(episodes[0].keys()))
            w.writeheader()
            w.writerows(episodes)

    with (BASE / 'summary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'episodes', 'success', 'success_rate', 'wilson_lo', 'wilson_hi',
                    'autonomous_release', 'mean_lift_m', 'mean_xy_error_m'])
        for s in summaries:
            w.writerow([s['group'], s['episodes'], s['success'], s['success_rate'],
                        s['wilson_lo'], s['wilson_hi'], s['autonomous_release'],
                        s['mean_lift_m'], s['mean_xy_error_m']])

    with (BASE / 'failures_primary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'failure_primary', 'count'])
        for s in summaries:
            for k, v in sorted(s['primary_failures'].items(), key=lambda kv: -kv[1]):
                w.writerow([s['group'], k, v])

    # ---- 配对检验 -------------------------------------------------------
    pairs = [
        ('strict chunk50 vs strict n10 (IID-20)',
         'VLA strict · chunk50 · IID-20', 'VLA strict · n10 · IID-20'),
        ('strict vs assisted chunk50 (IID-20)',
         'VLA strict · chunk50 · IID-20', 'VLA assisted · chunk50 · IID-20'),
        ('ss1 vs ss2 n10 (IID-20)',
         'VLA strict · n10 · IID-20', 'VLA strict · n10 · ss2 · IID-20'),
    ]
    paired_rows = []
    for label, a_name, b_name in pairs:
        a, b = by_name.get(a_name), by_name.get(b_name)
        if not a or not b:
            continue
        mapa = {r['seed']: bool(r.get('success')) for r in a}
        mapb = {r['seed']: bool(r.get('success')) for r in b}
        seeds = sorted(set(mapa) & set(mapb))
        both = sum(1 for s in seeds if mapa[s] and mapb[s])
        only_a = sum(1 for s in seeds if mapa[s] and not mapb[s])
        only_b = sum(1 for s in seeds if not mapa[s] and mapb[s])
        neither = sum(1 for s in seeds if not mapa[s] and not mapb[s])
        p = mcnemar(only_a, only_b)
        paired_rows.append({
            'comparison': label, 'n_pairs': len(seeds), 'both_success': both,
            'only_a': only_a, 'only_b': only_b, 'both_fail': neither,
            'mcnemar_p': round(p, 4),
        })

    with (BASE / 'paired_tests.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(paired_rows[0].keys()) if paired_rows else
                           ['comparison', 'n_pairs', 'both_success', 'only_a', 'only_b',
                            'both_fail', 'mcnemar_p'])
        w.writeheader()
        w.writerows(paired_rows)

    # ---- 文字汇总 -------------------------------------------------------
    lines = ['# 实验 1.1 汇总（自动生成）', '']
    lines.append('| 组别 | 集数 | 成功 | 成功率 | 95% Wilson | 自主松爪 | 平均抬升/m | 平均放置误差/m |')
    lines.append('|---|---:|---:|---:|---|---:|---:|---:|')
    for s in summaries:
        rel = '—' if s['category'] == 'expert' else f"{s['autonomous_release']}/{s['episodes']}"
        lines.append(
            f"| {s['group']} | {s['episodes']} | {s['success']} | "
            f"{100 * s['success_rate']:.0f}% | "
            f"{100 * s['wilson_lo']:.0f}–{100 * s['wilson_hi']:.0f}% | "
            f"{rel} | {s['mean_lift_m']:.3f} | "
            f"{s['mean_xy_error_m'] if s['mean_xy_error_m'] is not None else '-'} |")
    lines.append('')
    if paired_rows:
        lines.append('## 同种子配对检验（McNemar 精确检验，双尾）')
        lines.append('')
        lines.append('| 比较 | 配对数 | 都成功 | 仅前者 | 仅后者 | 都失败 | p |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for r in paired_rows:
            lines.append(f"| {r['comparison']} | {r['n_pairs']} | {r['both_success']} | "
                         f"{r['only_a']} | {r['only_b']} | {r['both_fail']} | {r['mcnemar_p']} |")
        lines.append('')
    lines.append('## 主要失败原因')
    lines.append('')
    for s in summaries:
        if s['success'] == s['episodes']:
            continue
        items = ', '.join(f'{k}×{v}' for k, v in
                          sorted(s['primary_failures'].items(), key=lambda kv: -kv[1])
                          if k != 'completed')
        lines.append(f"- **{s['group']}**：{items or '无'}")
    (BASE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # ---- 图 -------------------------------------------------------------
    def font(size):
        for path in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
                     '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def bar_chart(path, items, title):
        w, h = 980, 90 + 34 * len(items)
        img = Image.new('RGB', (w, h), 'white')
        d = ImageDraw.Draw(img)
        d.text((16, 12), title, fill=(20, 20, 20), font=font(18))
        left, right = 360, w - 90
        for i, (label, value, lo, hi, color) in enumerate(items):
            y = 60 + i * 34
            d.text((16, y + 4), label, fill=(40, 40, 40), font=font(13))
            d.rectangle([left, y, left + (right - left) * value, y + 18], fill=color)
            if lo is not None:
                x1 = left + (right - left) * lo
                x2 = left + (right - left) * hi
                d.line([x1, y + 9, x2, y + 9], fill=(30, 30, 30), width=2)
                d.line([x1, y + 3, x1, y + 15], fill=(30, 30, 30), width=2)
                d.line([x2, y + 3, x2, y + 15], fill=(30, 30, 30), width=2)
            d.text((right + 8, y + 3), f'{100 * value:.0f}%', fill=(40, 40, 40), font=font(13))
            if i == 0:
                for t in range(0, 101, 25):
                    x = left + (right - left) * t / 100
                    d.line([x, 52, x, h - 8], fill=(225, 225, 225))
        img.save(path)

    items = [(s['group'], s['success_rate'], s['wilson_lo'], s['wilson_hi'],
              COLORS[s['category']]) for s in summaries]
    bar_chart(BASE / 'success_rate.png', items, '实验 1.1 成功率（95% Wilson 区间）')

    prim_items = []
    for s in summaries:
        for k, v in sorted(s['primary_failures'].items(), key=lambda kv: -kv[1]):
            if k in ('completed',):
                continue
            prim_items.append((f"{s['group']} · {k}", v / s['episodes'], None, None,
                               COLORS[s['category']]))
    if prim_items:
        w, h = 980, 90 + 30 * len(prim_items)
        img = Image.new('RGB', (w, h), 'white')
        d = ImageDraw.Draw(img)
        d.text((16, 12), '失败主要原因占该组全部 episode 的比例', fill=(20, 20, 20), font=font(18))
        left, right = 460, w - 80
        for i, (label, value, _, _, color) in enumerate(prim_items):
            y = 58 + i * 30
            d.text((16, y + 3), label, fill=(40, 40, 40), font=font(12))
            d.rectangle([left, y, left + (right - left) * value, y + 16], fill=color)
            d.text((right + 8, y + 2), f'{100 * value:.0f}%', fill=(40, 40, 40), font=font(12))
        img.save(BASE / 'failure_primary.png')

    print(json.dumps({'groups': len(summaries), 'episodes': len(episodes),
                      'paired': paired_rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
