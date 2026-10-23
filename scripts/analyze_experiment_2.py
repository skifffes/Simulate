"""实验 2 阶段 4/5 汇总：原子成功率、专家对照、组合成功率与配对表。

读取 results/experiment_2/ 下的 eval_atomic / expert_atomic / eval_composition，
输出 summary.md / summary.csv / 成功率图。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/experiment_2'
SKILLS = ['A0', 'A90', 'B', 'C']
CHAINS = ['A0_B', 'A90_B', 'A0_C', 'A90_C']


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load(path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def font(size):
    for p in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    atomic, expert, comp = [], [], []
    for s in SKILLS:
        d = load(BASE / f'eval_atomic/eval_{s}.json')
        if d:
            atomic.append(d['summary'])
        e = load(BASE / f'expert_atomic/eval_{s}.json')
        if e:
            expert.append(e['summary'])
    for c in CHAINS:
        d = load(BASE / f'eval_composition/eval_{c}.json')
        if d:
            comp.append(d['summary'])

    lines = ['# 实验 2：原子技能与组合评测汇总', '']
    lines.append('## 原子技能（严格自主，20 集/技能）')
    lines.append('')
    lines.append('| 技能 | VLA 成功 | VLA 成功率 | 95% CI | 专家 成功 | 专家成功率 |')
    lines.append('|---|---:|---:|---|---:|---:|')
    for a in atomic:
        lo, hi = wilson(a['success'], a['episodes'])
        e = next((x for x in expert if x['skill'] == a['skill']), None)
        lines.append(
            f"| {a['skill']} | {a['success']}/{a['episodes']} | {100 * a['success_rate']:.0f}% | "
            f"{100 * lo:.0f}–{100 * hi:.0f}% | "
            f"{e['success']}/{e['episodes']} | {100 * e['success_rate']:.0f}% |"
            if e else
            f"| {a['skill']} | {a['success']}/{a['episodes']} | {100 * a['success_rate']:.0f}% | "
            f"{100 * lo:.0f}–{100 * hi:.0f}% | – | – |")
    lines.append('')
    lines.append('## 组合（同名种子，A 成功后不重置）')
    lines.append('')
    lines.append('| 组合 | 整链成功 | 整链成功率 | A 成功 | A 成功后后继成功 | 条件成功率 |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for c in comp:
        lines.append(
            f"| {c['chain']} | {c['chain_success']}/{c['episodes']} | {100 * c['chain_rate']:.0f}% | "
            f"{c['A_success']}/{c['episodes']} | {c['conditional_B_success']}/{max(c['A_success'], 1)} | "
            f"{100 * c['conditional_B_rate']:.0f}% |")
    (BASE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    with (BASE / 'summary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['kind', 'name', 'episodes', 'success', 'rate'])
        for a in atomic:
            w.writerow(['atomic', a['skill'], a['episodes'], a['success'], a['success_rate']])
        for e in expert:
            w.writerow(['expert', e['skill'], e['episodes'], e['success'], e['success_rate']])
        for c in comp:
            w.writerow(['chain', c['chain'], c['episodes'], c['chain_success'], c['chain_rate']])

    # 图
    items = []
    for a in atomic:
        e = next((x for x in expert if x['skill'] == a['skill']), None)
        items.append((f'VLA {a["skill"]}', a['success'], a['episodes'], (52, 110, 199)))
        if e:
            items.append((f'专家 {e["skill"]}', e['success'], e['episodes'], (60, 150, 90)))
    for c in comp:
        items.append((f'组合 {c["chain"]}', c['chain_success'], c['episodes'], (190, 70, 70)))

    w_px, h_px = 980, 90 + 30 * len(items)
    img = Image.new('RGB', (w_px, h_px), 'white')
    d = ImageDraw.Draw(img)
    d.text((16, 12), '实验 2：原子技能与组合成功率', fill=(20, 20, 20), font=font(18))
    left, right = 300, w_px - 90
    for i, (label, k, n, color) in enumerate(items):
        y = 56 + i * 30
        rate = k / max(n, 1)
        d.text((16, y + 4), label, fill=(40, 40, 40), font=font(13))
        d.rectangle([left, y, left + (right - left) * rate, y + 16], fill=color)
        d.text((right + 8, y + 3), f'{k}/{n}', fill=(40, 40, 40), font=font(13))
    img.save(BASE / 'success_rates.png')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
