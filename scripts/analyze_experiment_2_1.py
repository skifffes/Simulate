"""实验 2.1 阶段 3：合并 T0/T1/T2/T3 诊断结果并生成汇总。

读取 results/experiment_2_1/diagnosis/*/diagnosis_*.json，输出：
    summary.md / summary.csv
    conditions.png
    failures.csv（按条件统计失败与有效样本）
"""
from __future__ import annotations

import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/experiment_2_1/diagnosis'
CONDITIONS = ['T0', 'T1', 'T2', 'T3']
LABELS = {
    'T0': 'T0 真实交接',
    'T1': 'T1 只恢复机器人',
    'T2': 'T2 只恢复物体',
    'T3': 'T3 全标准状态',
}


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


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    data = {}
    for path in glob.glob(str(BASE / '**' / 'diagnosis_*.json'), recursive=True):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        data[d['chain']] = d

    rows = []
    lines = ['# 实验 2.1 · 阶段 3：交接状态归因（T0/T1/T2/T3）', '']
    lines.append('| 组合 | 条件 | 有效/总数 | 成功 | 成功率 | 95% CI |')
    lines.append('|---|---|---:|---:|---:|---|')
    for chain in sorted(data):
        d = data[chain]
        for cond in CONDITIONS:
            group = [e for e in d['episodes'] if e['condition'] == cond]
            if not group:
                continue
            valid = [e for e in group if e['valid']]
            ok = sum(1 for e in valid if e['success'])
            lo, hi = wilson(ok, len(valid))
            rate = f'{100 * ok / len(valid):.0f}%' if valid else '—'
            ci = f'{100 * lo:.0f}–{100 * hi:.0f}%' if valid else '—'
            lines.append(f'| {chain} | {cond} | {len(valid)}/{len(group)} | {ok} | {rate} | {ci} |')
            rows.append({'chain': chain, 'condition': cond, 'n': len(group),
                         'valid': len(valid), 'success': ok,
                         'rate': round(ok / len(valid), 4) if valid else None})
        lines.append('')

    # 配对变化：T0 → T1 / T0 → T3
    lines.append('## 配对变化（同快照）')
    lines.append('')
    lines.append('| 组合 | T0→T1 变成功 | T0→T1 变失败 | T0→T3 变成功 | 说明 |')
    lines.append('|---|---:|---:|---:|---|')
    for chain in sorted(data):
        d = data[chain]
        by_seed = {}
        for e in d['episodes']:
            by_seed.setdefault(e['seed'], {})[e['condition']] = e
        t1_gain = sum(1 for s, v in by_seed.items()
                      if v.get('T0') and v.get('T1') and not v['T0']['success'] and v['T1']['success'])
        t1_loss = sum(1 for s, v in by_seed.items()
                      if v.get('T0') and v.get('T1') and v['T0']['success'] and not v['T1']['success'])
        t3_gain = sum(1 for s, v in by_seed.items()
                      if v.get('T0') and v.get('T3') and not v['T0']['success'] and v['T3']['success'])
        lines.append(f'| {chain} | {t1_gain} | {t1_loss} | {t3_gain} | — |')
    lines.append('')
    (BASE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    with (BASE / 'summary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['chain', 'condition', 'n', 'valid', 'success', 'rate'])
        w.writeheader()
        w.writerows(rows)

    # 失败/无效原因
    with (BASE / 'failures.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['chain', 'condition', 'invalid_reason', 'count'])
        for chain in sorted(data):
            for cond in CONDITIONS:
                group = [e for e in data[chain]['episodes'] if e['condition'] == cond]
                bad = [e for e in group if not e['valid']]
                if bad:
                    w.writerow([chain, cond, 'invalid_init', len(bad)])

    # 图
    chains = sorted(data)
    if chains:
        width = 980
        bar_h = 26
        height = 90 + len(chains) * (len(CONDITIONS) * bar_h + 30)
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((16, 12), '实验 2.1：不同交接条件下的后继成功率', fill=(20, 20, 20), font=font(18))
        left, right = 300, width - 90
        colors = {'T0': (190, 70, 70), 'T1': (52, 110, 199), 'T2': (150, 150, 150),
                  'T3': (60, 150, 90)}
        y = 54
        for chain in chains:
            d = data[chain]
            for cond in CONDITIONS:
                group = [e for e in d['episodes'] if e['condition'] == cond]
                valid = [e for e in group if e['valid']]
                ok = sum(1 for e in valid if e['success'])
                label = f'{chain} · {LABELS[cond]}'
                draw.text((16, y + 4), label, fill=(40, 40, 40), font=font(12))
                if valid:
                    rate = ok / len(valid)
                    draw.rectangle([left, y, left + (right - left) * rate, y + 16],
                                   fill=colors[cond])
                    draw.text((right + 8, y + 3), f'{ok}/{len(valid)}', fill=(40, 40, 40),
                              font=font(12))
                else:
                    draw.text((left + 4, y + 3), 'invalid', fill=(120, 120, 120), font=font(12))
                y += bar_h
            y += 8
        img.save(BASE / 'conditions.png')

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
