"""实验 2.3 阶段 2：汇总 B 专家多起点冒烟结果并画图。

输入 results/experiment_2_3/smoke/episodes.jsonl
输出 analysis.json / success_by_mode_yaw.csv / overview.png

用法：
    python scripts/analyze_experiment_2_3_smoke.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'results' / 'experiment_2_3' / 'smoke'
MODES = ['home', 'home_neighborhood', 'multi_ik', 'task_relevant']
COLORS = {'home': (90, 90, 90), 'home_neighborhood': (70, 130, 180),
          'multi_ik': (60, 160, 90), 'task_relevant': (200, 120, 60)}
SHORT = {'home': 'HOME', 'home_neighborhood': 'HOME-nbhd',
         'multi_ik': 'multi-IK', 'task_relevant': 'task-rel'}
# 实验 2.1 测得的 A0 / A90 真实终态到 HOME 的关节距离（rad）
A_TERMINAL_DIST = {'A0': 2.004, 'A90': 1.650}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def font(size: int):
    for path in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_overview(records: list[dict], path: Path):
    width, height = 1080, 460
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    title = font(20)
    label = font(13)
    small = font(11)

    # ---- 左图：mode × yaw 分组柱状图 ----
    left = (70, 70, 560, 390)
    draw.text((70, 20), 'B expert success by robot start mode (48 episodes)', font=title, fill='black')
    draw.line([left[0], left[3], left[2], left[3]], fill='black', width=1)
    draw.line([left[0], left[1], left[0], left[3]], fill='black', width=1)
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = left[3] - (left[3] - left[1]) * frac
        draw.line([left[0], y, left[2], y], fill=(225, 225, 225), width=1)
        draw.text((30, y - 7), f'{int(frac * 100)}%', font=small, fill='black')
    groups = []
    for mode in MODES:
        for yaw in (0.0, 90.0):
            rows = [r for r in records if r['mode'] == mode and r['yaw0_deg'] == yaw]
            if rows:
                groups.append((mode, yaw, sum(r['success'] for r in rows), len(rows)))
    bar_w = 22
    gap = 8
    x = left[0] + 18
    for mode, yaw, k, n in groups:
        rate = k / n
        y0 = left[3] - (left[3] - left[1]) * rate
        color = COLORS[mode] if yaw == 0.0 else tuple(min(255, c + 60) for c in COLORS[mode])
        draw.rectangle([x, y0, x + bar_w, left[3]], fill=color, outline='black')
        draw.text((x - 2, y0 - 15), f'{k}/{n}', font=small, fill='black')
        x += bar_w + gap
    for i, mode in enumerate(MODES):
        cx = left[0] + 18 + i * 125
        draw.rectangle([cx, 400, cx + 12, 412], fill=COLORS[mode], outline='black')
        draw.text((cx + 16, 399), SHORT[mode], font=small, fill='black')
    draw.text((left[0] + 18, 428), 'dark = object yaw 0 deg, light = yaw 90 deg',
              font=small, fill=(80, 80, 80))

    # ---- 右图：起点到 HOME 的距离 vs 成功 ----
    right = (650, 70, 1040, 390)
    draw.text((650, 20), 'Success vs distance from HOME at start', font=title, fill='black')
    draw.line([right[0], right[3], right[2], right[3]], fill='black', width=1)
    draw.line([right[0], right[1], right[0], right[3]], fill='black', width=1)
    for frac in (0.0, 0.5, 1.0):
        y = right[3] - (right[3] - right[1]) * frac
        draw.text((620, y - 7), {0.0: 'fail', 0.5: '0.5', 1.0: 'ok'}[frac], font=small, fill='black')
    xmax = 2.6
    for name, value in A_TERMINAL_DIST.items():
        x = right[0] + (right[2] - right[0]) * value / xmax
        draw.line([x, right[1], x, right[3]], fill=(200, 60, 60), width=1)
        draw.text((x + 2, right[1] - 2), name, font=small, fill=(200, 60, 60))
    for i, r in enumerate(records):
        x = right[0] + (right[2] - right[0]) * min(r['home_distance_rad'], xmax) / xmax
        yy = right[3] - (right[3] - right[1]) * (1.0 if r['success'] else 0.1)
        yy += ((i % 5) - 2) * 2
        draw.ellipse([x - 4, yy - 4, x + 4, yy + 4], fill=COLORS[r['mode']], outline='black')
    for i, mode in enumerate(MODES):
        cx = 650 + i * 100
        draw.rectangle([cx, 400, cx + 12, 412], fill=COLORS[mode], outline='black')
        draw.text((cx + 16, 399), SHORT[mode], font=small, fill='black')
    draw.text((650, 428), 'x: max-abs joint distance to HOME (rad), red lines = A0/A90 terminals',
              font=small, fill=(80, 80, 80))
    image.save(path)


def main():
    records = [json.loads(line) for line in
               (SMOKE / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    # 兼容旧字段名
    for r in records:
        r.setdefault('yaw0_deg', 0.0 if r.get('yaw_deg', 0) == 0 else 90.0)

    def block(rows):
        k = sum(1 for r in rows if r['success'])
        return {'n': len(rows), 'success': k,
                'rate': round(k / len(rows), 4) if rows else 0.0,
                'ci95': wilson(k, len(rows))}

    by_mode = {m: block([r for r in records if r['mode'] == m]) for m in MODES}
    by_yaw = {label: block([r for r in records if (r['yaw0_deg'] == 0.0) == zero])
              for label, zero in (('long_Y_yaw0', True), ('long_X_yaw90', False))}
    diverse = [r for r in records if r['mode'] != 'home']
    by_mode_yaw = {}
    for m in MODES:
        for yaw in (0.0, 90.0):
            by_mode_yaw[f'{m}@yaw{int(yaw)}'] = block(
                [r for r in records if r['mode'] == m and r['yaw0_deg'] == yaw])

    analysis = {
        'episodes': len(records),
        'overall': block(records),
        'by_mode': by_mode,
        'by_yaw_bin': by_yaw,
        'diverse_only': block(diverse),
        'by_mode_yaw': by_mode_yaw,
        'start_distance_by_mode': {
            m: {
                'mean': round(sum(r['home_distance_rad'] for r in records if r['mode'] == m) /
                              max(1, sum(1 for r in records if r['mode'] == m)), 3),
                'min': round(min(r['home_distance_rad'] for r in records if r['mode'] == m), 3),
                'max': round(max(r['home_distance_rad'] for r in records if r['mode'] == m), 3),
            } for m in MODES
        },
        'max_joint_step_rad': max(r['max_joint_step_rad'] for r in records),
        'max_lift_m': max(r['lift_m'] for r in records),
        'lifted_count': sum(1 for r in records if r['lifted_flag']),
        'mean_ticks': round(sum(r['ticks'] for r in records) / len(records), 1),
        'failures': [{'mode': r['mode'], 'seed': r['seed'], 'reason': r['reason'],
                      'yaw0_deg': r['yaw0_deg'], 'ticks': r['ticks']} for r in records
                     if not r['success']],
        'a_terminal_distance_rad': A_TERMINAL_DIST,
    }
    (SMOKE / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                                         encoding='utf-8')

    rows = []
    for key, value in by_mode_yaw.items():
        rows.append({'group': key, **value})
    for key, value in (('all', block(records)), ('diverse_only', block(diverse))):
        rows.append({'group': key, **value})
    with (SMOKE / 'success_by_mode_yaw.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=['group', 'success', 'n', 'rate', 'ci95'])
        writer.writeheader()
        writer.writerows(rows)

    draw_overview(records, SMOKE / 'overview.png')
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
