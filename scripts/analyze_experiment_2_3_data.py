"""实验 2.3 阶段 3：汇总 D1/D2 数据采集结果、覆盖与预算匹配。

输入（从服务器拉到本地）：
    results/experiment_2_3/data/D1_episodes.jsonl
    results/experiment_2_3/data/D2_episodes.jsonl
    results/experiment_2_3/data/summary_D1.json
    results/experiment_2_3/data/summary_D2.json
    results/experiment_2_3/data/used_specs.json

输出：
    results/experiment_2_3/data/analysis.json
    results/experiment_2_3/data/summary.csv
    results/experiment_2_3/data/coverage.png

用法：
    python scripts/analyze_experiment_2_3_data.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'experiment_2_3' / 'data'
MODES = ['home', 'home_neighborhood', 'multi_ik', 'task_relevant']
COLORS = {'home': (90, 90, 90), 'home_neighborhood': (70, 130, 180),
          'multi_ik': (60, 160, 90), 'task_relevant': (200, 120, 60)}
SHORT = {'home': 'HOME', 'home_neighborhood': 'HOME-nbhd',
         'multi_ik': 'multi-IK', 'task_relevant': 'task-rel'}
A_TERMINAL = {'A0': 2.004, 'A90': 1.650}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def font(size: int):
    for path in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw(records: dict[str, list[dict]], path: Path):
    width, height = 1080, 440
    image = Image.new('RGB', (width, height), 'white')
    d = ImageDraw.Draw(image)
    title, small = font(19), font(11)

    # 左：起点到 HOME 的距离（D1 上条带，D2 下条带）
    left = (80, 70, 540, 340)
    d.text((80, 20), 'Robot start distance to HOME (first frame)', font=title, fill='black')
    d.line([left[0], left[3], left[2], left[3]], fill='black', width=1)
    d.line([left[0], left[1], left[0], left[3]], fill='black', width=1)
    xmax = 2.6
    for value in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        x = left[0] + (left[2] - left[0]) * value / xmax
        d.line([x, left[3], x, left[3] + 4], fill='black', width=1)
        d.text((x - 8, left[3] + 6), f'{value:.1f}', font=small, fill='black')
    d.text((left[0] + 150, left[3] + 20), 'max-abs joint distance to HOME (rad)',
           font=small, fill=(80, 80, 80))
    for name, value in A_TERMINAL.items():
        x = left[0] + (left[2] - left[0]) * value / xmax
        d.line([x, left[1], x, left[3]], fill=(200, 60, 60), width=1)
        d.text((x + 2, left[1] - 2), name, font=small, fill=(200, 60, 60))
    bands = {'D1': (left[1] + 30, left[1] + 110), 'D2': (left[1] + 140, left[3] - 10)}
    for variant, color in (('D1', (110, 110, 110)), ('D2', (60, 160, 90))):
        top, bottom = bands[variant]
        rows = [r for r in records.get(variant, []) if r.get('sampled')]
        d.text((left[0] - 26, top - 4), variant, font=small, fill=color)
        for i, r in enumerate(rows):
            x = left[0] + (left[2] - left[0]) * min(r['home_distance_rad'], xmax) / xmax
            y = top + (bottom - top) * (i % 24) / 23.0
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color, outline='black')
    d.text((left[0], 380), 'D1: 120 episodes from HOME; D2: 60 HOME + 60 diverse starts',
           font=small, fill=(80, 80, 80))
    d.text((left[0], 398), 'red lines = A0 (2.00 rad) / A90 (1.65 rad) real terminal states',
           font=small, fill=(80, 80, 80))

    # 右：物体初始 gap（D1/D2 使用完全相同 spec，分布应重合）
    right = (650, 70, 1040, 340)
    d.text((650, 20), 'Initial front gap to wall (matched specs)', font=title, fill='black')
    d.line([right[0], right[3], right[2], right[3]], fill='black', width=1)
    d.line([right[0], right[1], right[0], right[3]], fill='black', width=1)
    gmin, gmax = 0.16, 0.23
    for value in (0.17, 0.19, 0.21, 0.23):
        x = right[0] + (right[2] - right[0]) * (value - gmin) / (gmax - gmin)
        d.line([x, right[3], x, right[3] + 4], fill='black', width=1)
        d.text((x - 10, right[3] + 6), f'{value:.2f}', font=small, fill='black')
    d.text((right[0] + 130, right[3] + 20), 'initial front gap to wall (m)',
           font=small, fill=(80, 80, 80))
    rows_y = {'yaw0': right[1] + 80, 'yaw90': right[1] + 190}
    for group, y0 in rows_y.items():
        d.text((right[0] - 45, y0 - 5), group, font=small, fill='black')
    for variant, color, offset in (('D1', (110, 110, 110), -7), ('D2', (60, 160, 90), 7)):
        for i, r in enumerate(records.get(variant, [])):
            if not r.get('sampled'):
                continue
            gap = r['spec']['gap_target_m']
            x = right[0] + (right[2] - right[0]) * (gap - gmin) / (gmax - gmin)
            group = 'yaw0' if r['spec']['group'] == 'yaw0' else 'yaw90'
            y = rows_y[group] + offset
            d.ellipse([x - 4, y - 3, x + 4, y + 3], fill=color, outline='black')
    d.text((right[0], 380), 'both variants reuse the same 120 object specs (same seeds)',
           font=small, fill=(80, 80, 80))
    d.text((right[0], 398), 'gap differs by yaw because the block depth differs (2 cm vs 5 cm)',
           font=small, fill=(80, 80, 80))
    image.save(path)


def main():
    d1 = load(DATA / 'D1_episodes.jsonl')
    d2 = load(DATA / 'D2_episodes.jsonl')
    specs = json.loads((DATA / 'used_specs.json').read_text(encoding='utf-8'))
    s1 = json.loads((DATA / 'summary_D1.json').read_text(encoding='utf-8'))
    s2 = json.loads((DATA / 'summary_D2.json').read_text(encoding='utf-8'))

    def attempts(records):
        return [r for r in records if r.get('sampled')]

    def rate(rows):
        k = sum(1 for r in rows if r['success'])
        return {'success': k, 'n': len(rows), 'rate': round(k / len(rows), 4) if rows else 0.0,
                'ci95': wilson(k, len(rows))}

    d1_att = attempts(d1)
    # attempt==99 是多样起点全部失败后的 HOME 回退，不计入常规尝试数
    d2_att = [r for r in attempts(d2) if r['attempt'] != 99]
    d2_fallbacks = [r for r in attempts(d2) if r['attempt'] == 99]
    d1_first = [r for r in d1_att if r['attempt'] == 1]
    d2_first = [r for r in d2_att if r['attempt'] == 1]

    def by_yaw(rows):
        out = {}
        for label, zero in (('long_Y_yaw0', True), ('long_X_yaw90', False)):
            sub = [r for r in rows if (r['spec']['group'] == 'yaw0') == zero]
            out[label] = rate(sub)
        return out

    analysis = {
        'D1': {
            'dataset': s1['dataset'], 'episodes': s1['episodes'], 'frames': s1['frames'],
            'attempts': s1['attempts'], 'elapsed_s': s1['elapsed_s'],
            'attempt_success': rate(d1_att),
            'first_attempt_success': rate(d1_first),
            'by_yaw_attempt': by_yaw(d1_att),
            'mean_ticks': round(sum(r['ticks'] for r in d1_att if r.get('success')) /
                                max(1, sum(1 for r in d1_att if r.get('success'))), 1),
        },
        'D2': {
            'dataset': s2['dataset'], 'episodes': s2['episodes'], 'frames': s2['frames'],
            'attempts': s2['attempts'], 'elapsed_s': s2['elapsed_s'],
            'attempt_success': rate(d2_att),
            'first_attempt_success': rate(d2_first),
            'home_fallbacks': len(d2_fallbacks),
            'by_yaw_attempt': by_yaw(d2_att),
            'by_mode_episodes': {
                m: {'n': sum(1 for r in d2 if r.get('success') and r['mode'] == m),
                    'mean_home_distance': round(sum(r['home_distance_rad'] for r in d2
                                                    if r.get('success') and r['mode'] == m) /
                                                max(1, sum(1 for r in d2
                                                           if r.get('success') and r['mode'] == m)), 3)}
                for m in MODES
            },
            'mean_ticks': round(sum(r['ticks'] for r in d2_att if r.get('success')) /
                                max(1, sum(1 for r in d2_att if r.get('success'))), 1),
        },
        'specs': {
            'n': len(specs),
            'yaw0': sum(1 for s in specs if s['group'] == 'yaw0'),
            'yaw90': sum(1 for s in specs if s['group'] == 'yaw90'),
            'gap_range_yaw0': [round(min(s['gap_target_m'] for s in specs if s['group'] == 'yaw0'), 4),
                               round(max(s['gap_target_m'] for s in specs if s['group'] == 'yaw0'), 4)],
            'gap_range_yaw90': [round(min(s['gap_target_m'] for s in specs if s['group'] == 'yaw90'), 4),
                                round(max(s['gap_target_m'] for s in specs if s['group'] == 'yaw90'), 4)],
            'x_range': [round(min(s['start'][0] for s in specs), 4),
                        round(max(s['start'][0] for s in specs), 4)],
            'y_range': [round(min(s['start'][1] for s in specs), 4),
                        round(max(s['start'][1] for s in specs), 4)],
        },
        'budget': {
            'episodes_equal': s1['episodes'] == s2['episodes'],
            'frames_ratio_D2_over_D1': round(s2['frames'] / max(1, s1['frames']), 4),
        },
        'a_terminal_distance_rad': A_TERMINAL,
    }
    (DATA / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                                        encoding='utf-8')

    rows = []
    for variant, block in (('D1', analysis['D1']), ('D2', analysis['D2'])):
        rows.append({'variant': variant, 'episodes': block['episodes'], 'frames': block['frames'],
                     'attempts': block['attempts'],
                     'attempt_success_rate': block['attempt_success']['rate'],
                     'first_attempt_success_rate': block['first_attempt_success']['rate'],
                     'yaw0_attempt_success_rate': block['by_yaw_attempt']['long_Y_yaw0']['rate'],
                     'yaw90_attempt_success_rate': block['by_yaw_attempt']['long_X_yaw90']['rate'],
                     'mean_ticks': block['mean_ticks']})
    with (DATA / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        import csv
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    draw({'D1': d1, 'D2': d2}, DATA / 'coverage.png')
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
