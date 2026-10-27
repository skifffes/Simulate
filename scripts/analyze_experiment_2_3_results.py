"""实验 2.3：汇总 D0/D1/D2 的原子、未知起点、组合与固定交接结果。

输入（服务器生成后拉到本地）：
    results/experiment_2/eval_atomic/eval_{A0,A90,B,C}.json          # D0 标准原子
    results/experiment_2_3/eval/atomic_std/{D1,D2}/eval_*.json      # D1/D2 标准原子
    results/experiment_2_3/eval/atomic_starts/{D0,D1,D2}/eval.json  # 未知起点
    results/experiment_2_3/eval/composition/{D0,D1,D2}/A*_B.json
    results/experiment_2_3/eval/snapshot/{D0,D1,D2}/snapshot_A*.json

输出：
    results/experiment_2_3/analysis/results.json
    results/experiment_2_3/analysis/*.csv
    results/experiment_2_3/analysis/*.png

用法：
    python scripts/analyze_experiment_2_3_results.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'results' / 'experiment_2_3' / 'eval'
OUT = ROOT / 'results' / 'experiment_2_3' / 'analysis'
TAGS = ['D0', 'D1', 'D2']
COLORS = {'D0': (120, 120, 120), 'D1': (70, 110, 180), 'D2': (60, 160, 90)}
A_TERMINAL = {'A0': 2.004, 'A90': 1.650}


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(rows, key='success'):
    n = len(rows)
    k = sum(1 for r in rows if r.get(key))
    return {'success': k, 'n': n, 'rate': round(k / n, 4) if n else 0.0,
            'ci95': wilson(k, n)}


def mcnemar_exact(b: int, c: int):
    """精确 McNemar：b/c 为两种方法的 discordant 对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def font(size: int):
    for path in ('C:/Windows/Fonts/arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_std(tag: str) -> dict:
    out = {}
    for skill in ('A0', 'A90', 'B', 'C'):
        path = (ROOT / 'results' / 'experiment_2' / 'eval_atomic' / f'eval_{skill}.json'
                if tag == 'D0' else
                EVAL / 'atomic_std' / tag / f'eval_{skill}.json')
        data = load(path)
        if data:
            out[skill] = data['episodes']
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {'atomic_std': {}, 'atomic_starts': {}, 'composition': {}, 'snapshot': {}}

    # ---- 标准原子 ----
    std = {tag: load_std(tag) for tag in TAGS}
    for tag, skills in std.items():
        if not skills:
            continue
        results['atomic_std'][tag] = {}
        for skill, rows in skills.items():
            results['atomic_std'][tag][skill] = rate(rows)
            if skill == 'B':
                groups = {'yaw0': [], 'yaw90': []}
                for r in rows:
                    yaw = abs(float(r.get('scene', {}).get('yaw', 0.0)))
                    yaw = abs(math.atan2(math.sin(yaw), math.cos(yaw)))
                    groups['yaw0' if yaw < math.pi / 4 else 'yaw90'].append(r)
                results['atomic_std'][tag]['B_by_yaw'] = {
                    g: rate(v) for g, v in groups.items()}

    # ---- 未知合法起点 ----
    for tag in TAGS:
        data = load(EVAL / 'atomic_starts' / tag / 'eval.json')
        if data:
            results['atomic_starts'][tag] = data['summary']

    # ---- 连续组合 ----
    for tag in TAGS:
        results['composition'][tag] = {}
        for chain in ('A0_B', 'A90_B'):
            data = load(EVAL / 'composition' / tag / f'{chain}.json')
            if data:
                results['composition'][tag][chain] = data['summary']

    # ---- 固定交接状态 ----
    for tag in TAGS:
        results['snapshot'][tag] = {}
        for a_skill in ('A0', 'A90'):
            data = load(EVAL / 'snapshot' / tag / f'snapshot_{a_skill}.json')
            if data:
                results['snapshot'][tag][a_skill] = data['summary']

    # ---- 配对 McNemar（D1 vs D2，同一 seed/起点）----
    def paired(tag_a, tag_b, loader):
        a = {r['seed']: r for r in loader(tag_a)}
        b = {r['seed']: r for r in loader(tag_b)}
        seeds = sorted(set(a) & set(b))
        only_a = sum(1 for s in seeds if a[s].get('success') and not b[s].get('success'))
        only_b = sum(1 for s in seeds if b[s].get('success') and not a[s].get('success'))
        return {'n_pairs': len(seeds), 'only_D1': only_a, 'only_D2': only_b,
                'mcnemar_p': mcnemar_exact(only_a, only_b)}

    comparisons = {}
    for chain in ('A0_B', 'A90_B'):
        def comp_loader(tag, chain=chain):
            data = load(EVAL / 'composition' / tag / f'{chain}.json')
            return data['episodes'] if data else []
        comparisons[f'composition_{chain}'] = paired('D1', 'D2', comp_loader)
    for a_skill in ('A0', 'A90'):
        def snap_loader(tag, a_skill=a_skill):
            data = load(EVAL / 'snapshot' / tag / f'snapshot_{a_skill}.json')
            return data['episodes'] if data else []
        comparisons[f'snapshot_{a_skill}'] = paired('D1', 'D2', snap_loader)

    def starts_loader(tag):
        data = load(EVAL / 'atomic_starts' / tag / 'eval.json')
        return data['episodes'] if data else []
    comparisons['atomic_starts'] = paired('D1', 'D2', starts_loader)

    def starts_yaw90_loader(tag):
        return [r for r in starts_loader(tag) if r['group'] == 'yaw90']
    comparisons['atomic_starts_yaw90'] = paired('D1', 'D2', starts_yaw90_loader)

    def std_b_loader(tag):
        data = load_std(tag).get('B')
        return data or []
    comparisons['atomic_std_B'] = paired('D1', 'D2', std_b_loader)
    results['paired_comparisons'] = comparisons

    # ---- 失败动力学：A 之后 B 把物体抬高/搬走，而不是推到墙 ----
    dynamics = {}
    for tag in TAGS:
        for chain in ('A0_B', 'A90_B'):
            data = load(EVAL / 'composition' / tag / f'{chain}.json')
            if not data:
                continue
            zs, moved = [], []
            for e in data['episodes']:
                if not e.get('final_object'):
                    continue
                zs.append(e['final_object']['position'][2])
                sw = e.get('switch_object', {}).get('position')
                if sw:
                    moved.append(math.dist(e['final_object']['position'][:2], sw[:2]))
            dynamics[f'composition_{chain}_{tag}'] = {
                'n': len(zs), 'z_mean': round(sum(zs) / max(len(zs), 1), 4),
                'z_max': round(max(zs), 4) if zs else None,
                'xy_moved_mean': round(sum(moved) / max(len(moved), 1), 4),
                'xy_moved_max': round(max(moved), 4) if moved else None,
            }
        for a_skill in ('A0', 'A90'):
            data = load(EVAL / 'snapshot' / tag / f'snapshot_{a_skill}.json')
            if not data:
                continue
            zs, moved = [], []
            for e in data['episodes']:
                if not e.get('final_object'):
                    continue
                zs.append(e['final_object']['position'][2])
                sw = e.get('switch_object', {}).get('position')
                if sw:
                    moved.append(math.dist(e['final_object']['position'][:2], sw[:2]))
            dynamics[f'snapshot_{a_skill}_{tag}'] = {
                'n': len(zs), 'z_mean': round(sum(zs) / max(len(zs), 1), 4),
                'z_max': round(max(zs), 4) if zs else None,
                'xy_moved_mean': round(sum(moved) / max(len(moved), 1), 4),
                'xy_moved_max': round(max(moved), 4) if moved else None,
            }
    results['failure_dynamics'] = dynamics

    # ---- 覆盖 vs 成功（D2 未知起点）----
    d2_starts = starts_loader('D2')
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.6)]
    results['coverage_vs_success'] = {
        f'{lo:.1f}-{hi:.1f}': rate([r for r in d2_starts if lo <= r['home_distance_rad'] < hi])
        for lo, hi in bins}

    (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding='utf-8')

    # ---- CSV 表 ----
    import csv
    with (OUT / 'atomic_std.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['tag', 'group', 'success', 'n', 'rate', 'ci_low', 'ci_high'])
        for tag in TAGS:
            if tag not in results['atomic_std']:
                continue
            for skill in ('A0', 'A90', 'B', 'C'):
                r = results['atomic_std'][tag].get(skill)
                if r:
                    w.writerow([tag, skill, r['success'], r['n'], r['rate'], *r['ci95']])
            for g, r in results['atomic_std'][tag].get('B_by_yaw', {}).items():
                w.writerow([tag, f'B@{g}', r['success'], r['n'], r['rate'], *r['ci95']])

    with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['setting', 'tag', 'success', 'n', 'rate'])
        for tag in TAGS:
            for chain, summary in results['composition'].get(tag, {}).items():
                w.writerow([f'composition_{chain}', tag, summary['chain_success'],
                            summary['n'], summary['chain_rate']])
            for a_skill, summary in results['snapshot'].get(tag, {}).items():
                w.writerow([f'snapshot_{a_skill}', tag, summary['success'],
                            summary['n'], summary['rate']])
            s = results['atomic_starts'].get(tag)
            if s:
                w.writerow(['atomic_starts_all', tag, s['all']['success'],
                            s['all']['n'], s['all']['rate']])
                for g, v in s['by_yaw'].items():
                    w.writerow([f'atomic_starts_{g}', tag, v['success'], v['n'], v['rate']])

    # ---- 图 1：原子 B（HOME 标准 + 未知起点，按 yaw）----
    def bars(draw, box, groups, title):
        x0, y0, x1, y1 = box
        draw.text((x0, y0 - 45), title, font=font(17), fill='black')
        draw.line([x0, y1, x1, y1], fill='black')
        draw.line([x0, y0, x0, y1], fill='black')
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y1 - (y1 - y0) * frac
            draw.line([x0, y, x1, y], fill=(225, 225, 225))
            draw.text((x0 - 34, y - 7), f'{int(frac * 100)}%', font=font(10), fill='black')
        n_groups = len(groups)
        bar_w = 18
        slot = (x1 - x0) / max(n_groups, 1)
        for gi, (label, values) in enumerate(groups):
            bx = x0 + slot * gi + slot * 0.15
            for ti, tag in enumerate(TAGS):
                v = values.get(tag)
                if not v:
                    continue
                x = bx + ti * (bar_w + 4)
                h = (y1 - y0) * v['rate']
                draw.rectangle([x, y1 - h, x + bar_w, y1], fill=COLORS[tag], outline='black')
                draw.text((x - 2, y1 - h - 14), f'{v["success"]}/{v["n"]}',
                          font=font(9), fill='black')
            draw.text((bx, y1 + 8), label, font=font(10), fill='black')
        for ti, tag in enumerate(TAGS):
            lx = x0 + ti * 70
            draw.rectangle([lx, y1 + 40, lx + 10, y1 + 50], fill=COLORS[tag], outline='black')
            draw.text((lx + 14, y1 + 38), tag, font=font(10), fill='black')

    img = Image.new('RGB', (1150, 480), 'white')
    d = ImageDraw.Draw(img)
    groups = [('HOME yaw0', {tag: results['atomic_std'].get(tag, {}).get('B_by_yaw', {})
                             .get('yaw0') for tag in TAGS}),
              ('HOME yaw90', {tag: results['atomic_std'].get(tag, {}).get('B_by_yaw', {})
                              .get('yaw90') for tag in TAGS}),
              ('start yaw0', {tag: results['atomic_starts'].get(tag, {}).get('by_yaw', {})
                              .get('yaw0') for tag in TAGS}),
              ('start yaw90', {tag: results['atomic_starts'].get(tag, {}).get('by_yaw', {})
                               .get('yaw90') for tag in TAGS})]
    bars(d, (80, 90, 520, 360), groups, 'B atomic success (HOME vs unseen starts)')
    img.save(OUT / 'atomic_b.png')

    # ---- 图 2：连续组合与固定交接 ----
    img = Image.new('RGB', (1050, 470), 'white')
    d = ImageDraw.Draw(img)
    groups = [('A0->B', {tag: (results['composition'].get(tag, {})
                               .get('A0_B', {}) and
                               {'success': results['composition'][tag]['A0_B']['chain_success'],
                                'n': results['composition'][tag]['A0_B']['n'],
                                'rate': results['composition'][tag]['A0_B']['chain_rate']})
                         for tag in TAGS}),
              ('A90->B', {tag: (results['composition'].get(tag, {})
                                .get('A90_B', {}) and
                                {'success': results['composition'][tag]['A90_B']['chain_success'],
                                 'n': results['composition'][tag]['A90_B']['n'],
                                 'rate': results['composition'][tag]['A90_B']['chain_rate']})
                          for tag in TAGS}),
              ('snap A0', {tag: results['snapshot'].get(tag, {}).get('A0') for tag in TAGS}),
              ('snap A90', {tag: results['snapshot'].get(tag, {}).get('A90') for tag in TAGS})]
    bars(d, (80, 90, 560, 360), groups, 'A->B hard switch and fixed-handoff B')
    img.save(OUT / 'composition.png')
    print(json.dumps({'atomic_std': results['atomic_std'],
                      'atomic_starts': {k: v['all'] for k, v in results['atomic_starts'].items()},
                      'composition': results['composition'],
                      'snapshot': results['snapshot'],
                      'paired': results['paired_comparisons'],
                      'coverage': results['coverage_vs_success']},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
