"""实验 2.5：C0-C4 / R0-R1 汇总与图。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
R24 = ROOT / 'results/experiment_2_4'
R25 = ROOT / 'results/experiment_2_5'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except Exception:
        return ImageFont.load_default()


def read_csv(p):
    if not p.exists():
        return []
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def load_eps(base: Path, seeds=None):
    out = []
    for d in sorted(base.glob('seed_*')):
        if not (d / 'summary.json').exists():
            continue
        seed = int(d.name.split('_')[-1])
        if seeds and seed not in seeds:
            continue
        out.append({'dir': d, 'seed': seed,
                    'summary': json.loads((d / 'summary.json').read_text())})
    return out


def enrich(e):
    d = e['dir']
    e['phys'] = read_csv(d / 'physics_trace.csv')
    e['contacts'] = read_csv(d / 'contact_trace.csv')
    e['policy'] = [json.loads(x) for x in (d / 'policy_trace.jsonl').read_text().splitlines() if x.strip()] if (d / 'policy_trace.jsonl').exists() else []
    conn = [r for r in e['phys'] if r['skill'] in ('connect', 'retreat')]
    e['conn_tilt'] = max([float(r['obj_tilt_deg']) for r in conn], default=0.0)
    e['conn_dx'] = max([abs(float(r['obj_dx_m'])) for r in conn], default=0.0)
    e['conn_ticks'] = len(conn)
    e['conn_contacts'] = sum(1 for r in e['contacts']
                             if r['skill'] in ('connect', 'retreat')
                             and ('panda' in r['nameA'] or 'panda' in r['nameB']))
    e['gate_ticks'] = sum(1 for r in e['policy'] if r.get('gripper_gate_active'))
    e['gate_exec_min'] = min([r['executed_gripper_target'] for r in e['policy']
                              if r.get('gripper_gate_active')], default=None)
    e['gate_orig_min'] = min([r['original_gripper_target'] for r in e['policy']
                              if r.get('gripper_gate_active')], default=None)
    return e


def stats(eps):
    n = len(eps)
    return {
        'n': n,
        'success': sum(1 for e in eps if e['summary']['success']),
        'lift_gt_3cm': sum(1 for e in eps if e['summary']['max_lift_m'] > 0.03),
        'max_lift_mean': round(float(np.mean([e['summary']['max_lift_m'] for e in eps])), 4) if eps else 0,
        'both_fingers': sum(1 for e in eps if e['summary'].get('both_fingers')),
        'table_loss': sum(1 for e in eps if e['summary'].get('table_loss')),
        'wall_contact': sum(1 for e in eps if e['summary'].get('wall_contact')),
        'conn_tilt_mean': round(float(np.mean([e['conn_tilt'] for e in eps])), 2) if eps else 0,
        'conn_dx_mean': round(float(np.mean([e['conn_dx'] for e in eps])), 4) if eps else 0,
        'conn_contacts': sum(e['conn_contacts'] for e in eps),
        'conn_ticks_mean': round(float(np.mean([e['conn_ticks'] for e in eps])), 0) if eps else 0,
        'gate_ticks_mean': round(float(np.mean([e['gate_ticks'] for e in eps])), 0) if eps else 0,
    }


def plot_bars(path, labels, series, title, ylabel='fraction'):
    W, H = 1000, 420
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    d.text((50, 20), title, font=font(18), fill='black')
    L, R, T, B = 70, 950, 70, 330
    d.rectangle([L, T, R, B], outline='black')
    for i in range(5):
        y = B - (B - T) * i / 4
        d.line([L, y, R, y], fill=(225, 225, 225))
        d.text((20, y - 7), f'{i * 25}%', font=font(11), fill='black')
    slot = (R - L) / len(labels)
    bw = slot / (len(series) + 0.5)
    for gi, lab in enumerate(labels):
        x0 = L + slot * gi + slot * 0.1
        for si, (name, color, vals) in enumerate(series):
            h = (B - T) * min(1, vals[gi])
            x = x0 + si * bw
            d.rectangle([x, B - h, x + bw * 0.9, B], fill=color, outline='black')
            d.text((x, B - h - 13), f'{vals[gi] * 100:.0f}', font=font(10), fill='black')
        d.text((x0, B + 8), lab, font=font(11), fill='black')
    for si, (name, color, _) in enumerate(series):
        x = L + si * 200
        d.rectangle([x, B + 40, x + 12, B + 52], fill=color, outline='black')
        d.text((x + 16, B + 38), name, font=font(11), fill='black')
    img.save(path)


def plot_line(path, series, title, ylabel):
    W, H = 1000, 420
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    d.text((50, 20), title, font=font(18), fill='black')
    L, R, T, B = 80, 950, 70, 330
    d.rectangle([L, T, R, B], outline='black')
    xs = [x for _, _, x, _ in series]
    ys = [y for _, _, _, y in series]
    xmin, xmax = min(min(v) for v in xs), max(max(v) for v in xs)
    ymin, ymax = min(min(v) for v in ys), max(max(v) for v in ys)
    if ymax == ymin:
        ymax = ymin + 1
    for i in range(5):
        y = B - (B - T) * i / 4
        d.line([L, y, R, y], fill=(225, 225, 225))
        d.text((10, y - 7), f'{ymin + (ymax - ymin) * i / 4:.3f}', font=font(10), fill='black')
    for i in range(5):
        x = L + (R - L) * i / 4
        d.text((x - 10, B + 8), f'{xmin + (xmax - xmin) * i / 4:.1f}', font=font(10), fill='black')
    for name, color, x, y in series:
        pts = [(L + (R - L) * (xi - xmin) / max(1e-9, xmax - xmin),
                B - (B - T) * (yi - ymin) / max(1e-9, ymax - ymin)) for xi, yi in zip(x, y)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=2)
    for i, (name, color, _, _) in enumerate(series):
        y = B + 40 + i * 16
        d.line([L, y, L + 16, y], fill=color, width=3)
        d.text((L + 20, y - 7), name, font=font(10), fill='black')
    img.save(path)


def main():
    seeds = [40000000, 40000001, 40000002]
    E25 = R25
    conds = {}
    conds['C0'] = enrich_all(load_eps(R24 / 'G1_a90b_direct', seeds))
    for c in ('C1', 'C2', 'C3', 'C4'):
        conds[c] = enrich_all(load_eps(E25 / 'G1' / c, seeds))
    r0 = enrich_all(load_eps(R24 / 'G5_a0b_local_retreat', seeds))
    r1 = enrich_all(load_eps(E25 / 'G5' / 'R1', seeds))
    g3 = enrich_all(load_eps(R24 / 'G3_b_home_yaw90'))

    summary = {c: stats(v) for c, v in conds.items()}
    summary['R0'] = stats(r0)
    summary['R1'] = stats(r1)
    summary['G3'] = stats(g3)

    # C1 gate behavior
    c1 = conds.get('C1', [])
    summary['C1_gate'] = {
        'gate_ticks_mean': round(float(np.mean([e['gate_ticks'] for e in c1])), 1) if c1 else 0,
        'exec_min_mean': round(float(np.mean([e['gate_exec_min'] or 0 for e in c1])), 4) if c1 else 0,
        'orig_min_mean': round(float(np.mean([e['gate_orig_min'] or 0 for e in c1])), 4) if c1 else 0,
    }

    (E25 / 'summary.csv').parent.mkdir(parents=True, exist_ok=True)
    keys = ['n', 'success', 'lift_gt_3cm', 'max_lift_mean', 'both_fingers', 'table_loss',
            'wall_contact', 'conn_tilt_mean', 'conn_dx_mean', 'conn_contacts', 'conn_ticks_mean']
    with (E25 / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['condition'] + keys)
        for c, s in summary.items():
            if c in ('C1_gate',):
                continue
            w.writerow([c] + [s.get(k) for k in keys])
    (E25 / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding='utf-8')

    fig = E25 / 'figures'
    fig.mkdir(exist_ok=True)
    labels = ['C0', 'C1', 'C2', 'C3', 'C4']
    plot_bars(fig / 'fig4_interventions.png', labels,
              [('B success', '#2ca02c', [summary[c]['success'] / max(1, summary[c]['n']) for c in labels]),
               ('lift >3cm', '#d62728', [summary[c]['lift_gt_3cm'] / max(1, summary[c]['n']) for c in labels]),
               ('wall contact', '#ff7f0e', [summary[c]['wall_contact'] / max(1, summary[c]['n']) for c in labels])],
              'G1 interventions: B success / lift / wall contact')
    # C1 gripper timeline (seed 40000000)
    e = next((x for x in c1 if x['seed'] == 40000000), None)
    if e and e['phys']:
        t = [float(r['physics_time_sec']) for r in e['phys']]
        plot_line(fig / 'fig1_gripper_timeline.png',
                  [('actual grip', '#1f77b4', t, [float(r['grip_actual']) for r in e['phys']]),
                   ('commanded grip', '#d62728', t, [float(r['grip_target']) for r in e['phys']])],
                  'C1 gripper: actual vs executed command (seed 40000000)', 'width (m)')
        plot_line(fig / 'fig3_ee_object.png',
                  [('EE z', '#1f77b4', t, [float(r['ee_z']) for r in e['phys']]),
                   ('object z', '#d62728', t, [float(r['obj_z']) for r in e['phys']])],
                  'C1 EE vs object Z (seed 40000000)', 'z (m)')
    plot_bars(fig / 'fig5_g5_connectors.png', ['R0', 'R1'],
              [('connector tilt', '#d62728', [(summary['R0']['conn_tilt_mean'] / 90),
                                              (summary['R1']['conn_tilt_mean'] / 90)]),
               ('B success', '#2ca02c', [summary['R0']['success'] / max(1, summary['R0']['n']),
                                          summary['R1']['success'] / max(1, summary['R1']['n'])])],
              'G5 original vs clearance-first retreat')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def enrich_all(eps):
    return [enrich(e) for e in eps]


if __name__ == '__main__':
    main()
