"""实验 2.6 汇总：A 中间状态独立启动 / B 夹爪开度变化。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/scratch/lijh/simulate_vla')
BASE = ROOT / 'results/experiment_2_6'
FIG = BASE / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except Exception:
        return ImageFont.load_default()


def last_phys(d):
    p = d / 'physics_trace.csv'
    if not p.exists():
        return {}
    with p.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def first_contact(d):
    p = d / 'contact_trace.csv'
    if not p.exists():
        return None
    with p.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if 'panda' in r.get('nameA', '') or 'panda' in r.get('nameB', ''):
                return {'time': float(r['physics_time_sec']), 'link': r.get('nameA')
                        if 'panda' in r.get('nameA', '') else r.get('nameB')}
    return None


def collect(path):
    out = []
    for cond in ('expert', 'frozen'):
        for d in sorted((path / cond).glob('*')):
            sp = d / 'summary.json'
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            ev = json.loads((d / 'event_summary.json').read_text())
            lp = last_phys(d)
            out.append({
                'condition': cond, 'state': d.name, 'success': bool(s['success']),
                'max_lift_m': s.get('max_lift_m', 0), 'wall_contact': s.get('wall_contact'),
                'first_success_tick': s.get('first_success_tick'),
                'final_gap_m': float(lp.get('obj_wall_gap_m', -1)) if lp else None,
                'final_tilt_deg': float(lp.get('obj_tilt_deg', -1)) if lp else None,
                'pushed_x_m': float(lp.get('obj_dx_m', -1)) if lp else None,
                'first_robot_contact': first_contact(d),
                'gripper_init': ev.get('gripper_init'),
                'source_episode': ev.get('episode'), 'source_frame': ev.get('frame'),
                'group': ev.get('group'), 'mode': ev.get('mode'),
            })
    return out


def bars(path, labels, series, title):
    W, H = 1050, 420
    img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
    d.text((40, 18), title, font=font(18), fill='black')
    L, R, T, B = 70, 980, 70, 330; d.rectangle([L, T, R, B], outline='black')
    mx = max(max(v) for _, _, v in series) * 1.15 or 1
    slot = (R - L) / len(labels); bw = slot / (len(series) + 0.5)
    for i, lab in enumerate(labels):
        x0 = L + i * slot + slot * 0.08
        for j, (name, col, vals) in enumerate(series):
            h = (B - T) * vals[i] / mx; x = x0 + j * bw
            d.rectangle([x, B - h, x + bw * 0.88, B], fill=col, outline='black')
            d.text((x, B - h - 13), f'{vals[i]:.2f}', font=font(9), fill='black')
        d.text((x0, B + 8), lab.replace('_', '\n'), font=font(8), fill='black')
    for j, (name, col, _) in enumerate(series):
        x = L + j * 210; y = B + 45
        d.rectangle([x, y, x + 12, y + 12], fill=col, outline='black')
        d.text((x + 16, y - 2), name, font=font(10), fill='black')
    img.save(path)


def main():
    A = collect(BASE / 'A_midstate_launch')
    B = collect(BASE / 'B_gripper_variation')
    with (BASE / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(A[0].keys()))
        w.writeheader(); w.writerows(A + B)
    # A figure by state
    states = sorted({r['state'] for r in A})
    succ = {c: [] for c in ('expert', 'frozen')}
    lift = {c: [] for c in ('expert', 'frozen')}
    for s in states:
        for c in ('expert', 'frozen'):
            r = next((x for x in A if x['state'] == s and x['condition'] == c), None)
            succ[c].append(1.0 if r and r['success'] else 0.0)
            lift[c].append(float(r['max_lift_m']) if r else 0)
    bars(FIG / 'figA_success.png', states,
         [('expert', '#2ca02c', succ['expert']), ('frozen D2', '#d62728', succ['frozen'])],
         'A: midstate launch success by state')
    bars(FIG / 'figA_lift.png', states,
         [('expert', '#2ca02c', lift['expert']), ('frozen D2', '#d62728', lift['frozen'])],
         'A: max object lift (m)')
    # B figure by width
    labels = sorted({r['state'] for r in B if 'ep066' in r['state']})
    widths = sorted({int(round(x['gripper_init']['requested_m'] * 1000)) for x in B
                     if x.get('gripper_init')})
    labels = [f'{w}mm' for w in widths]
    bs = {c: [] for c in ('expert', 'frozen')}
    for w in widths:
        for c in ('expert', 'frozen'):
            r = next((x for x in B if x['condition'] == c and x['state'].startswith('ep066')
                      and x.get('gripper_init') and
                      int(round(x['gripper_init']['requested_m'] * 1000)) == w), None)
            bs[c].append(1.0 if r and r['success'] else 0.0)
    bars(FIG / 'figB_success.png', labels,
         [('expert', '#2ca02c', bs['expert']), ('frozen D2', '#d62728', bs['frozen'])],
         'B: ep66 f39 success vs initial gripper width')
    out = {'A': A, 'B': B,
           'A_summary': {c: {'n': sum(1 for r in A if r['condition'] == c),
                             'success': sum(1 for r in A if r['condition'] == c and r['success'])}
                         for c in ('expert', 'frozen')},
           'B_summary': {c: {'n': sum(1 for r in B if r['condition'] == c),
                             'success': sum(1 for r in B if r['condition'] == c and r['success'])}
                         for c in ('expert', 'frozen')}}
    (BASE / 'analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out['A_summary'], indent=1))
    print(json.dumps(out['B_summary'], indent=1))


if __name__ == '__main__':
    main()
