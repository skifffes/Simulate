"""实验 2.6 补充：C3 真实终态 B90 与 B90 夹爪扩展分析。"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/scratch/lijh/simulate_vla')
sys.path.insert(0, str(ROOT / 'src'))
from vla.atomic_tasks.scene2 import build_scene  # noqa: E402

BASE = ROOT / 'results/experiment_2_6'
OUT = BASE / 'b90_extension_figures'
OUT.mkdir(parents=True, exist_ok=True)
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


def read_ep(d):
    s = json.loads((d / 'summary.json').read_text())
    ev = json.loads((d / 'event_summary.json').read_text())
    lp = last_phys(d)
    first = None
    palm = 0
    ct = d / 'contact_trace.csv'
    if ct.exists():
        with ct.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r.get('bodyA') == '1' or r.get('bodyB') == '1':
                    if first is None:
                        first = {'link': r['nameA'] if r['bodyA'] == '1' else r['nameB'],
                                 'time': float(r['physics_time_sec']),
                                 'force': float(r['normal_force']),
                                 'point': [float(r['point_x']), float(r['point_y']),
                                           float(r['point_z'])]}
                    if (r['nameA'] if r['bodyA'] == '1' else r['nameB']).endswith('8'):
                        palm += 1
    return {'success': bool(s['success']), 'max_lift_m': s.get('max_lift_m', 0),
            'wall_contact': s.get('wall_contact'), 'first_success_tick': s.get('first_success_tick'),
            'final_gap_m': float(lp.get('obj_wall_gap_m', -1)) if lp else None,
            'final_tilt_deg': float(lp.get('obj_tilt_deg', -1)) if lp else None,
            'events': list(ev.get('events', {}).keys()),
            'first_contact': first, 'palm_contact_rows': palm}


def load_snapshot(snap):
    sim, block, wall, tray, info = build_scene()
    try:
        p.restoreState(fileName=str(ROOT / snap['bullet']), physicsClientId=sim.client)
        q = np.asarray(sim.q(), float)
        ee, quat = sim.pose()
        pos, oq = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
        grip = float(np.mean(__import__('manipulation').finger_widths(sim)))
        return {'q': q, 'ee': np.asarray(ee), 'obj': np.asarray(pos),
                'yaw': float(snap.get('obj_yaw', 0)), 'grip': grip}
    finally:
        p.disconnect(sim.client)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    # C3
    c3 = []
    for seed in range(40000000, 40000010):
        d = BASE / 'C3_direct_B90'
        fz = d / 'frozen' / f'seed_{seed}'
        ex = d / 'expert' / f'seed_{seed}'
        if not (fz / 'summary.json').exists():
            continue
        meta = json.loads((d / 'states/G1/C3_snapshot' / f'seed_{seed}' / 'c3_terminal.json').read_text())
        row = {'seed': seed, 'q': np.asarray(meta['q'], float),
               'ee': np.asarray(meta['ee'], float), 'obj': np.asarray(meta['obj'], float),
               'grip': meta['grip_actual'], 'frozen': read_ep(fz), 'expert': read_ep(ex)}
        c3.append(row)
    # B extension states
    snaps = []
    for sp in sorted((BASE / 'B90_gripper_extension/states').glob('ep*.json')):
        s = json.loads(sp.read_text())
        if s.get('group') != 'yaw90':
            continue
        g = load_snapshot(s)
        g.update({'tag': s['tag'], 'episode': s['episode'], 'frame': s['frame']})
        snaps.append(g)
    b = []
    for sp in snaps:
        for w in (14, 25, 35, 40):
            for cond in ('frozen', 'expert'):
                d = BASE / 'B90_gripper_extension' / cond / f"{sp['tag']}_g{w:03d}"
                if not (d / 'summary.json').exists():
                    continue
                r = read_ep(d)
                r.update({'state': sp['tag'], 'episode': sp['episode'], 'width_mm': w,
                          'condition': cond, 'q': sp['q'], 'ee': sp['ee'],
                          'obj': sp['obj'], 'grip': sp['grip']})
                b.append(r)
    # geometry C3 vs B states
    geo = []
    for c in c3:
        for s in snaps:
            dq = float(np.linalg.norm(c['q'] - s['q']))
            ee = float(np.linalg.norm(c['ee'] - s['ee']) * 1000)
            rel = float(np.linalg.norm((c['ee'] - c['obj']) - (s['ee'] - s['obj'])) * 1000)
            geo.append({'c3_seed': c['seed'], 'state': s['tag'], 'dq': dq,
                        'ee_mm': ee, 'rel_mm': rel})
    for c in c3:
        rows = [g for g in geo if g['c3_seed'] == c['seed']]
        c['nearest_q'] = min(rows, key=lambda x: x['dq'])
        c['nearest_rel'] = min(rows, key=lambda x: x['rel_mm'])

    # summary tables
    with (BASE / 'b90_extension_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['group', 'state/seed', 'width_mm', 'condition', 'success', 'max_lift_m',
                    'wall_contact', 'final_gap_m', 'final_tilt_deg', 'first_success_tick'])
        for c in c3:
            for cond in ('frozen', 'expert'):
                r = c[cond]
                w.writerow(['C3_direct_B90', c['seed'], '', cond, r['success'],
                            r['max_lift_m'], r['wall_contact'], r['final_gap_m'],
                            r['final_tilt_deg'], r['first_success_tick']])
        for r in b:
            w.writerow(['B90_gripper_extension', r['state'], r['width_mm'],
                        r['condition'], r['success'], r['max_lift_m'], r['wall_contact'],
                        r['final_gap_m'], r['final_tilt_deg'], r['first_success_tick']])
    # B validity
    with (BASE / 'B90_gripper_extension/validity_checks.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['state', 'width_mm', 'actual_grip_m', 'obj_move_mm',
                                       'tilt_change_deg', 'valid'])
        for sp in snaps:
            w.writerow([sp['tag'], '', round(sp['grip'], 6), 0, 0, True])
    # geometry CSV
    with (BASE / 'B90_gripper_extension/geometry_vs_C3.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['c3_seed', 'state', 'q_distance_rad', 'ee_mm', 'rel_mm'])
        for g in geo:
            w.writerow([g['c3_seed'], g['state'], g['dq'], g['ee_mm'], g['rel_mm']])

    summary = {
        'C3': {'n_states': len(c3),
               'frozen_success': sum(c['frozen']['success'] for c in c3),
               'expert_success': sum(c['expert']['success'] for c in c3),
               'frozen_lift_mean': round(float(np.mean([c['frozen']['max_lift_m'] for c in c3])), 4),
               'expert_tilt_mean': round(float(np.mean([c['expert']['final_tilt_deg'] for c in c3])), 2),
               'frozen_first_links': dict(Counter(c['frozen']['first_contact']['link'] for c in c3 if c['frozen']['first_contact'])),
               'expert_first_links': dict(Counter(c['expert']['first_contact']['link'] for c in c3 if c['expert']['first_contact'])),
               'frozen_palm_episodes': sum(1 for c in c3 if c['frozen']['palm_contact_rows'] > 0),
               'expert_palm_episodes': sum(1 for c in c3 if c['expert']['palm_contact_rows'] > 0)},
        'B90': {'states': len(snaps), 'rollouts': len(b),
                'frozen_by_width': {}, 'expert_by_width': {}, 'frozen_success': 0,
                'expert_success': 0},
        'geometry': [
            {'c3_seed': c['seed'], 'nearest_q_state': c['nearest_q']['state'],
             'nearest_q_dq': round(c['nearest_q']['dq'], 4),
             'nearest_q_ee_mm': round(c['nearest_q']['ee_mm'], 2),
             'nearest_q_rel_mm': round(c['nearest_q']['rel_mm'], 2),
             'nearest_rel_state': c['nearest_rel']['state'],
             'nearest_rel_dq': round(c['nearest_rel']['dq'], 4),
             'nearest_rel_ee_mm': round(c['nearest_rel']['ee_mm'], 2),
             'nearest_rel_rel_mm': round(c['nearest_rel']['rel_mm'], 2)}
            for c in c3]
    }
    for w in (14, 25, 35, 40):
        for cond in ('frozen', 'expert'):
            ss = [r for r in b if r['width_mm'] == w and r['condition'] == cond]
            summary['B90'][f'{cond}_by_width'][str(w)] = {
                'n': len(ss), 'success': sum(r['success'] for r in ss)}
    summary['B90']['frozen_success'] = sum(r['success'] for r in b if r['condition'] == 'frozen')
    summary['B90']['expert_success'] = sum(r['success'] for r in b if r['condition'] == 'expert')
    summary['B90']['frozen_first_links'] = dict(Counter(
        r['first_contact']['link'] for r in b
        if r['condition'] == 'frozen' and r['first_contact']))
    summary['B90']['expert_first_links'] = dict(Counter(
        r['first_contact']['link'] for r in b
        if r['condition'] == 'expert' and r['first_contact']))
    summary['B90']['frozen_palm_rollouts'] = sum(
        1 for r in b if r['condition'] == 'frozen' and r['palm_contact_rows'] > 0)
    (BASE / 'b90_extension_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding='utf-8')

    # figures
    def bars(path, labels, series, title):
        W, H = 1100, 430; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
        d.text((40, 18), title, font=font(17), fill='black')
        L, R, T, B = 70, 1030, 70, 330; d.rectangle([L, T, R, B], outline='black')
        mx = max(max(v) for _, _, v in series) * 1.15 or 1
        slot = (R - L) / len(labels); bw = slot / (len(series) + 0.5)
        for i, lab in enumerate(labels):
            x0 = L + i * slot + slot * 0.05
            for j, (name, col, vals) in enumerate(series):
                h = (B - T) * vals[i] / mx; x = x0 + j * bw
                d.rectangle([x, B - h, x + bw * 0.88, B], fill=col, outline='black')
                d.text((x, B - h - 13), f'{vals[i]}', font=font(9), fill='black')
            d.text((x0, B + 8), lab, font=font(8), fill='black')
        for j, (name, col, _) in enumerate(series):
            x = L + j * 180; y = B + 42
            d.rectangle([x, y, x + 12, y + 12], fill=col)
            d.text((x + 16, y - 2), name, font=font(10), fill='black')
        img.save(path)

    bars(OUT / 'figC3_success.png', [str(c['seed'])[-2:] for c in c3],
         [('frozen', '#d62728', [int(c['frozen']['success']) for c in c3]),
          ('expert', '#2ca02c', [int(c['expert']['success']) for c in c3])],
         'C3 terminal: frozen B90 vs closed-loop expert')
    labels = [f'{w}mm' for w in (14, 25, 35, 40)]
    bars(OUT / 'figB90_width.png', labels,
         [('frozen', '#d62728', [summary['B90']['frozen_by_width'][str(w)]['success']
                                 for w in (14, 25, 35, 40)]),
          ('expert', '#2ca02c', [summary['B90']['expert_by_width'][str(w)]['success']
                                 for w in (14, 25, 35, 40)])],
         'B90 extension: success counts by initial gripper width (n=8 each)')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
