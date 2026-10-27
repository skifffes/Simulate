"""实验 2.4：汇总六组 rollout 的接触/抬升/成功统计，生成论文用图与视频索引。

用法（服务器，需要 PIL + PyBullet）：
    PYTHONPATH=src venv/bin/python scripts/analyze_experiment_2_4.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ['G0_a0b_direct', 'G1_a90b_direct', 'G2_b_home_yaw0', 'G3_b_home_yaw90',
          'G4_a0b_retreat_home', 'G5_a0b_local_retreat']
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def load_episodes(root: Path):
    eps = []
    for group in GROUPS:
        gdir = root / group
        if not gdir.exists():
            continue
        for ep in sorted(gdir.glob('seed_*')):
            sp = ep / 'summary.json'
            if not sp.exists():
                continue
            summary = json.loads(sp.read_text(encoding='utf-8'))
            ev = json.loads((ep / 'event_summary.json').read_text(encoding='utf-8'))
            trace = read_csv(ep / 'physics_trace.csv')
            contacts = read_csv(ep / 'contact_trace.csv')
            timeline = read_csv(ep / 'success_timeline.csv')
            eps.append({'group': group, 'dir': ep, 'summary': summary, 'events': ev,
                        'physics': trace, 'contacts': contacts, 'timeline': timeline})
    return eps


def classify(ep):
    s = ep['summary']
    ev = ep['events']['events']
    phys = ep['physics']
    tl = ep['timeline']
    labels = []
    if s['success']:
        return 'success'
    if 'first_contact' not in ev:
        return 'no_contact'
    max_tilt = max((float(r['obj_tilt_deg']) for r in phys), default=0.0)
    final_gap = float(phys[-1]['obj_wall_gap_m']) if phys else 9
    lift = float(s.get('max_lift_m', 0))
    wall = 'first_wall_contact' in ev
    lift_t = ev.get('first_lift', {}).get('physics_time_sec', 1e9)
    wall_t = ev.get('first_wall_contact', {}).get('physics_time_sec', 1e9)
    if any(r.get('success_pred') == '1' for r in tl):
        labels.append('success_then_lost')
    if lift > 0.03 and s.get('both_fingers') and s.get('table_loss'):
        labels.append('grasp_lift')
    if lift > 0.03 and wall and wall_t < lift_t:
        labels.append('wall_induced_lift')
    if max_tilt > 20:
        labels.append('tipping')
    last = tl[-1] if tl else None
    if wall and lift <= 0.03 and last and (abs(float(last['dyaw_deg'])) > 15
                                          or abs(float(last['lateral_dy_m'])) > 0.02):
        labels.append('push_rotated')
    if not wall and final_gap > 0.02:
        labels.append('not_reached')
    if not labels:
        labels.append('unknown')
    # primary priority
    for p in ('success_then_lost', 'wall_induced_lift', 'grasp_lift', 'tipping',
              'push_rotated', 'not_reached', 'unknown'):
        if p in labels:
            return p
    return labels[0]


def line_plot(path, series, title, ylabel, events=None, ymin=None, ymax=None):
    W, H = 1100, 460
    left, right, top, bottom = 80, 1060, 60, 380
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    d.text((left, 20), title, font=font(20), fill='black')
    xs = [x for _, _, x, _ in series]
    ys = [y for _, _, _, y in series]
    xmin, xmax = min(min(v) for v in xs), max(max(v) for v in xs)
    ymin = min(min(v) for v in ys) if ymin is None else ymin
    ymax = max(max(v) for v in ys) if ymax is None else ymax
    if ymax == ymin:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    d.rectangle([left, top, right, bottom], outline='black')
    for frac in np.linspace(0, 1, 5):
        y = bottom - (bottom - top) * frac
        d.line([left, y, right, y], fill=(225, 225, 225))
        d.text((10, y - 7), f'{ymin + (ymax - ymin) * frac:.3f}', font=font(12), fill='black')
    for frac in np.linspace(0, 1, 6):
        x = left + (right - left) * frac
        d.line([x, bottom, x, bottom + 5], fill='black')
        d.text((x - 12, bottom + 8), f'{xmin + (xmax - xmin) * frac:.1f}', font=font(12), fill='black')
    d.text(((left + right) / 2 - 40, bottom + 30), 'time (s)', font=font(13), fill='black')
    d.text((10, top - 25), ylabel, font=font(13), fill='black')
    for label, color, x, y in series:
        pts = []
        for xi, yi in zip(x, y):
            px = left + (right - left) * (xi - xmin) / max(1e-9, xmax - xmin)
            py = bottom - (bottom - top) * (yi - ymin) / max(1e-9, ymax - ymin)
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=color, width=2)
    events = events or {}
    for name, (t, color) in events.items():
        if t < xmin or t > xmax:
            continue
        px = left + (right - left) * (t - xmin) / max(1e-9, xmax - xmin)
        d.line([px, top, px, bottom], fill=color, width=2)
        d.text((px + 2, top + 2), name, font=font(11), fill=color)
    for i, (label, color, _, _) in enumerate(series):
        x = left + 10 + (i % 3) * 260
        y = bottom + 55 + (i // 3) * 18
        d.line([x, y + 7, x + 20, y + 7], fill=color, width=3)
        d.text((x + 25, y), label, font=font(12), fill='black')
    img.save(path)


def fk_target_z(rows, q_cols):
    """离线 FK：用 q_target 还原末端目标 z（不进入仿真控制回路）。"""
    import pybullet as p
    import pybullet_data
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=client)
    out = []
    for r in rows:
        for j in range(7):
            p.resetJointState(robot, j, float(r[q_cols[j]]), physicsClientId=client)
        out.append(p.getLinkState(robot, 11, computeForwardKinematics=True,
                                  physicsClientId=client)[4][2])
    p.disconnect(client)
    return out


def bar_plot(path, labels, series, title, ylabel, total=None):
    W, H = 1100, 460
    left, right, top, bottom = 80, 1060, 60, 380
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    d.text((left, 20), title, font=font(20), fill='black')
    d.rectangle([left, top, right, bottom], outline='black')
    d.line([left, bottom, right, bottom], fill='black')
    for frac in np.linspace(0, 1, 5):
        y = bottom - (bottom - top) * frac
        d.line([left, y, right, y], fill=(225, 225, 225))
        d.text((20, y - 7), f'{frac * 100:.0f}%', font=font(12), fill='black')
    n = len(labels)
    slot = (right - left) / max(n, 1)
    bw = slot / (len(series) + 1)
    for gi, label in enumerate(labels):
        x0 = left + slot * gi + slot * 0.1
        for si, (name, color, values) in enumerate(series):
            v = values[gi]
            x = x0 + si * bw
            h = (bottom - top) * min(1.0, v)
            d.rectangle([x, bottom - h, x + bw * 0.85, bottom], fill=color, outline='black')
            tag = f'{round(v * total)}' if total else f'{v * 100:.0f}'
            d.text((x, bottom - h - 14), tag, font=font(11), fill='black')
        d.text((x0, bottom + 8), label, font=font(11), fill='black')
    for si, (name, color, _) in enumerate(series):
        x = left + 10 + si * 200
        y = bottom + 50
        d.rectangle([x, y, x + 14, y + 14], fill=color, outline='black')
        d.text((x + 18, y), name, font=font(12), fill='black')
    img.save(path)


def timeline_plot(path, events, title):
    W, H = 1100, 420
    left, right, top = 260, 1060, 70
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    d.text((30, 20), title, font=font(19), fill='black')
    rows = sorted(events, key=lambda kv: kv[1])
    tmax = max([t for _, t in rows] + [1.0]) * 1.08
    d.line([left, top - 10, left, top + 26 * len(rows) + 10], fill='black')
    for i, (name, t) in enumerate(rows):
        y = top + i * 26
        d.text((10, y - 8), name, font=font(12), fill='black')
        x = left + (right - left) * t / tmax
        d.line([left, y, x, y], fill=(220, 220, 220), width=2)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill='#d62728')
        d.text((x + 8, y - 8), f'{t:.2f}s', font=font(11), fill='black')
    axis_y = top + 26 * len(rows) + 10
    d.line([left, axis_y, right, axis_y], fill='black')
    for tick in range(0, int(tmax) + 1):
        x = left + (right - left) * tick / tmax
        d.line([x, axis_y, x, axis_y + 5], fill='black')
        d.text((x - 5, axis_y + 8), str(tick), font=font(11), fill='black')
    d.text((right - 30, axis_y + 24), 'time (s)', font=font(12), fill='black')
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'results/experiment_2_4')
    args = ap.parse_args()
    root = args.root
    eps = load_episodes(root)
    figdir = root / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)

    for ep in eps:
        ep['failure_type'] = classify(ep)

    # 分组统计
    group_stats = {}
    for group in GROUPS:
        sub = [e for e in eps if e['group'] == group]
        if not sub:
            continue
        n = len(sub)
        group_stats[group] = {
            'n': n,
            'success': sum(1 for e in sub if e['summary']['success']),
            'a_success': sum(1 for e in sub if e['summary'].get('a_success')),
            'wall_contact': sum(1 for e in sub if e['summary'].get('wall_contact')),
            'both_fingers': sum(1 for e in sub if e['summary'].get('both_fingers')),
            'table_loss': sum(1 for e in sub if e['summary'].get('table_loss')),
            'lift_gt_3cm': sum(1 for e in sub if float(e['summary'].get('max_lift_m', 0)) > 0.03),
            'max_lift_mean': round(float(np.mean([e['summary'].get('max_lift_m', 0) for e in sub])), 4),
            'max_lift_max': round(float(np.max([e['summary'].get('max_lift_m', 0) for e in sub])), 4),
            'failure_types': dict(Counter(e['failure_type'] for e in sub)),
        }

    # 代表性案例
    def pick(group, key, reverse=False):
        sub = [e for e in eps if e['group'] == group]
        return sorted(sub, key=key, reverse=reverse)[0] if sub else None

    g0 = [e for e in eps if e['group'] == 'G0_a0b_direct']
    g1 = [e for e in eps if e['group'] == 'G1_a90b_direct']
    g2 = [e for e in eps if e['group'] == 'G2_b_home_yaw0']
    g3 = [e for e in eps if e['group'] == 'G3_b_home_yaw90']
    g4 = [e for e in eps if e['group'] == 'G4_a0b_retreat_home']
    g5 = [e for e in eps if e['group'] == 'G5_a0b_local_retreat']
    reps = {
        'G0_worst_lift': max(g0, key=lambda e: e['summary'].get('max_lift_m', 0)) if g0 else None,
        'G4_best': max(g4, key=lambda e: (e['summary']['success'], -e['summary'].get('max_lift_m', 1))) if g4 else None,
        'G5_best': max(g5, key=lambda e: (e['summary']['success'], -e['summary'].get('max_lift_m', 1))) if g5 else None,
        'G3_success': next((e for e in g3 if e['summary']['success']), g3[0] if g3 else None),
        'G2_success': next((e for e in g2 if e['summary']['success']), g2[0] if g2 else None),
    }
    rep_info = {}
    for name, ep in reps.items():
        if not ep:
            continue
        rep_info[name] = {'group': ep['group'], 'seed': ep['summary']['seed'],
                          'dir': str(ep['dir']), 'success': ep['summary']['success'],
                          'max_lift': ep['summary'].get('max_lift_m'),
                          'failure': ep['failure_type']}

    # 案例图：G0 worst lift
    ep = reps['G0_worst_lift']
    if ep and ep['physics']:
        phys = ep['physics']
        t = [float(r['physics_time_sec']) for r in phys]
        ev = ep['events']['events']
        events = {}
        if 'first_lift' in ev:
            events['LIFT'] = (ev['first_lift']['physics_time_sec'], '#d62728')
        if 'first_wall_contact' in ev:
            events['WALL'] = (ev['first_wall_contact']['physics_time_sec'], '#ff7f0e')
        sw = ep['events'].get('switch')
        if sw:
            events['SWITCH'] = (sw['physics_time_sec'], '#17becf')
        line_plot(figdir / 'fig1_object_motion.png',
                  [('object X (m)', '#1f77b4', t, [float(r['obj_x']) for r in phys]),
                   ('object Z (m)', '#d62728', t, [float(r['obj_z']) for r in phys])],
                  'Figure 1: Object motion (G0 worst-lift case)', 'position (m)', events)
        q_cols = [f'q_target{i}' for i in range(1, 8)]
        target_z = fk_target_z(phys, q_cols)
        line_plot(figdir / 'fig2_ee_vs_object.png',
                  [('EE actual Z', '#1f77b4', t, [float(r['ee_z']) for r in phys]),
                   ('EE target Z (FK)', '#9467bd', t, target_z),
                   ('object Z', '#d62728', t, [float(r['obj_z']) for r in phys])],
                  'Figure 2: EE vs object vertical motion', 'Z (m)', events)
        line_plot(figdir / 'fig3_contact_forces.png',
                  [('left finger force', '#1f77b4', t, [float(r['left_force']) for r in phys]),
                   ('right finger force', '#d62728', t, [float(r['right_force']) for r in phys]),
                   ('wall force', '#ff7f0e', t, [float(r['wall_force']) for r in phys]),
                   ('total force', '#2ca02c', t, [float(r['total_force']) for r in phys])],
                  'Figure 3: Contact forces (G0 worst-lift case)', 'normal force (N)', events)
        line_plot(figdir / 'fig4_gripper.png',
                  [('grip actual (m)', '#1f77b4', t, [float(r['grip_actual']) for r in phys]),
                   ('grip target (m)', '#9467bd', t, [float(r['grip_target']) for r in phys]),
                   ('L contact', '#2ca02c', t, [float(r['left_finger_contact']) for r in phys]),
                   ('R contact', '#d62728', t, [float(r['right_finger_contact']) for r in phys])],
                  'Figure 4: Gripper state and finger contacts', 'value', events)
        tl = ep['timeline']
        if tl:
            tt = [float(r['physics_time_sec']) for r in tl]
            line_plot(figdir / 'fig6_success_predicate.png',
                      [('gap ok', '#1f77b4', tt, [float(r['cond_gap']) for r in tl]),
                       ('lateral ok', '#2ca02c', tt, [float(r['cond_lateral']) for r in tl]),
                       ('yaw ok', '#ff7f0e', tt, [float(r['cond_yaw']) for r in tl]),
                       ('speed ok', '#9467bd', tt, [float(r['cond_speed']) for r in tl]),
                       ('full predicate', '#d62728', tt, [float(r['success_pred']) for r in tl])],
                      'Figure 6: B success predicate timeline', 'boolean', events, ymin=-0.05, ymax=1.05)
        tl_events = []
        for name, key in (('switch A0->B', 'switch'), ('first contact', 'first_contact'),
                          ('both fingers', 'first_both_fingers'),
                          ('table support lost', 'first_table_loss'),
                          ('lift > 3cm', 'first_lift'), ('max lift', 'max_lift'),
                          ('first wall contact', 'first_wall_contact')):
            if key == 'switch' and ep['events'].get('switch'):
                tl_events.append((name, ep['events']['switch']['physics_time_sec']))
            elif key in ep['events']['events']:
                tl_events.append((name, ep['events']['events'][key]['physics_time_sec']))
        if tl_events:
            maxrow = max(phys, key=lambda r: float(r['b_lift_m']))
            tl_events.append(('max lift', float(maxrow['physics_time_sec'])))
            timeline_plot(figdir / 'fig5_physical_timeline.png', tl_events,
                          'Figure 5: Physical state timeline (G0 worst-lift case)')

    # 失败类型分布
    types = sorted({e['failure_type'] for e in eps})
    counts = [sum(1 for e in eps if e['failure_type'] == t) for t in types]
    bar_plot(figdir / 'fig7_failure_distribution.png',
             [t.replace('_', ' ') for t in types],
             [('episodes', '#1f77b4', [c / max(1, len(eps)) for c in counts])],
             'Figure 7: Failure type distribution (all rollouts)', 'fraction', total=len(eps))

    # 分组成功率 / 抬升率
    disp = {'G0_a0b_direct': 'G0 direct', 'G1_a90b_direct': 'G1 A90B',
            'G2_b_home_yaw0': 'G2 B yaw0', 'G3_b_home_yaw90': 'G3 B yaw90',
            'G4_a0b_retreat_home': 'G4 retreat', 'G5_a0b_local_retreat': 'G5 local'}
    labels = [g for g in GROUPS if g in group_stats]
    bar_plot(figdir / 'fig8_group_rates.png', [disp[g] for g in labels],
             [('success', '#2ca02c', [group_stats[g]['success'] / group_stats[g]['n'] for g in labels]),
              ('lift > 3cm', '#d62728', [group_stats[g]['lift_gt_3cm'] / group_stats[g]['n'] for g in labels]),
              ('wall contact', '#ff7f0e', [group_stats[g]['wall_contact'] / group_stats[g]['n'] for g in labels]),
              ('both fingers', '#9467bd', [group_stats[g]['both_fingers'] / group_stats[g]['n'] for g in labels])],
             'Figure 8: Group success and anomalous-lift rates', 'fraction')

    # summary.csv / video_manifest.csv
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'n', 'success', 'a_success', 'wall_contact', 'both_fingers',
                    'table_loss', 'lift_gt_3cm', 'max_lift_mean', 'max_lift_max'])
        for g in labels:
            s = group_stats[g]
            w.writerow([g, s['n'], s['success'], s['a_success'], s['wall_contact'],
                        s['both_fingers'], s['table_loss'], s['lift_gt_3cm'],
                        s['max_lift_mean'], s['max_lift_max']])

    with (root / 'video_manifest.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'seed', 'success', 'failure_type', 'max_lift_m', 'wall_contact',
                    'both_fingers', 'table_loss', 'top', 'side', 'contact', 'comparison', 'events'])
        for e in eps:
            d = e['dir']
            w.writerow([e['group'], e['summary']['seed'], e['summary']['success'],
                        e['failure_type'], round(float(e['summary'].get('max_lift_m', 0)), 4),
                        e['summary'].get('wall_contact'), e['summary'].get('both_fingers'),
                        e['summary'].get('table_loss'),
                        str(d / 'top.mp4'), str(d / 'side.mp4'), str(d / 'contact.mp4'),
                        str(d / 'comparison.mp4'), str(d / 'events.mp4')])

    analysis = {'groups': group_stats, 'representative': rep_info,
                'n_episodes': len(eps), 'failure_types_all': dict(Counter(e['failure_type'] for e in eps))}
    (root / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    print(json.dumps(analysis, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
