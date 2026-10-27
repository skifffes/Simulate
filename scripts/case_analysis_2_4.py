"""实验 2.4：为报告生成关键案例与聚合证据（接触 link、抬升前的 EE 运动、Q5）。"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results/experiment_2_4'
GROUPS = ['G0_a0b_direct', 'G1_a90b_direct', 'G2_b_home_yaw0', 'G3_b_home_yaw90',
          'G4_a0b_retreat_home', 'G5_a0b_local_retreat']


def read_csv(path):
    with path.open(encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def load(group):
    out = []
    for ep in sorted((RES / group).glob('seed_*')):
        if not (ep / 'summary.json').exists():
            continue
        out.append({
            'dir': ep, 'summary': json.loads((ep / 'summary.json').read_text()),
            'events': json.loads((ep / 'event_summary.json').read_text()),
            'physics': read_csv(ep / 'physics_trace.csv'),
            'contacts': read_csv(ep / 'contact_trace.csv'),
            'timeline': read_csv(ep / 'success_timeline.csv'),
        })
    return out


def nearest(rows, step):
    return min(rows, key=lambda r: abs(int(r['physics_step']) - step))


def contact_links_at(ep, step):
    links = Counter()
    for r in ep['contacts']:
        if abs(int(r['physics_step']) - step) > 6:
            continue
        for name in (r['nameA'], r['nameB']):
            if 'finger' in name or 'hand' in name or 'link' in name:
                links[name] += 1
    return dict(links)


def fk_z(rows):
    import pybullet as p
    import pybullet_data
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=c)
    out = []
    for r in rows:
        for j in range(7):
            p.resetJointState(robot, j, float(r[f'q_target{j + 1}']), physicsClientId=c)
        out.append(p.getLinkState(robot, 11, computeForwardKinematics=True,
                                  physicsClientId=c)[4][2])
    p.disconnect(c)
    return out


def summarize_ep(ep, with_fk=False):
    s = ep['summary']
    ev = ep['events']['events']
    phys = ep['physics']
    out = {'group': s['group'], 'seed': s['seed'], 'success': s['success'],
           'max_lift': s['max_lift_m'], 'both_fingers': s['both_fingers'],
           'table_loss': s['table_loss'], 'wall_contact': s['wall_contact'],
           'switch': ep['events'].get('switch'), 'events': ev}
    if 'first_contact' in ev:
        out['first_contact_links'] = contact_links_at(ep, ev['first_contact']['physics_step'])
    if 'first_lift' in ev:
        step = ev['first_lift']['physics_step']
        row = nearest(phys, step)
        before = nearest(phys, step - 48)  # 0.2 s
        out['at_lift'] = {
            'ee_z': float(row['ee_z']), 'ee_z_before': float(before['ee_z']),
            'ee_z_delta': float(row['ee_z']) - float(before['ee_z']),
            'obj_x': float(row['obj_x']), 'obj_z': float(row['obj_z']),
            'left': int(row['left_finger_contact']), 'right': int(row['right_finger_contact']),
            'palm': int(row['palm_contact']), 'table': int(row['table_contact']),
            'wall': int(row['wall_contact']), 'grip': float(row['grip_actual']),
        }
        if with_fk:
            zs = fk_z(phys)
            idx = phys.index(row)
            idx_b = phys.index(before)
            out['at_lift']['ee_target_z'] = float(zs[idx])
            out['at_lift']['ee_target_z_before'] = float(zs[idx_b])
            out['at_lift']['ee_target_z_delta'] = float(zs[idx]) - float(zs[idx_b])
    out['final'] = {
        'obj_x': float(phys[-1]['obj_x']), 'obj_z': float(phys[-1]['obj_z']),
        'gap': float(phys[-1]['obj_wall_gap_m']), 'tilt': float(phys[-1]['obj_tilt_deg']),
        'dx': float(phys[-1]['obj_dx_m']), 'dy': float(phys[-1]['obj_dy_m']),
    }
    if ep['timeline']:
        pred = [float(r['success_pred']) for r in ep['timeline']]
        out['success_pred_ever'] = max(pred) > 0
        out['success_pred_first_tick'] = next((int(r['policy_step']) for r in ep['timeline']
                                               if r['success_pred'] == '1'), None)
    return out


def main():
    data = {g: load(g) for g in GROUPS}
    cases = {}
    # 选择案例
    g0 = max(data['G0_a0b_direct'], key=lambda e: e['summary']['max_lift_m'])
    g1 = max(data['G1_a90b_direct'], key=lambda e: e['summary']['max_lift_m'])
    g2 = next(e for e in data['G2_b_home_yaw0'] if e['summary']['seed'] == 40000000)
    g3 = next(e for e in data['G3_b_home_yaw90'] if e['summary']['success'])
    g4 = next(e for e in data['G4_a0b_retreat_home'] if e['summary']['seed'] == 40000000)
    g5 = next(e for e in data['G5_a0b_local_retreat'] if e['summary']['seed'] == 40000002)
    for name, ep in (('G0_worst', g0), ('G1_worst', g1), ('G2_yaw0', g2),
                     ('G3_success', g3), ('G4_retreat', g4), ('G5_local', g5)):
        cases[name] = summarize_ep(ep, with_fk=name.startswith(('G0', 'G1')))

    # 聚合：抬升前 EE 实际/目标 z 变化、首次接触 link、wall-before-lift
    agg = {}
    for g in ('G0_a0b_direct', 'G1_a90b_direct'):
        rows = [summarize_ep(e, with_fk=True) for e in data[g]]
        lifts = [r for r in rows if 'at_lift' in r]
        agg[g] = {
            'n': len(rows), 'success': sum(1 for r in rows if r['success']),
            'episodes_with_lift': len(lifts),
            'ee_z_delta_mean': round(float(np.mean([r['at_lift']['ee_z_delta'] for r in lifts])), 4) if lifts else None,
            'ee_target_z_delta_mean': round(float(np.mean([r['at_lift']['ee_target_z_delta'] for r in lifts])), 4) if lifts else None,
            'both_fingers_at_lift': sum(1 for r in lifts if r['at_lift']['left'] and r['at_lift']['right']),
            'table_contact_at_lift': sum(1 for r in lifts if r['at_lift']['table']),
            'wall_contact_at_lift': sum(1 for r in lifts if r['at_lift']['wall']),
            'first_contact_links': dict(Counter(k for r in rows for k in r.get('first_contact_links', {}))),
        }
    # Q5：达到过成功判据但最终失败
    q5 = []
    for g in GROUPS:
        for e in data[g]:
            r = summarize_ep(e)
            if r.get('success_pred_ever') and not r['success']:
                q5.append({'group': g, 'seed': r['seed'],
                           'first_success_tick': r.get('success_pred_first_tick'),
                           'max_lift': r['max_lift'], 'final': r['final']})
    # wall contact before lift
    wall_before_lift = {}
    for g in GROUPS:
        cnt = 0
        for e in data[g]:
            ev = e['events']['events']
            if 'first_wall_contact' in ev and 'first_lift' in ev:
                if ev['first_wall_contact']['physics_time_sec'] < ev['first_lift']['physics_time_sec']:
                    cnt += 1
        wall_before_lift[g] = cnt

    out = {'cases': cases, 'aggregate': agg, 'q5_success_then_lost': q5,
           'wall_before_lift': wall_before_lift}
    (RES / 'case_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
