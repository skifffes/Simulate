"""实验 2.5 补充：C3 B 启动状态 vs D2 B 训练起点最近邻。

本地可用数据：
- results/experiment_2_3/data/D2_episodes.jsonl：120 条成功 B 示范的 start_q/start_ee/scene
- results/experiment_2_5/{policy_trace.jsonl,physics_trace.csv}：C3 seed 40000000 的启动状态

D2 全帧数据集在服务器，当前网络不可达；本脚本只做 initial-state NN，并明确标记
trajectory/interior NN 不可用，不构造替代数据。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/experiment_2_5/d2_c3_coverage'
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except Exception:
        return ImageFont.load_default()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_angle_deg(q1, q2):
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    d = abs(float(np.dot(q1, q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def fk_quat_all(qs):
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=c)
    out = []
    for q in qs:
        for j in range(7):
            p.resetJointState(robot, j, float(q[j]), physicsClientId=c)
        out.append(p.getLinkState(robot, 11, computeForwardKinematics=True,
                                  physicsClientId=c)[5])
    p.disconnect(c)
    return out


def load_d2():
    rows = [json.loads(x) for x in
            (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text(
                encoding='utf-8').splitlines() if x.strip()]
    ok = [r for r in rows if r['success'] and r['reason'] == 'ok']
    qs = [np.asarray(r['start_q'], float) for r in ok]
    quats = fk_quat_all(qs)
    for r, quat in zip(ok, quats):
        r['_q'] = np.asarray(r['start_q'], float)
        r['_quat'] = quat
        r['_obj'] = np.asarray(r['scene']['block_start'], float)
        r['_yaw'] = float(r['scene']['block_yaw'])
        r['_gap'] = float(r['scene']['wall_inner_x']) - r['_obj'][0] - (0.05 if abs(r['_yaw']) > 0.8 else 0.02)
        r['_ee'] = np.asarray(r['start_ee'], float)
    return ok


def load_c3():
    pol = [json.loads(x) for x in
           (ROOT / 'results/experiment_2_5/policy_trace.jsonl').read_text(
               encoding='utf-8').splitlines() if x.strip()]
    b = next(r for r in pol if r['skill'] == 'B')
    phys = list(csv.DictReader((ROOT / 'results/experiment_2_5/physics_trace.csv').open(
        encoding='utf-8-sig')))
    pr = min(phys, key=lambda r: abs(int(r['physics_step']) - int(b['physics_step'])))
    q = np.asarray(b['state'][:7], float)
    ee = np.array([float(pr['ee_x']), float(pr['ee_y']), float(pr['ee_z'])])
    quat = np.array([float(pr[f'ee_q{i}']) for i in ('x', 'y', 'z', 'w')])
    obj = np.array([float(pr['obj_x']), float(pr['obj_y']), float(pr['obj_z'])])
    yaw = float(pr['obj_yaw_rad'])
    gap = float(pr['obj_wall_gap_m'])
    return {
        'seed': 40000000, 'policy_step': b['policy_step'], 'physics_step': b['physics_step'],
        'time': float(pr['physics_time_sec']), 'q': q, 'ee': ee, 'quat': quat,
        'obj': obj, 'yaw': yaw, 'gap': gap, 'grip': float(pr['grip_actual']),
        'task': b['task'],
    }


def main():
    d2 = load_d2()
    c3 = load_c3()
    D = []
    for r in d2:
        dq = float(np.linalg.norm(c3['q'] - r['_q']))
        ee_mm = float(np.linalg.norm(c3['ee'] - r['_ee']) * 1000)
        ori = quat_angle_deg(c3['quat'], r['_quat'])
        rel_c3 = c3['ee'] - c3['obj']
        rel_tr = r['_ee'] - r['_obj']
        rel_mm = float(np.linalg.norm(rel_c3 - rel_tr) * 1000)
        yaw_deg = abs(math.degrees(wrap(c3['yaw'] - r['_yaw'])))
        yaw_sym = min(yaw_deg, 180.0 - yaw_deg)
        gap_mm = abs(c3['gap'] - r['_gap']) * 1000
        D.append({**r, 'dq': dq, 'ee_mm': ee_mm, 'ori_deg': ori,
                  'rel_mm': rel_mm, 'yaw_deg': yaw_sym, 'gap_mm': gap_mm})
    D.sort(key=lambda x: x['dq'])
    by_rel = sorted(D, key=lambda x: x['rel_mm'])
    by_ee = sorted(D, key=lambda x: x['ee_mm'])
    yaw90 = [x for x in D if x['spec']['group'] == 'yaw90']
    yaw0 = [x for x in D if x['spec']['group'] == 'yaw0']

    # 校准：同一组训练起点内部的最近邻（排除自身）
    def ref_nn(sub):
        vals = []
        for i, a in enumerate(sub):
            ds = [np.linalg.norm(a['_q'] - b['_q']) for j, b in enumerate(sub) if j != i]
            vals.append(min(ds))
        return np.asarray(vals)

    ref_all = ref_nn(D)
    ref_y90 = ref_nn(yaw90)

    # 输出 query
    with (OUT / 'c3_query_states.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'policy_step', 'physics_step', 'time_s', 'q1..q7',
                    'dq1..dq7', 'grip_actual', 'ee_x', 'ee_y', 'ee_z',
                    'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'obj_x', 'obj_y', 'obj_z',
                    'obj_yaw_deg', 'obj_gap_m', 'task', 'status'])
        w.writerow([c3['seed'], c3['policy_step'], c3['physics_step'], c3['time']]
                   + [round(v, 6) for v in c3['q']] + ['', round(c3['grip'], 5)]
                   + [round(v, 6) for v in c3['ee']] + [round(v, 6) for v in c3['quat']]
                   + [round(v, 6) for v in c3['obj']] + [round(math.degrees(c3['yaw']), 3),
                      round(c3['gap'], 5), c3['task'], 'available'])
    # manifest
    with (OUT / 'd2_b_state_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['episode_index', 'attempt', 'seed', 'group', 'mode', 'start_type',
                    'q1..q7', 'ee_x', 'ee_y', 'ee_z', 'obj_x', 'obj_y', 'obj_z',
                    'obj_yaw_deg', 'wall_gap_m', 'home_distance_rad'])
        for r in sorted(d2, key=lambda x: x['index']):
            w.writerow([r['index'], r['attempt'], r['spec']['seed'], r['spec']['group'],
                        r['mode'], 'HOME' if r['mode'] == 'home' else 'diverse']
                       + [round(float(v), 6) for v in r['_q']]
                       + [round(float(v), 6) for v in r['_ee']]
                       + [round(float(v), 6) for v in r['_obj']]
                       + [round(math.degrees(r['_yaw']), 3), round(r['_gap'], 5),
                          r['home_distance_rad']])

    # initial NN top20 all / yaw90
    def write_nn(path, sub):
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'episode_index', 'seed', 'group', 'start_type', 'mode',
                        'dq', 'ee_mm', 'ori_deg', 'rel_mm', 'yaw_deg', 'gap_mm',
                        'home_distance_rad'])
            for i, r in enumerate(sub[:20], 1):
                w.writerow([i, r['index'], r['spec']['seed'], r['spec']['group'],
                            'HOME' if r['mode'] == 'home' else 'diverse', r['mode'],
                            round(r['dq'], 6), round(r['ee_mm'], 2), round(r['ori_deg'], 2),
                            round(r['rel_mm'], 2), round(r['yaw_deg'], 2),
                            round(r['gap_mm'], 2), r['home_distance_rad']])

    write_nn(OUT / 'initial_nn.csv', D)
    write_nn(OUT / 'initial_nn_yaw90.csv', yaw90)
    write_nn(OUT / 'initial_nn_by_relative_geometry.csv', by_rel)
    write_nn(OUT / 'initial_nn_by_ee_distance.csv', by_ee)

    # reference distribution
    with (OUT / 'reference_distance_distribution.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'stat', 'value_rad'])
        for name, arr in (('all_120', ref_all), ('yaw90_60', ref_y90)):
            for stat, val in (('min', arr.min()), ('p25', np.percentile(arr, 25)),
                              ('median', np.median(arr)), ('p75', np.percentile(arr, 75)),
                              ('p95', np.percentile(arr, 95)), ('max', arr.max())):
                w.writerow([name, stat, round(float(val), 6)])

    summary = {
        'status': 'partial_initial_state_only',
        'blocked': 'D2 全帧训练数据集位于服务器 data/lerobot_panda_e23_D2；本机网络不可达，'
                   '无法读取 interior/all frames，故未计算 trajectory NN。',
        'd2_episodes': len(d2), 'd2_yaw0': len(yaw0), 'd2_yaw90': len(yaw90),
        'd2_home': sum(1 for r in d2 if r['mode'] == 'home'),
        'd2_diverse': sum(1 for r in d2 if r['mode'] != 'home'),
        'c3_query': {'seed': c3['seed'], 'q': [round(float(v), 6) for v in c3['q']],
                     'ee': [round(float(v), 6) for v in c3['ee']],
                     'obj': [round(float(v), 6) for v in c3['obj']],
                     'obj_yaw_deg': round(math.degrees(c3['yaw']), 3),
                     'gap_m': round(c3['gap'], 5), 'grip': round(c3['grip'], 5),
                     'policy_step': c3['policy_step'], 'physics_step': c3['physics_step']},
        'initial_nn_all': {k: round(float(D[0][k]), 6) for k in
                           ('dq', 'ee_mm', 'ori_deg', 'rel_mm', 'yaw_deg', 'gap_mm')},
        'initial_nn_yaw90': {k: round(float(yaw90[0][k]), 6) for k in
                             ('dq', 'ee_mm', 'ori_deg', 'rel_mm', 'yaw_deg', 'gap_mm')},
        'initial_nn_all_episode': D[0]['index'],
        'initial_nn_yaw90_episode': yaw90[0]['index'],
        'initial_nn_all_joint_abs': [round(float(v), 6) for v in np.abs(c3['q'] - D[0]['_q'])],
        'initial_nn_yaw90_joint_abs': [round(float(v), 6) for v in np.abs(c3['q'] - yaw90[0]['_q'])],
        'nearest_by_relative_geometry': {
            'episode': by_rel[0]['index'], 'group': by_rel[0]['spec']['group'],
            'mode': by_rel[0]['mode'], 'rel_mm': round(by_rel[0]['rel_mm'], 2),
            'dq': round(by_rel[0]['dq'], 4), 'yaw_deg': round(by_rel[0]['yaw_deg'], 2)},
        'nearest_by_ee_distance': {
            'episode': by_ee[0]['index'], 'group': by_ee[0]['spec']['group'],
            'mode': by_ee[0]['mode'], 'ee_mm': round(by_ee[0]['ee_mm'], 2),
            'dq': round(by_ee[0]['dq'], 4), 'yaw_deg': round(by_ee[0]['yaw_deg'], 2)},
        'counts_within': {
            'q_lt_0.5': sum(1 for r in D if r['dq'] < 0.5),
            'q_lt_0.75': sum(1 for r in D if r['dq'] < 0.75),
            'q_lt_1.0': sum(1 for r in D if r['dq'] < 1.0),
            'rel_lt_50mm': sum(1 for r in D if r['rel_mm'] < 50),
            'rel_lt_100mm': sum(1 for r in D if r['rel_mm'] < 100),
            'ee_lt_100mm': sum(1 for r in D if r['ee_mm'] < 100),
            'yaw_lt_10deg': sum(1 for r in D if r['yaw_deg'] < 10),
            'gap_lt_10mm': sum(1 for r in D if r['gap_mm'] < 10),
        },
        'reference': {'all_min': round(float(ref_all.min()), 6),
                      'all_median': round(float(np.median(ref_all)), 6),
                      'all_p95': round(float(np.percentile(ref_all, 95)), 6),
                      'all_max': round(float(ref_all.max()), 6),
                      'yaw90_min': round(float(ref_y90.min()), 6),
                      'yaw90_median': round(float(np.median(ref_y90)), 6),
                      'yaw90_p95': round(float(np.percentile(ref_y90, 95)), 6),
                      'yaw90_max': round(float(ref_y90.max()), 6)},
        'c3_distance_percentile_in_ref': {
            'all': round(float((ref_all <= D[0]['dq']).mean() * 100), 1),
            'yaw90': round(float((ref_y90 <= yaw90[0]['dq']).mean() * 100), 1)},
        'trajectory_nn': 'unavailable',
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding='utf-8')

    # 图 1：yaw90 起点距离
    def bar(path, vals, labels, title, ylabel='q L2 (rad)'):
        W, H = 1000, 420
        img = Image.new('RGB', (W, H), 'white')
        d = ImageDraw.Draw(img)
        d.text((40, 20), title, font=font(18), fill='black')
        L, R, T, B = 70, 960, 70, 330
        d.rectangle([L, T, R, B], outline='black')
        mx = max(vals) * 1.1
        slot = (R - L) / len(vals)
        for i, (v, lab) in enumerate(zip(vals, labels)):
            h = (B - T) * v / mx
            x = L + i * slot + slot * 0.15
            d.rectangle([x, B - h, x + slot * 0.7, B], fill='#1f77b4', outline='black')
            d.text((x, B - h - 14), f'{v:.3f}', font=font(9), fill='black')
            d.text((x, B + 8), lab, font=font(8), fill='black')
        for i in range(5):
            y = B - (B - T) * i / 4
            d.text((10, y - 6), f'{mx * i / 4:.2f}', font=font(9), fill='black')
        img.save(path)

    bar(FIG / 'fig1_initial_state_nn.png',
        [r['dq'] for r in yaw90[:10]],
        [f"ep{r['index']}" for r in yaw90[:10]],
        'Figure 1: C3 vs D2 B yaw90 training starts (q L2)')

    # 图 3：关节角对比
    def joints(path):
        W, H = 1000, 420
        img = Image.new('RGB', (W, H), 'white')
        d = ImageDraw.Draw(img)
        d.text((40, 20), 'Figure 3: C3 vs nearest D2 B starts (7 joints)', font=font(18), fill='black')
        L, R, T, B = 70, 960, 70, 330
        d.rectangle([L, T, R, B], outline='black')
        vals = np.array([c3['q'], D[0]['_q'], yaw90[0]['_q']])
        lo, hi = vals.min() - 0.2, vals.max() + 0.2
        for j in range(7):
            x = L + 40 + j * 120
            for k, (col, lab) in enumerate([('#1f77b4', 'C3'), ('#d62728', 'NN all'),
                                            ('#2ca02c', 'NN yaw90')]):
                v = vals[k, j]
                y = B - (B - T) * (v - lo) / (hi - lo)
                d.rectangle([x + k * 16, y, x + 14 + k * 16, B], fill=col, outline='black')
            d.text((x + 10, B + 8), f'q{j + 1}', font=font(10), fill='black')
        for k, (col, lab) in enumerate([('#1f77b4', 'C3'), ('#d62728', 'NN all'),
                                        ('#2ca02c', 'NN yaw90')]):
            y = B + 45 + k * 16
            d.rectangle([L, y, L + 14, y + 12], fill=col)
            d.text((L + 18, y - 2), lab, font=font(10), fill='black')
        img.save(path)

    joints(FIG / 'fig3_joint_comparison.png')

    # 图 4：相对 EE-object 位置（x-z）
    def rel(path):
        W, H = 1000, 480
        img = Image.new('RGB', (W, H), 'white')
        d = ImageDraw.Draw(img)
        d.text((40, 20), 'Figure 4: relative EE-object position (x-z)', font=font(18), fill='black')
        L, R, T, B = 80, 900, 70, 390
        d.rectangle([L, T, R, B], outline='black')
        pts = [(c3['ee'] - c3['obj'])] + [(r['_ee'] - r['_obj']) for r in D[:10]]
        xs = [p[0] for p in pts]; zs = [p[2] for p in pts]
        xmin, xmax = min(xs) - 0.01, max(xs) + 0.01
        zmin, zmax = min(zs) - 0.01, max(zs) + 0.01
        for i, p in enumerate(pts):
            x = L + (R - L) * (p[0] - xmin) / (xmax - xmin)
            y = B - (B - T) * (p[2] - zmin) / (zmax - zmin)
            col = '#d62728' if i == 0 else '#1f77b4'
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)
            if i == 0:
                d.text((x + 7, y - 8), 'C3', font=font(12), fill=col)
        d.text((L, B + 10), 'relative x (m)', font=font(10), fill='black')
        d.text((10, 30), 'relative z', font=font(10), fill='black')
        img.save(path)

    rel(FIG / 'fig4_relative_geometry.png')

    # 图 6：参考分布
    def hist(path):
        W, H = 1000, 420
        img = Image.new('RGB', (W, H), 'white')
        d = ImageDraw.Draw(img)
        d.text((40, 20), 'Figure 6: D2 B start internal NN distance; C3 marker', font=font(18), fill='black')
        L, R, T, B = 70, 960, 70, 330
        d.rectangle([L, T, R, B], outline='black')
        bins = np.linspace(0, max(ref_all.max(), D[0]['dq']) * 1.05, 25)
        histv, edges = np.histogram(ref_all, bins=bins)
        mx = max(histv) * 1.1
        for i, v in enumerate(histv):
            x0 = L + (R - L) * (edges[i] / bins[-1])
            x1 = L + (R - L) * (edges[i + 1] / bins[-1])
            h = (B - T) * v / mx
            d.rectangle([x0, B - h, x1 - 1, B], fill='#1f77b4')
        x = L + (R - L) * D[0]['dq'] / bins[-1]
        d.line([x, T, x, B], fill='#d62728', width=3)
        d.text((x + 4, T + 4), f"C3 {D[0]['dq']:.3f}", font=font(11), fill='#d62728')
        d.text((L, B + 10), 'q L2 (rad)', font=font(10), fill='black')
        img.save(path)

    hist(FIG / 'fig6_reference_distribution.png')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print('\nTop all:')
    for r in D[:10]:
        print(r['index'], r['spec']['group'], r['mode'], round(r['dq'], 4),
              round(r['ee_mm'], 1), round(r['ori_deg'], 1), round(r['rel_mm'], 1),
              round(r['yaw_deg'], 1), round(r['gap_mm'], 1))
    print('\nTop yaw90:')
    for r in yaw90[:10]:
        print(r['index'], r['mode'], round(r['dq'], 4), round(r['ee_mm'], 1),
              round(r['ori_deg'], 1), round(r['rel_mm'], 1), round(r['yaw_deg'], 1),
              round(r['gap_mm'], 1))


if __name__ == '__main__':
    main()
