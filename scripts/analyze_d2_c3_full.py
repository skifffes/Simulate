"""D2-C3 full coverage: B shard all frames, 3 C3 queries, start/interior/all NN.

Server: /scratch/lijh/simulate_vla
Reads data/_shards_e23/B_D2 (120 episodes, 14653 frames) and results/experiment_2_5/G1/C3.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/scratch/lijh/simulate_vla')
SHARD = ROOT / 'data/_shards_e23/B_D2'
OUT = ROOT / 'results/experiment_2_5/d2_c3_coverage'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
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
    d = abs(float(np.dot(q1, q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def fk_poses(qs):
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=c)
    out = []
    for q in qs:
        for j in range(7):
            p.resetJointState(robot, j, float(q[j]), physicsClientId=c)
        ls = p.getLinkState(robot, 11, computeForwardKinematics=True, physicsClientId=c)
        out.append((np.array(ls[4]), np.array(ls[5])))
    p.disconnect(c)
    return out


def load_b():
    eps = [json.loads(x) for x in (SHARD / 'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
    Q, A, G, meta = [], [], [], []
    for e in eps:
        idx = e['episode_index']
        f = SHARD / f'data/chunk-000/episode_{idx:06d}.parquet'
        t = pq.read_table(f, columns=['observation.state', 'action', 'frame_index', 'timestamp']).to_pydict()
        st = np.asarray(t['observation.state'], float)
        ac = np.asarray(t['action'], float)
        Q.append(st[:, :7]); G.append(st[:, 7:9].mean(1)); A.append(ac)
        meta.append({'episode': idx, 'length': len(st), 'frame_index': np.asarray(t['frame_index']),
                     'timestamp': np.asarray(t['timestamp'])})
    return Q, A, G, meta


def load_d2meta():
    rows = [json.loads(x) for x in (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text().splitlines() if x.strip()]
    return [r for r in rows if r['success'] and r['reason'] == 'ok']


def load_c3(seed):
    d = ROOT / f'results/experiment_2_5/G1/C3/seed_{seed}'
    pol = [json.loads(x) for x in (d / 'policy_trace.jsonl').read_text().splitlines() if x.strip()]
    b = next(r for r in pol if r['skill'] == 'B')
    phys = list(csv.DictReader((d / 'physics_trace.csv').open(encoding='utf-8-sig')))
    pr = min(phys, key=lambda r: abs(int(r['physics_step']) - int(b['physics_step'])))
    q = np.asarray(b['state'][:7], float)
    ee = np.array([float(pr['ee_x']), float(pr['ee_y']), float(pr['ee_z'])])
    quat = np.array([float(pr[f'ee_q{i}']) for i in ('x', 'y', 'z', 'w')])
    obj = np.array([float(pr['obj_x']), float(pr['obj_y']), float(pr['obj_z'])])
    polB = [r for r in pol if r['skill'] == 'B']
    succ = [np.asarray(r['state'][:7], float) for r in polB[:40]]
    act = [np.asarray(r['action'][:7], float) for r in polB[:40]]
    width = [float(r['executed_gripper_target']) for r in polB[:40]]
    return {'seed': seed, 'q': q, 'ee': ee, 'quat': quat, 'obj': obj,
            'yaw': float(pr['obj_yaw_rad']), 'gap': float(pr['obj_wall_gap_m']),
            'grip': float(pr['grip_actual']), 'policy_step': b['policy_step'],
            'physics_step': b['physics_step'], 'vla_q': succ, 'vla_action': act,
            'vla_width': width, 'task': b['task']}


def main():
    Q, A, G, meta = load_b()
    d2meta = load_d2meta()
    # match each shard episode start to collection metadata
    starts = np.array([q[0] for q in Q])
    ds, mapping = [], []
    for i, q0 in enumerate(starts):
        j = int(np.argmin([np.linalg.norm(q0 - np.asarray(r['start_q'])) for r in d2meta]))
        dist = float(np.linalg.norm(q0 - np.asarray(d2meta[j]['start_q'])))
        mapping.append((i, j, dist))
        ds.append(d2meta[j])
    print('mapping max dist', max(x[2] for x in mapping))
    # flatten
    FQ = np.vstack(Q)
    FG = np.concatenate(G)
    FA = np.vstack(A)
    fe = np.concatenate([[i] * len(Q[i]) for i in range(len(Q))])
    ff = np.concatenate([meta[i]['frame_index'] for i in range(len(Q))])
    ft = np.concatenate([meta[i]['timestamp'] for i in range(len(Q))])
    lengths = np.array([len(q) for q in Q])
    starts_idx = np.concatenate([[0] + list(np.cumsum(lengths)[:-1])])
    interior_mask = np.ones(len(FQ), bool)
    for s, l in zip(starts_idx, lengths):
        interior_mask[s] = False
    ep_group = [ds[i]['spec']['group'] for i in range(len(Q))]
    yaw90_eps = {i for i, g in enumerate(ep_group) if g == 'yaw90'}
    interior90_mask = interior_mask & np.isin(fe, list(yaw90_eps))

    # initial NN reference
    def ref_nn(sub):
        vals = []
        for i, a in enumerate(sub):
            vals.append(min(np.linalg.norm(a - b) for j, b in enumerate(sub) if j != i))
        return np.asarray(vals)

    starts_q = [Q[i][0] for i in range(len(Q))]
    ref_start = ref_nn(starts_q)
    # interior reference: each episode min to other episodes
    ref_inter = []
    for i in range(len(Q)):
        mask = fe != i
        d = np.linalg.norm(FQ[mask] - Q[i][0], axis=1).min()
        ref_inter.append(d)
    ref_inter = np.asarray(ref_inter)

    queries = [load_c3(s) for s in (40000000, 40000001, 40000002)]
    # start NN
    init_rows = []
    for q in queries:
        dsq = np.linalg.norm(starts - q['q'], axis=1)
        order = np.argsort(dsq)
        for rank, i in enumerate(order[:20], 1):
            r = ds[i]
            ee = np.asarray(r['start_ee'])
            obj = np.asarray(r['scene']['block_start'])
            yaw = float(r['scene']['block_yaw'])
            yawdiff = abs(math.degrees(wrap(q['yaw'] - yaw)))
            yawdiff = min(yawdiff, 180 - yawdiff)
            gap = float(r['scene']['wall_inner_x']) - obj[0] - (0.05 if abs(yaw) > 0.8 else 0.02)
            init_rows.append({
                'query_seed': q['seed'], 'rank': rank, 'episode_index': i,
                'collection_index': r['index'], 'group': r['spec']['group'], 'mode': r['mode'],
                'start_type': 'HOME' if r['mode'] == 'home' else 'diverse',
                'dq': float(dsq[i]),
                'ee_mm': float(np.linalg.norm(q['ee'] - ee) * 1000),
                'rel_mm': float(np.linalg.norm((q['ee'] - q['obj']) - (ee - obj)) * 1000),
                'ori_deg': quat_angle_deg(q['quat'], fk_poses([r['start_q']])[0][1]),
                'yaw_deg': yawdiff, 'gap_mm': abs(q['gap'] - gap) * 1000,
                'grip_obs': float(G[i][0]),
                'grip_diff_mm': abs(q['grip'] - G[i][0]) * 1000,
                'home_distance_rad': r['home_distance_rad'],
                'q0_dist': 0.0,
            })
    # interior/all NN
    inter_rows, all_rows = [], []
    inter90_rows = []
    nearest_info = []
    nearest90_info = []
    for q in queries:
        dq = np.linalg.norm(FQ - q['q'], axis=1)
        order = np.argsort(dq)
        # all
        for rank, k in enumerate(order[:20], 1):
            all_rows.append({'query_seed': q['seed'], 'rank': rank, 'episode_index': int(fe[k]),
                             'frame_index': int(ff[k]), 'timestamp': float(ft[k]),
                             'progress': float(ff[k] / max(1, lengths[int(fe[k])] - 1)),
                             'dq': float(dq[k]), 'grip_obs': float(FG[k]),
                             'grip_diff_mm': abs(float(FG[k]) - q['grip']) * 1000})
        # interior
        order_i = [k for k in order if interior_mask[k]][:20]
        for rank, k in enumerate(order_i, 1):
            inter_rows.append({'query_seed': q['seed'], 'rank': rank, 'episode_index': int(fe[k]),
                               'frame_index': int(ff[k]), 'timestamp': float(ft[k]),
                               'progress': float(ff[k] / max(1, lengths[int(fe[k])] - 1)),
                               'dq': float(dq[k]), 'grip_obs': float(FG[k]),
                             'grip_diff_mm': abs(float(FG[k]) - q['grip']) * 1000})
        k = order_i[0]
        nearest_info.append((q, int(fe[k]), int(ff[k]), float(dq[k]), int(k)))
        order90 = [k for k in order if interior90_mask[k]][:20]
        for rank, k in enumerate(order90, 1):
            inter90_rows.append({'query_seed': q['seed'], 'rank': rank,
                                 'episode_index': int(fe[k]), 'frame_index': int(ff[k]),
                                 'timestamp': float(ft[k]),
                                 'progress': float(ff[k] / max(1, lengths[int(fe[k])] - 1)),
                                 'dq': float(dq[k]), 'grip_obs': float(FG[k]),
                                 'grip_diff_mm': abs(float(FG[k]) - q['grip']) * 1000})
        k90 = order90[0]
        nearest90_info.append((q, int(fe[k90]), int(ff[k90]), float(dq[k90]), int(k90)))
    # dedup per episode
    dedup = []
    for q in queries:
        dq = np.linalg.norm(FQ - q['q'], axis=1)
        best = {}
        for i in range(len(Q)):
            k = np.where(fe == i)[0][np.argmin(dq[fe == i])]
            best[i] = (dq[k], k)
        for i, (d, k) in sorted(best.items(), key=lambda kv: kv[1][0])[:20]:
            dedup.append({'query_seed': q['seed'], 'episode_index': i,
                          'frame_index': int(ff[k]), 'timestamp': float(ft[k]),
                          'progress': float(ff[k] / max(1, lengths[i] - 1)),
                          'dq': float(d), 'grip_obs': float(FG[k]),
                          'grip_diff_mm': abs(float(FG[k]) - q['grip']) * 1000})
    # gripper-constrained searches (analysis tolerance, not a success threshold)
    grip_rows = []
    start_mask = np.zeros(len(FQ), bool)
    start_mask[starts_idx] = True
    for tol in (0.002, 0.0051, 0.010):
        for q in queries:
            for subset_name, mask in (('starts', start_mask),
                                      ('interior', interior_mask),
                                      ('interior_yaw90', interior90_mask),
                                      ('all_frames', np.ones(len(FQ), bool))):
                m = mask & (np.abs(FG - q['grip']) <= tol)
                if not m.any():
                    grip_rows.append({'query_seed': q['seed'], 'tolerance_mm': tol * 1000,
                                      'subset': subset_name, 'n_frames': 0})
                    continue
                d = np.linalg.norm(FQ[m] - q['q'], axis=1)
                kk = np.where(m)[0][int(np.argmin(d))]
                grip_rows.append({'query_seed': q['seed'], 'tolerance_mm': tol * 1000,
                                  'subset': subset_name, 'n_frames': int(m.sum()),
                                  'episode_index': int(fe[kk]),
                                  'frame_index': int(ff[kk]),
                                  'progress': round(float(ff[kk] / max(1, lengths[int(fe[kk])] - 1)), 3),
                                  'dq': round(float(d.min()), 6),
                                  'grip_obs': round(float(FG[kk]), 6),
                                  'grip_diff_mm': round(abs(float(FG[kk]) - q['grip']) * 1000, 3),
                                  'n_episodes': int(len(set(fe[m].tolist())))})

    # write
    def wcsv(path, rows):
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            if not rows:
                return
            fields = []
            for r in rows:
                for k in r:
                    if k not in fields:
                        fields.append(k)
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    wcsv(OUT / 'initial_nn.csv', init_rows)
    wcsv(OUT / 'interior_nn.csv', inter_rows)
    wcsv(OUT / 'interior_nn_yaw90.csv', inter90_rows)
    wcsv(OUT / 'gripper_constrained_nn.csv', grip_rows)
    wcsv(OUT / 'all_frame_nn.csv', all_rows)
    wcsv(OUT / 'episode_deduplicated_nn.csv', dedup)

    # query csv
    qrows = []
    for q in queries:
        qrows.append({'seed': q['seed'], 'policy_step': q['policy_step'],
                      'physics_step': q['physics_step'],
                      **{f'q{i+1}': float(q['q'][i]) for i in range(7)},
                      'grip': q['grip'], 'ee_x': q['ee'][0], 'ee_y': q['ee'][1], 'ee_z': q['ee'][2],
                      'obj_x': q['obj'][0], 'obj_y': q['obj'][1], 'obj_z': q['obj'][2],
                      'obj_yaw_deg': math.degrees(q['yaw']), 'gap_m': q['gap']})
    wcsv(OUT / 'c3_query_states.csv', qrows)

    # reference
    with (OUT / 'reference_distance_distribution.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['subset', 'stat', 'value_rad'])
        for name, arr in (('start_120', ref_start), ('interior_episode_min_120', ref_inter)):
            for st, v in (('min', arr.min()), ('median', np.median(arr)),
                          ('p95', np.percentile(arr, 95)), ('max', arr.max())):
                w.writerow([name, st, round(float(v), 6)])

    summary = {'n_b_episodes': len(Q), 'n_b_frames': len(FQ),
               'mapping_max_start_distance': round(max(x[2] for x in mapping), 8),
               'queries': [], 'reference': {
                   'start_median': round(float(np.median(ref_start)), 6),
                   'start_p95': round(float(np.percentile(ref_start, 95)), 6),
                   'start_max': round(float(ref_start.max()), 6),
                   'interior_episode_min_median': round(float(np.median(ref_inter)), 6),
                   'interior_episode_min_p95': round(float(np.percentile(ref_inter, 95)), 6),
                   'interior_episode_min_max': round(float(ref_inter.max()), 6)}}
    for q in queries:
        dq_start = np.linalg.norm(starts - q['q'], axis=1)
        dq_all = np.linalg.norm(FQ - q['q'], axis=1)
        inter = dq_all.copy(); inter[~interior_mask] = 1e9
        inter90 = dq_all.copy(); inter90[~interior90_mask] = 1e9
        k = int(np.argmin(inter))
        k90 = int(np.argmin(inter90))
        summary['queries'].append({
            'seed': q['seed'], 'q': [round(float(v), 6) for v in q['q']],
            'start_min': round(float(dq_start.min()), 6),
            'start_episode': int(np.argmin(dq_start)),
            'interior_min': round(float(inter.min()), 6),
            'interior_episode': int(fe[k]), 'interior_frame': int(ff[k]),
            'interior_progress': round(float(ff[k] / max(1, lengths[int(fe[k])] - 1)), 3),
            'interior_yaw90_min': round(float(inter90.min()), 6),
            'interior_yaw90_episode': int(fe[k90]), 'interior_yaw90_frame': int(ff[k90]),
            'interior_yaw90_progress': round(float(ff[k90] / max(1, lengths[int(fe[k90])] - 1)), 3),
            'all_min': round(float(dq_all.min()), 6),
            'start_percentile': round(float((ref_start <= dq_start.min()).mean() * 100), 1),
            'interior_percentile': round(float((ref_inter <= inter.min()).mean() * 100), 1)})
    start_grips = [float(G[i][0]) for i in range(len(Q))]
    k66 = int(starts_idx[66] + 39)
    audit = {'c3_obs_grip_m': [round(q['grip'], 6) for q in queries],
             'c3_first_action_grip_m': [round(float(q['vla_width'][0]), 6) for q in queries],
             'b_start_grip_min_m': round(min(start_grips), 6),
             'b_start_grip_max_m': round(max(start_grips), 6),
             'ep66_f39_obs_grip_m': round(float(FG[k66]), 6),
             'ep66_f39_action_grip_m': round(float(FA[k66, 7]), 6),
             'ep66_f39_frame': int(ff[k66])}
    summary['state_alignment_audit'] = audit
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # nearest trajectory comparison: expert next 30 vs VLA next 30
    comp = []
    for q, ep, fr, d, k in nearest_info:
        start = starts_idx[ep] + fr
        q_ex = Q[ep][fr:fr + 30]
        a_ex = A[ep][fr:fr + 30]
        g_ex = G[ep][fr:fr + 30]
        poses = fk_poses(q_ex)
        vq = np.asarray(q['vla_q'][:30]); va = np.asarray(q['vla_action'][:30])
        vposes = fk_poses(vq)
        for t in range(min(30, len(q_ex), len(vq))):
            comp.append({'query_seed': q['seed'], 't': t,
                         'expert_episode': ep, 'expert_frame': fr + t,
                         'expert_q1': q_ex[t][0], 'expert_q2': q_ex[t][1], 'expert_q3': q_ex[t][2],
                         'expert_q4': q_ex[t][3], 'expert_q5': q_ex[t][4], 'expert_q6': q_ex[t][5],
                         'expert_q7': q_ex[t][6], 'expert_grip': g_ex[t],
                         'expert_ee_x': poses[t][0][0], 'expert_ee_y': poses[t][0][1], 'expert_ee_z': poses[t][0][2],
                         'vla_q1': vq[t][0], 'vla_q2': vq[t][1], 'vla_q3': vq[t][2],
                         'vla_q4': vq[t][3], 'vla_q5': vq[t][4], 'vla_q6': vq[t][5], 'vla_q7': vq[t][6],
                         'vla_grip': q['vla_width'][t],
                         'vla_ee_x': vposes[t][0][0], 'vla_ee_y': vposes[t][0][1], 'vla_ee_z': vposes[t][0][2]})
    wcsv(OUT / 'nearest_trajectory_comparison.csv', comp)
    # yaw90 task-compatible nearest frame comparison
    comp90 = []
    for q, ep, fr, d, k in nearest90_info:
        q_ex = Q[ep][fr:fr + 30]
        g_ex = G[ep][fr:fr + 30]
        poses = fk_poses(q_ex)
        vq = np.asarray(q['vla_q'][:30]); vposes = fk_poses(vq)
        for t in range(min(30, len(q_ex), len(vq))):
            comp90.append({'query_seed': q['seed'], 't': t,
                           'expert_episode': ep, 'expert_frame': fr + t,
                           'expert_q1': q_ex[t][0], 'expert_q2': q_ex[t][1], 'expert_q3': q_ex[t][2],
                           'expert_q4': q_ex[t][3], 'expert_q5': q_ex[t][4], 'expert_q6': q_ex[t][5],
                           'expert_q7': q_ex[t][6], 'expert_grip': g_ex[t],
                           'expert_ee_x': poses[t][0][0], 'expert_ee_y': poses[t][0][1], 'expert_ee_z': poses[t][0][2],
                           'vla_q1': vq[t][0], 'vla_q2': vq[t][1], 'vla_q3': vq[t][2],
                           'vla_q4': vq[t][3], 'vla_q5': vq[t][4], 'vla_q6': vq[t][5], 'vla_q7': vq[t][6],
                           'vla_grip': q['vla_width'][t],
                           'vla_ee_x': vposes[t][0][0], 'vla_ee_y': vposes[t][0][1], 'vla_ee_z': vposes[t][0][2]})
    wcsv(OUT / 'nearest_trajectory_comparison_yaw90.csv', comp90)

    # figures
    def bar(path, vals, labels, title, ylabel='q L2 (rad)'):
        W, H = 1000, 420; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
        d.text((40, 20), title, font=font(18), fill='black'); L, R, T, B = 70, 960, 70, 330
        d.rectangle([L, T, R, B], outline='black'); mx = max(vals) * 1.15 or 1
        slot = (R - L) / len(vals)
        for i, (v, lab) in enumerate(zip(vals, labels)):
            h = (B - T) * v / mx; x = L + i * slot + slot * 0.15
            d.rectangle([x, B - h, x + slot * 0.7, B], fill='#1f77b4', outline='black')
            d.text((x, B - h - 14), f'{v:.3f}', font=font(9), fill='black')
            d.text((x, B + 8), lab, font=font(9), fill='black')
        img.save(path)

    # Fig1 top initial yaw90 for seed 0
    rows0 = [r for r in init_rows if r['query_seed'] == 40000000 and r['group'] == 'yaw90'][:10]
    bar(FIG / 'fig1_initial_state_nn.png', [r['dq'] for r in rows0],
        [f"ep{r['episode_index']}" for r in rows0], 'Figure 1: C3 vs D2 B yaw90 starts')
    # Fig2 start vs interior per query
    qs = [q['seed'] for q in queries]
    bar(FIG / 'fig2_start_vs_interior.png',
        [x for s in summary['queries'] for x in (s['start_min'], s['interior_min'])],
        [f'{s}\nstart' if i % 2 == 0 else f'{s}\ninterior' for i, s in enumerate(qs * 2)],
        'Figure 2: start vs interior nearest distance')
    # Fig3 joints C3 vs nearest start/interior (seed 0)
    W, H = 1000, 420; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
    d.text((40, 20), 'Figure 3: C3 vs nearest start/interior (seed 40000000)', font=font(18), fill='black')
    L, R, T, B = 70, 960, 70, 330; d.rectangle([L, T, R, B], outline='black')
    q0 = queries[0]; k = int(np.argmin(np.linalg.norm(FQ - q0['q'], axis=1)))
    vals = np.array([q0['q'], starts[np.argmin(np.linalg.norm(starts - q0['q'], axis=1))], FQ[k]])
    lo, hi = vals.min() - 0.2, vals.max() + 0.2
    for j in range(7):
        x = L + 40 + j * 120
        for kk, col in enumerate(['#1f77b4', '#d62728', '#2ca02c']):
            y = B - (B - T) * (vals[kk, j] - lo) / (hi - lo)
            d.rectangle([x + kk * 16, y, x + 14 + kk * 16, B], fill=col)
        d.text((x + 10, B + 8), f'q{j+1}', font=font(10), fill='black')
    img.save(FIG / 'fig3_joint_comparison.png')

    # Fig5 nearest frame position + expert/VLA grip
    if comp:
        W, H = 1000, 420; img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
        d.text((40, 20), 'Figure 5: expert vs VLA subsequent gripper (nearest frame)', font=font(16), fill='black')
        L, R, T, B = 70, 960, 70, 330; d.rectangle([L, T, R, B], outline='black')
        sub = [r for r in comp if r['query_seed'] == 40000000]
        xs = [r['t'] for r in sub]; ye = [r['expert_grip'] for r in sub]; yv = [r['vla_grip'] for r in sub]
        for vals, col in ((ye, '#d62728'), (yv, '#1f77b4')):
            pts = [(L + (R - L) * x / max(1, max(xs)), B - (B - T) * v / 0.045) for x, v in zip(xs, vals)]
            d.line(pts, fill=col, width=2)
        img.save(FIG / 'fig5_nearest_subsequent.png')

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print('yaw90 nearest:')
    for q, ep, fr, d, k in nearest90_info:
        print(q['seed'], 'ep', ep, 'frame', fr, 'dq', round(d, 4), 'progress', round(fr / max(1, lengths[ep] - 1), 3),
              'grip', round(float(FG[k]), 4))


if __name__ == '__main__':
    main()
