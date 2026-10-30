"""实验 2.8-S1：D2 专家动作目标高度、夹爪时序、条件覆盖与 2.8 模型对照。"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import pyarrow.parquet as pq

def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

ROOT = Path('/scratch/lijh/simulate_vla')
SHARD = ROOT / 'data/_shards_e23/B_D2'
OUT = ROOT / 'results/experiment_2_8/supplement_s1'
DATA = OUT / 'data'; DATA.mkdir(parents=True, exist_ok=True)
HORIZONS = [0, 4, 9, 19, 49]


def fk_xyz(qs):
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=c)
    out = []
    for q in qs:
        for j in range(7):
            p.resetJointState(robot, j, float(q[j]), physicsClientId=c)
        out.append(p.getLinkState(robot, 11, computeForwardKinematics=True,
                                  physicsClientId=c)[4])
    p.disconnect(c)
    return np.asarray(out)


def sustained(mask, n=3):
    m = np.asarray(mask, bool)
    for i in range(0, max(0, len(m) - n + 1)):
        if m[i:i + n].all():
            return i
    return None


def main():
    eps = [json.loads(x) for x in (SHARD / 'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
    meta = [json.loads(x) for x in (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text().splitlines() if x.strip()]
    ok = [r for r in meta if r['success'] and r['reason'] == 'ok']
    dyn = []
    chunk = []
    for e in eps:
        i = e['episode_index']
        if i < 60 or i >= len(ok) or ok[i]['spec']['group'] != 'yaw90':
            continue
        d = pq.read_table(SHARD / f'data/chunk-000/episode_{i:06d}.parquet',
                          columns=['observation.state', 'action', 'timestamp']).to_pydict()
        st = np.asarray(d['observation.state'], float)
        ac = np.asarray(d['action'], float)
        ts = np.asarray(d['timestamp'], float)
        zobs = fk_xyz(st[:, :7])[:, 2]
        zact = fk_xyz(ac[:, :7])[:, 2]
        n = len(st)
        vz = np.zeros(n)
        for t in range(n - 1):
            dt = ts[t + 1] - ts[t]
            if dt > 0:
                vz[t] = (zobs[t + 1] - zobs[t]) / dt * 1000.0
        for t in range(n):
            dyn.append({'episode_id': i, 'frame_id': t, 'timestamp': float(ts[t]),
                        'yaw': 'yaw90', 'progress': t / max(1, n - 1),
                        'gripper_obs_mm': float(st[t, 7:9].mean() * 1000),
                        'q_obs': ';'.join(f'{v:.7f}' for v in st[t, :7]),
                        'q_target': ';'.join(f'{v:.7f}' for v in ac[t, :7]),
                        'z_obs_mm': float(zobs[t] * 1000),
                        'z_target_mm': float(zact[t] * 1000),
                        'dz_target_mm': float((zact[t] - zobs[t]) * 1000),
                        'gripper_target_mm': float(ac[t, 7] * 1000),
                        'vz_actual_mm_s': float(vz[t])})
            for k in HORIZONS:
                valid = t + k < n
                chunk.append({'episode_id': i, 'frame_id': t, 'timestamp': float(ts[t]),
                              'horizon': k, 'valid': int(valid),
                              'gripper_obs_mm': float(st[t, 7:9].mean() * 1000),
                              'z_obs_mm': float(zobs[t] * 1000),
                              'z_target_mm': float(zact[t + k] * 1000) if valid else '',
                              'dz_target_mm': float((zact[t + k] - zobs[t]) * 1000) if valid else ''})
        print('ep', i, n, flush=True)
    with (DATA / 'expert_target_dynamics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(dyn[0].keys())); w.writeheader(); w.writerows(dyn)
    with (DATA / 'expert_chunk_targets.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(chunk[0].keys())); w.writeheader(); w.writerows(chunk)

    # grip-bin stats. Use numeric tolerance so 34.99998 and 35.00002
    # (the same physical initial opening) are not split across bins.
    G = np.array([r['gripper_obs_mm'] for r in dyn])
    bins = [(13.0, 15.5, '14mm'), (15.5, 24.5, '16-25mm'),
            (24.5, 34.5, '25-35mm'), (34.5, 40.5, '35-40mm')]
    chunk_map = {(int(c['episode_id']), int(c['frame_id']), int(c['horizon'])): c
                 for c in chunk}
    stat_rows = []
    for lo, hi, name in bins:
        m = (G >= lo) & (G < hi)
        if not m.any():
            continue
        row = {'grip_bin': name, 'n_frames': int(m.sum()),
               'n_episodes': len(set(int(dyn[i]['episode_id']) for i in np.where(m)[0]))}
        for k in HORIZONS:
            vals = []
            for i in np.where(m)[0]:
                r = dyn[i]
                c = chunk_map[(int(r['episode_id']), int(r['frame_id']), k)]
                if int(c['valid']) == 1:
                    vals.append(float(c['dz_target_mm']))
            vals = np.asarray(vals)
            row[f'dz_target_h{k}_mean_mm'] = float(vals.mean()) if len(vals) else ''
            row[f'dz_target_h{k}_std_mm'] = float(vals.std()) if len(vals) else ''
        vals = np.array([r['dz_target_mm'] for r in dyn])[m]
        row['dz_target0_mean_mm'] = float(vals.mean())
        row['target_descent_frac'] = float((vals < -1).mean())
        row['actual_vz_mean_mm_s'] = float(np.mean([r['vz_actual_mm_s'] for r in dyn if (lo <= r['gripper_obs_mm'] < hi)]))
        stat_rows.append(row)
    with (DATA / 'gripper_target_statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys())); w.writeheader(); w.writerows(stat_rows)

    # per-episode events, multiple thresholds
    by_ep = {}
    for r in dyn:
        by_ep.setdefault(r['episode_id'], []).append(r)
    ev = []
    variants = [(2, 3, -2, -20), (2, 3, -5, -50), (2, 3, -10, -80)]
    for eid, rs in sorted(by_ep.items()):
        rs.sort(key=lambda x: x['frame_id'])
        g = np.array([r['gripper_obs_mm'] for r in rs]); z = np.array([r['dz_target_mm'] for r in rs]); v = np.array([r['vz_actual_mm_s'] for r in rs])
        close = sustained(g <= g[0] - 2.0, 2)
        first25 = sustained(g <= 25, 2)
        first16 = sustained(g <= 16, 2)
        row = {'episode_id': eid, 'n_frames': len(rs), 'grip_initial_mm': float(g[0]),
               'close_start_t': close * 0.05 if close is not None else '',
               'first25_t': first25 * 0.05 if first25 is not None else '',
               'first16_t': first16 * 0.05 if first16 is not None else ''}
        for name, (_, _, zthr, athr) in zip(('lo', 'mid', 'hi'), variants):
            zt = sustained(z < zthr, 3); at = sustained(v < athr, 3)
            row[f'target_descent_{name}_t'] = zt * 0.05 if zt is not None else ''
            row[f'actual_descent_{name}_t'] = at * 0.05 if at is not None else ''
            if close is not None and zt is not None:
                row[f'close_to_target_{name}_s'] = (zt - close) * 0.05
            if close is not None and at is not None:
                row[f'close_to_actual_{name}_s'] = (at - close) * 0.05
        ev.append(row)
    with (DATA / 'episode_timing_events.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(ev[0].keys())); w.writeheader(); w.writerows(ev)
    # summarize timing thresholds
    timing = []
    for name, zthr, athr in (('lo', -2, -20), ('mid', -5, -50), ('hi', -10, -80)):
        for key in (f'close_to_target_{name}_s', f'close_to_actual_{name}_s'):
            vals = [float(r[key]) for r in ev if r.get(key, '') != '']
            if vals:
                timing.append({'variant': name, 'comparison': key, 'n': len(vals),
                               'mean_s': float(np.mean(vals)), 'median_s': float(np.median(vals)),
                               'min_s': float(np.min(vals)), 'max_s': float(np.max(vals)),
                               'positive': int(sum(v > 0 for v in vals)), 'negative': int(sum(v < 0 for v in vals))})
    with (DATA / 'timing_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(timing[0].keys())); w.writeheader(); w.writerows(timing)

    # conditional stats by progress x grip bin. A row is estimable only if
    # its own group and at least one other group in the same progress bin
    # both have >=50 frames from >=10 independent episodes.
    cond = []
    for plo, phi in ((0, .15), (.15, .3), (.3, .5), (.5, 1.01)):
        pm = [r for r in dyn if plo <= r['progress'] < phi]
        rows_here = []
        for lo, hi, name in bins:
            sub = [r for r in pm if lo <= r['gripper_obs_mm'] < hi]
            if not sub:
                continue
            rows_here.append({'progress_bin': f'{plo:.2f}-{min(phi,1):.2f}', 'grip_bin': name,
                              'n_frames': len(sub), 'n_episodes': len(set(r['episode_id'] for r in sub)),
                              'dz_target_mean_mm': float(np.mean([r['dz_target_mm'] for r in sub])),
                              'actual_vz_mean_mm_s': float(np.mean([r['vz_actual_mm_s'] for r in sub])),
                              'target_descent_frac': float(np.mean([r['dz_target_mm'] < -1 for r in sub]))})
        qualifying = [r for r in rows_here if r['n_frames'] >= 50 and r['n_episodes'] >= 10]
        for r in rows_here:
            r['n_qualifying_grip_groups'] = len(qualifying)
            r['estimable'] = bool(len(qualifying) >= 2 and r['n_frames'] >= 50
                                  and r['n_episodes'] >= 10)
            cond.append(r)
    with (DATA / 'conditional_target_statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(cond[0].keys())); w.writeheader(); w.writerows(cond)

    # local coverage for 32 states + model comparison
    states = list(csv.DictReader((ROOT / 'results/experiment_2_8/data/state_manifest.csv').open(encoding='utf-8-sig')))
    model = {r['state_id']: r for r in csv.DictReader((ROOT / 'results/experiment_2_8/data/cross_state_analysis.csv').open(encoding='utf-8-sig'))}
    Q = np.array([[float(x) for x in r['q_obs'].split(';')] for r in dyn])
    E = np.array([r['episode_id'] for r in dyn])
    local = []
    comp = []
    for s in states:
        q = np.array([float(x) for x in s['q'].split(';')])
        d = np.linalg.norm(Q - q, axis=1)
        for rad in (0.15, 0.30, 0.50, 0.75):
            m = d <= rad
            near14 = int(((G[m] >= 11) & (G[m] <= 17)).sum()) if m.any() else 0
            near35 = int(((G[m] >= 32) & (G[m] <= 41)).sum()) if m.any() else 0
            local.append({'state_id': s['state_id'], 'radius_rad': rad, 'n_frames': int(m.sum()),
                          'episodes': int(len(set(E[m].tolist()))) if m.any() else 0,
                          'grip_min_mm': float(G[m].min()) if m.any() else '',
                          'grip_max_mm': float(G[m].max()) if m.any() else '',
                          'near14_frames': near14, 'near35_frames': near35,
                          'dz_target_mean_mm': float(np.mean([r['dz_target_mm'] for i, r in enumerate(dyn) if m[i]])) if m.any() else '',
                          'actual_vz_mean_mm_s': float(np.mean([r['vz_actual_mm_s'] for i, r in enumerate(dyn) if m[i]])) if m.any() else '',
                          'has_common_support': bool(near14 >= 5 and near35 >= 5)})
        m = d <= 0.30
        comp.append({'state_id': s['state_id'], 'episode_id': s['episode_id'], 'frame_id': s['frame_id'],
                     'stage': s['stage_progress_label'], 'progress': s['episode_progress'],
                     'ee_z_m': s['ee_z'],
                     'model_delta_z_mm': model.get(s['state_id'], {}).get('mean_delta_z_mm', ''),
                     'local_n_frames': int(m.sum()), 'local_n_episodes': int(len(set(E[m].tolist()))) if m.any() else 0,
                     'local_grip_max_mm': float(G[m].max()) if m.any() else '',
                     'local_target_dz_mm': float(np.mean([r['dz_target_mm'] for i, r in enumerate(dyn) if m[i]])) if m.any() else '',
                     'local_actual_vz_mm_s': float(np.mean([r['vz_actual_mm_s'] for i, r in enumerate(dyn) if m[i]])) if m.any() else '',
                     'local_near35': int(((G[m] >= 32) & (G[m] <= 41)).sum()) if m.any() else 0})
    with (DATA / 'local_target_coverage.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(local[0].keys())); w.writeheader(); w.writerows(local)
    with (DATA / 'model_expert_comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys())); w.writeheader(); w.writerows(comp)

    # global correlations
    g = np.array([r['gripper_obs_mm'] for r in dyn])
    dz = np.array([r['dz_target_mm'] for r in dyn])
    v = np.array([r['vz_actual_mm_s'] for r in dyn])
    print(json.dumps({'n_frames': len(dyn), 'n_episodes': len(by_ep),
                      'pearson_g_dz': float(np.corrcoef(g, dz)[0, 1]),
                      'spearman_g_dz': spearman(g, dz),
                      'pearson_g_v': float(np.corrcoef(g, v)[0, 1]),
                      'spearman_g_v': spearman(g, v),
                      'dz_mean': float(dz.mean())}, indent=2))
    (OUT / 'analysis.json').write_text(json.dumps({'n_frames': len(dyn), 'n_episodes': len(by_ep),
                                                   'pearson_g_dz': float(np.corrcoef(g, dz)[0, 1]),
                                                   'spearman_g_dz': spearman(g, dz),
                                                   'pearson_g_v': float(np.corrcoef(g, v)[0, 1]),
                                                   'spearman_g_v': spearman(g, v)}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
