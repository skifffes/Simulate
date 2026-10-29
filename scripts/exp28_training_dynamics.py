"""实验 2.8 B：D2 B90 (yaw90) 专家夹爪—垂直速度统计。"""
from __future__ import annotations
import csv, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import pyarrow.parquet as pq

ROOT = Path('/scratch/lijh/simulate_vla')
SHARD = ROOT / 'data/_shards_e23/B_D2'
OUT = ROOT / 'results/experiment_2_8'
DATA = OUT / 'data'; DATA.mkdir(parents=True, exist_ok=True)


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


def main():
    meta = [json.loads(x) for x in (ROOT / 'results/experiment_2_3/data/D2_episodes.jsonl').read_text().splitlines() if x.strip()]
    ok = [r for r in meta if r['success'] and r['reason'] == 'ok']
    eps = [json.loads(x) for x in (SHARD / 'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
    rows = []
    for e in eps:
        i = e['episode_index']
        d = pq.read_table(SHARD / f'data/chunk-000/episode_{i:06d}.parquet',
                          columns=['observation.state', 'action', 'timestamp']).to_pydict()
        st = np.asarray(d['observation.state'], float)
        ac = np.asarray(d['action'], float)
        ts = np.asarray(d['timestamp'], float)
        q0 = st[0, :7]
        if i >= len(ok):
            continue
        rec = ok[i]
        group = rec['spec']['group']
        if group != 'yaw90':
            continue
        xyz = fk_xyz(st[:, :7])
        vz = np.zeros(len(st))
        dt = np.diff(ts)
        good = dt > 0
        vz[1:][good] = np.diff(xyz[:, 2])[good] / dt[good] * 1000.0
        for k in range(len(st)):
            rows.append({'episode_id': i, 'frame_id': k, 'timestamp': float(ts[k]),
                         'yaw': group, 'progress': k / max(1, len(st) - 1),
                         'gripper_per_finger_mm': float(st[k, 7:9].mean() * 1000),
                         'ee_x': float(xyz[k, 0]), 'ee_y': float(xyz[k, 1]),
                         'ee_z': float(xyz[k, 2]), 'vz_mm_s': float(vz[k]),
                         'expert_action_grip_mm': float(ac[k, 7] * 1000),
                         **{f'q{j+1}': float(st[k, j]) for j in range(7)}})
    with (DATA / 'expert_dynamics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary
    G = np.array([r['gripper_per_finger_mm'] for r in rows])
    V = np.array([r['vz_mm_s'] for r in rows])
    P = np.array([r['progress'] for r in rows])
    E = np.array([r['episode_id'] for r in rows])
    summary = []
    def add(name, mask):
        if mask.sum() < 2: return
        summary.append({'stratum': name, 'n_frames': int(mask.sum()),
                        'n_episodes': int(len(set(E[mask].tolist()))),
                        'grip_min': float(G[mask].min()), 'grip_max': float(G[mask].max()),
                        'grip_mean': float(G[mask].mean()),
                        'vz_mean': float(V[mask].mean()), 'vz_std': float(V[mask].std()),
                        'vz_p05': float(np.percentile(V[mask], 5)),
                        'vz_p50': float(np.percentile(V[mask], 50)),
                        'vz_p95': float(np.percentile(V[mask], 95))})
    add('all_yaw90', np.ones(len(rows), bool))
    for lo, hi, name in ((0, 0.15, 'progress_0_15'), (0.15, 0.3, 'progress_15_30'),
                         (0.3, 0.5, 'progress_30_50'), (0.5, 1.01, 'progress_50_100')):
        add(name, (P >= lo) & (P < hi))
    for lo, hi, name in ((0, 16, 'grip_lt16'), (16, 25, 'grip_16_25'),
                         (25, 35, 'grip_25_35'), (35, 41, 'grip_ge35')):
        add(name, (G >= lo) & (G < hi))
    with (DATA / 'expert_distribution_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    (OUT / 'expert_stats.json').write_text(json.dumps({
        'n_frames': len(rows), 'n_episodes': len(set(E.tolist())),
        'pearson_grip_vz': float(np.corrcoef(G, V)[0, 1]),
        'spearman_grip_vz': float(np.corrcoef(np.argsort(np.argsort(G)), np.argsort(np.argsort(V)))[0, 1]),
        'grip_max': float(G.max()), 'grip_40_frames': int((G >= 39.5).sum())}, indent=2), encoding='utf-8')
    print(len(rows), len(set(E.tolist())), np.corrcoef(G, V)[0, 1])


if __name__ == '__main__':
    main()
