"""实验 2.7 实验 3：D2 B90 训练数据中夹爪开度、末端高度、阶段覆盖统计。"""
from __future__ import annotations

import csv, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/scratch/lijh/simulate_vla')
SHARD = ROOT / 'data/_shards_e23/B_D2'
OUT = ROOT / 'results/experiment_2_7/d2_training_analysis'
OUT.mkdir(parents=True, exist_ok=True)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except Exception:
        return ImageFont.load_default()


def fk_z(qs):
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True, physicsClientId=c)
    out = []
    for q in qs:
        for j in range(7):
            p.resetJointState(robot, j, float(q[j]), physicsClientId=c)
        out.append(p.getLinkState(robot, 11, computeForwardKinematics=True,
                                  physicsClientId=c)[4][2])
    p.disconnect(c)
    return np.asarray(out)


def main():
    eps = [json.loads(x) for x in (SHARD / 'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
    Q, G, P, E = [], [], [], []
    for e in eps:
        i = e['episode_index']
        d = pq.read_table(SHARD / f'data/chunk-000/episode_{i:06d}.parquet',
                          columns=['observation.state']).to_pydict()
        st = np.asarray(d['observation.state'], float)
        n = len(st)
        Q.append(st[:, :7]); G.append(st[:, 7:9].mean(1))
        P.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, i))
    Q = np.vstack(Q); G = np.concatenate(G); P = np.concatenate(P); E = np.concatenate(E)
    Z = fk_z(Q)
    c3 = json.loads((ROOT / 'results/experiment_2_6/C3_direct_B90/states/G1/C3_snapshot/seed_40000000/c3_terminal.json').read_text())
    cq = np.asarray(c3['q'], float)
    dq = np.linalg.norm(Q - cq, axis=1)
    # bin table
    bins = [(0, 0.016, '<16'), (0.016, 0.025, '16-25'), (0.025, 0.035, '25-35'), (0.035, 1, '>=35')]
    rows = []
    for lo, hi, name in bins:
        m = (G >= lo) & (G < hi)
        if not m.any():
            rows.append({'bin_mm': name, 'n': 0}); continue
        rows.append({'bin_mm': name, 'n': int(m.sum()), 'episodes': len(set(E[m].tolist())),
                     'grip_mean': float(G[m].mean()), 'ee_z_mean': float(Z[m].mean()),
                     'ee_z_std': float(Z[m].std()), 'progress_mean': float(P[m].mean()),
                     'q_to_c3_mean': float(dq[m].mean())})
    with (OUT / 'state_distribution.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # local coverage to C3
    cov = []
    for rad in (0.1, 0.2, 0.3, 0.5, 0.75):
        m = dq <= rad
        cov.append({'radius_rad': rad, 'n_frames': int(m.sum()),
                    'episodes': len(set(E[m].tolist())) if m.any() else 0,
                    'grip_min': float(G[m].min()) if m.any() else None,
                    'grip_max': float(G[m].max()) if m.any() else None,
                    'grip_ge_30': int(((G[m] >= 0.030)).sum()) if m.any() else 0,
                    'grip_ge_35': int(((G[m] >= 0.035)).sum()) if m.any() else 0})
    with (OUT / 'local_coverage.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0].keys())); w.writeheader(); w.writerows(cov)
    # nearest frames
    idx = np.argsort(dq)[:50]
    near = [{'episode': int(E[i]), 'q_distance': float(dq[i]), 'grip_m': float(G[i]),
             'ee_z': float(Z[i]), 'progress': float(P[i])} for i in idx]
    with (OUT / 'nearest_c3_frames.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(near[0].keys())); w.writeheader(); w.writerows(near)
    summary = {'n_frames': len(Q), 'n_episodes': len(eps),
               'grip_min': float(G.min()), 'grip_max': float(G.max()),
               'grip_ge_30_frames': int((G >= 0.030).sum()),
               'grip_ge_35_frames': int((G >= 0.035).sum()),
               'nearest_q': float(dq.min()), 'nearest_episode': int(E[int(np.argmin(dq))]),
               'nearest_grip': float(G[int(np.argmin(dq))]),
               'coverage': cov, 'bins': rows}
    (OUT / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # figures: grip hist + ee z vs progress scatter
    W, H = 1050, 470
    img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
    d.text((40, 18), 'D2 B90: gripper observation distribution', font=font(18), fill='black')
    L, R, T, B = 80, 980, 70, 370
    d.rectangle([L, T, R, B], outline='black')
    hist, edges = np.histogram(G, bins=np.linspace(0.010, 0.040, 31))
    mx = max(hist) * 1.1
    for i, v in enumerate(hist):
        x0 = L + (R - L) * (edges[i] - 0.010) / 0.030
        x1 = L + (R - L) * (edges[i + 1] - 0.010) / 0.030
        h = (B - T) * v / mx
        d.rectangle([x0, B - h, x1 - 1, B], fill='#1f77b4')
    d.text((L, B + 12), 'single-finger observation (m)', font=font(11), fill='black')
    img.save(OUT / 'fig_gripper_hist.png')
    W, H = 1050, 470
    img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
    d.text((40, 18), 'D2 B90: EE height vs normalized progress (color=grip)', font=font(16), fill='black')
    L, R, T, B = 80, 980, 70, 370
    d.rectangle([L, T, R, B], outline='black')
    zmin, zmax = Z.min(), Z.max()
    for i in range(0, len(Q), 4):
        x = L + (R - L) * P[i]
        y = B - (B - T) * (Z[i] - zmin) / (zmax - zmin)
        c = int(255 * min(1, max(0, (G[i] - 0.014) / 0.021)))
        d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255 - c, 80, c))
    d.text((L, B + 12), 'normalized episode progress', font=font(11), fill='black')
    img.save(OUT / 'fig_grip_vs_phase.png')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
