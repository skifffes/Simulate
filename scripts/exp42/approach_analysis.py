"""补充诊断（CPU）：B90 ep1 为何 0 接触；A90/B90 在专家状态上预测的**垂直方向(Δz)**是否收敛。

1) B90 ep1：逐 tick 的 EE–物体 3D 距离、最近点、X 方向是否有接近；
2) 用 `expert_state_curve1800.npz`（31 唯一状态 × 4 seed）：对每行算
   FK(q+model_first) − FK(q) 与 FK(q+expert_first) − FK(q)，
   按阶段给出 Δz 的模型/专家均值 —— 检查"是否学会下降/保持高度"。

用法（simulate venv）：
  python scripts/exp42/approach_analysis.py \
    --rollouts results/experiment_4_2/rollouts_curve1800 \
    --npz results/experiment_4_2/contract_replay_p0c/expert_state_curve1800.npz \
    --out results/experiment_4_2/contract_replay_p0c/approach_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
EE_LINK = 11


def vec(s):
    return np.array([float(x) for x in s.split(';')], dtype=float)


def b90_ep1(path: Path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    ticks = {}
    for r in rows:
        ticks.setdefault(int(r['tick']), []).append(r)
    seq = []
    for t in sorted(ticks):
        rs = ticks[t]
        ee = vec(rs[0]['ee_before'])
        obj = vec(rs[0]['obj_before'])
        d = ee - obj
        seq.append({'tick': t, 'dist3d': float(np.linalg.norm(d)),
                    'dx': float(d[0]), 'dy': float(d[1]), 'dz': float(d[2]),
                    'ee_x': float(ee[0]), 'obj_x': float(obj[0]),
                    'contact': sum(int(r['robot_block_contacts']) > 0 for r in rs)})
    best = min(seq, key=lambda s: s['dist3d'])
    return {'min_dist3d': best['dist3d'], 'min_tick': best['tick'],
            'min_dxyz': [best['dx'], best['dy'], best['dz']],
            'ee_x_start': seq[0]['ee_x'], 'ee_x_end': seq[-1]['ee_x'],
            'obj_x_start': seq[0]['obj_x'], 'obj_x_end': seq[-1]['obj_x'],
            'ever_contact': any(s['contact'] > 0 for s in seq),
            'ticks': seq}


def ee_deltas(npz_path: Path):
    from exp41 import scene41

    z = np.load(npz_path, allow_pickle=False)
    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2, robot_q=None)
    sim = scene.sim

    def fk(q):
        import pybullet as p

        for j in range(7):
            p.resetJointState(sim.robot, j, float(q[j]), targetVelocity=0.0,
                              physicsClientId=sim.client)
        link = p.getLinkState(sim.robot, EE_LINK, computeForwardKinematics=True,
                              physicsClientId=sim.client)
        return np.asarray(link[4], dtype=float)

    out = {}
    try:
        n = len(z['tick'])
        rows = []
        for i in range(n):
            q = z['q'][i]
            e0 = fk(q)
            dm = fk(q + z['model_first'][i]) - e0
            de = fk(q + z['expert_first'][i]) - e0
            rows.append({'skill': str(z['skill'][i]), 'phase': str(z['phase'][i]),
                         'model_dz': float(dm[2]), 'expert_dz': float(de[2]),
                         'model_dx': float(dm[0]), 'expert_dx': float(de[0]),
                         'model_dy': float(dm[1]), 'expert_dy': float(de[1])})
    finally:
        scene.close()
    keys = sorted({(r['skill'], r['phase']) for r in rows})
    for skill, phase in keys:
        sel = [r for r in rows if r['skill'] == skill and r['phase'] == phase]
        out[f'{skill}:{phase}'] = {
            'n_rows': len(sel),
            'model_dz_mean': float(np.mean([r['model_dz'] for r in sel])),
            'expert_dz_mean': float(np.mean([r['expert_dz'] for r in sel])),
            'model_dx_mean': float(np.mean([r['model_dx'] for r in sel])),
            'expert_dx_mean': float(np.mean([r['expert_dx'] for r in sel])),
            'model_dy_mean': float(np.mean([r['model_dy'] for r in sel])),
            'expert_dy_mean': float(np.mean([r['expert_dy'] for r in sel])),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', default=str(ROOT / 'results' / 'experiment_4_2' / 'rollouts_curve1800'))
    ap.add_argument('--npz', default=str(ROOT / 'results' / 'experiment_4_2' / 'contract_replay_p0c' /
                                         'expert_state_curve1800.npz'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    res = {'b90_ep1': None, 'ee_dz_by_phase': None}
    p = Path(args.rollouts) / 'B90_ep01_substeps.csv'
    if p.exists():
        res['b90_ep1'] = b90_ep1(p)
        v = res['b90_ep1']
        print(f"B90 ep1: min_dist3d={v['min_dist3d']:.3f} @tick {v['min_tick']} "
              f"dxyz={np.round(v['min_dxyz'],3).tolist()} "
              f"ee_x {v['ee_x_start']:.3f}->{v['ee_x_end']:.3f} "
              f"obj_x {v['obj_x_start']:.3f}->{v['obj_x_end']:.3f} contact={v['ever_contact']}")
    if Path(args.npz).exists():
        res['ee_dz_by_phase'] = ee_deltas(Path(args.npz))
        for k, v in res['ee_dz_by_phase'].items():
            print(f"{k:22s} dz model={v['model_dz_mean']:+.4f} expert={v['expert_dz_mean']:+.4f} | "
                  f"dx model={v['model_dx_mean']:+.4f} expert={v['expert_dx_mean']:+.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                              encoding='utf-8')
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
