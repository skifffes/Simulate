"""Gate 4 任务 2（第 3 步）：分析 `expert_state_diagnostic.npz`。

逐行给出：关节 MAE、方向余弦（只在专家首行非零时算）、幅值比、夹爪差，
并用纯 FK（link 11，与 sim.pose 一致）把关节目标换成 EE 位移方向。
按阶段聚合，同时拆 bias（种子均值）与 variance（种子间标准差）。

用法（simulate venv）：
  python scripts/exp42/expert_state_analysis.py \
    --npz results/experiment_4_2/contract_replay_p0c/expert_state_diagnostic.npz \
    --out results/experiment_4_2/contract_replay_p0c/expert_state_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EE_LINK = 11


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-expert', type=float, default=0.01)
    args = ap.parse_args(argv)
    from exp41 import scene41

    z = np.load(args.npz, allow_pickle=False)
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

    rows = []
    try:
        n = len(z['tick'])
        for i in range(n):
            q = z['q'][i]
            ef, mf = z['expert_first'][i], z['model_first'][i]
            ee0 = fk(q)
            ee_e = fk(q + ef)
            ee_m = fk(q + mf)
            de, dm = (ee_e - ee0)[:2], (ee_m - ee0)[:2]
            rows.append({
                'skill': str(z['skill'][i]), 'episode': int(z['episode'][i]),
                'tick': int(z['tick'][i]), 'phase': str(z['phase'][i]),
                'seed': int(z['seed'][i]),
                'expert_absmax': float(np.abs(ef).max()),
                'model_absmax': float(np.abs(mf).max()),
                'joint_mae': float(np.abs(mf - ef).mean()),
                'cos_joint': float(unit(mf) @ unit(ef)) if np.linalg.norm(ef) > args.min_expert
                             else None,
                'mag_ratio': float(np.linalg.norm(mf) / max(np.linalg.norm(ef), 1e-9)),
                'ee_expert': de.tolist(), 'ee_model': dm.tolist(),
                'ee_cos': float(unit(dm) @ unit(de)) if np.linalg.norm(de) > 1e-4 else None,
                'gripper_expert': float(z['expert_gripper'][i]),
                'gripper_model': float(z['model_gripper'][i]),
                'gripper_err': float(abs(z['model_gripper'][i] - z['expert_gripper'][i])),
            })
    finally:
        scene.close()

    # 按 (skill, phase) 聚合；先按 tick 聚合 seeds 的均值/标准差
    agg = {}
    keys = sorted({(r['skill'], r['phase']) for r in rows})
    for skill, phase in keys:
        sel = [r for r in rows if r['skill'] == skill and r['phase'] == phase]
        ticks = sorted({(r['episode'], r['tick']) for r in sel})
        per_tick = []
        for ep, tk in ticks:
            sub = [r for r in sel if r['episode'] == ep and r['tick'] == tk]
            first = sub[0]
            per_tick.append({
                'expert_absmax': first['expert_absmax'],
                'model_absmax_mean': float(np.mean([s['model_absmax'] for s in sub])),
                'model_absmax_std': float(np.std([s['model_absmax'] for s in sub])),
                'joint_mae_mean': float(np.mean([s['joint_mae'] for s in sub])),
                'cos_joint': (float(np.mean([s['cos_joint'] for s in sub
                                             if s['cos_joint'] is not None]))
                              if any(s['cos_joint'] is not None for s in sub) else None),
                'ee_cos': (float(np.mean([s['ee_cos'] for s in sub
                                          if s['ee_cos'] is not None]))
                           if any(s['ee_cos'] is not None for s in sub) else None),
                'gripper_err': float(np.mean([s['gripper_err'] for s in sub])),
                'mag_ratio_median': float(np.median([s['mag_ratio'] for s in sub])),
            })
        agg[f'{skill}:{phase}'] = {
            'n_states': len(ticks), 'n_rows': len(sel),
            'expert_absmax_median': float(np.median([p['expert_absmax'] for p in per_tick])),
            'model_absmax_median': float(np.median([p['model_absmax_mean'] for p in per_tick])),
            'model_seed_std_median': float(np.median([p['model_absmax_std'] for p in per_tick])),
            'joint_mae_median': float(np.median([p['joint_mae_mean'] for p in per_tick])),
            'cos_joint_median': (float(np.median([p['cos_joint'] for p in per_tick
                                                  if p['cos_joint'] is not None]))
                                 if any(p['cos_joint'] is not None for p in per_tick) else None),
            'ee_cos_median': (float(np.median([p['ee_cos'] for p in per_tick
                                               if p['ee_cos'] is not None]))
                              if any(p['ee_cos'] is not None for p in per_tick) else None),
            'mag_ratio_median': float(np.median([p['mag_ratio_median'] for p in per_tick])),
            'gripper_err_median': float(np.median([p['gripper_err'] for p in per_tick])),
            'per_tick': per_tick,
        }

    res = {'npz': args.npz, 'n_rows': len(rows), 'min_expert_threshold': args.min_expert,
           'by_phase': agg, 'rows': rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    for k, v in agg.items():
        print(f"{k:22s} n={v['n_states']:2d} states  expert={v['expert_absmax_median']:.4f} "
              f"model={v['model_absmax_median']:.4f} (seed std {v['model_seed_std_median']:.4f}) "
              f"mag_ratio={v['mag_ratio_median']:.2f} joint_mae={v['joint_mae_median']:.4f} "
              f"cos={v['cos_joint_median']} ee_cos={v['ee_cos_median']} "
              f"grip_err={v['gripper_err_median']:.3f}")
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
