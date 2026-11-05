"""原始分布评测：整条 episode 采样后，比较模型 chunk 与专家 chunk（关节空间 + 分层）。

指标：
  * 首行：MAE、方向余弦、幅值比；
  * 逐 k（0..15）：MAE；
  * 分布统计：cos 的 mean/median/q10/q90、cos>0.5 比例、cos<0 比例；
  * 分层：按专家首行 |Δq| 三分位（静止/中等/快速）与夹爪开合，各层的 cos/MAE/幅值比。

用法：
  python scripts/exp42/dist_eval.py --npz <query 输出> --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def summarize(sel, name):
    if not sel:
        return None
    cs = np.array([x['cos'] for x in sel if x['cos'] is not None])
    mag = np.array([x['mag'] for x in sel])
    mae = np.array([x['mae'] for x in sel])
    return {
        'name': name, 'n': len(sel),
        'n_with_expert_motion': int(len(cs)),
        'cos_mean': float(cs.mean()) if len(cs) else None,
        'cos_median': float(np.median(cs)) if len(cs) else None,
        'cos_q10': float(np.quantile(cs, .1)) if len(cs) else None,
        'cos_q90': float(np.quantile(cs, .9)) if len(cs) else None,
        'frac_cos_gt_0.5': float((cs > 0.5).mean()) if len(cs) else None,
        'frac_cos_lt_0': float((cs < 0).mean()) if len(cs) else None,
        'mag_ratio_mean': float(mag.mean()), 'mag_ratio_median': float(np.median(mag)),
        'mae_mean': float(mae.mean()), 'mae_median': float(np.median(mae)),
        'expert_absmax_mean': float(np.mean([x['expert_absmax'] for x in sel])),
        'model_absmax_mean': float(np.mean([x['model_absmax'] for x in sel])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    args = ap.parse_args(argv)
    z = np.load(args.npz, allow_pickle=False)
    n = len(z['tick'])
    rows = []
    for i in range(n):
        q = z['q'][i]
        ef, mf = z['expert_first'][i], z['model_first'][i]
        expert_absmax = float(np.abs(ef).max())
        rows.append({
            'skill': str(z['skill'][i]), 'episode': int(z['episode'][i]),
            'tick': int(z['tick'][i]), 'seed': int(z['seed'][i]),
            'expert_absmax': expert_absmax,
            'model_absmax': float(np.abs(mf).max()),
            'mae': float(np.abs(mf - ef).mean()),
            'cos': float(unit(mf) @ unit(ef)) if expert_absmax > 0.01 else None,
            'mag': float(np.linalg.norm(mf) / max(np.linalg.norm(ef), 1e-9)),
            'grip_expert': float(z['expert_gripper'][i]),
            'grip_model': float(z['model_gripper'][i]),
        })

    out = {'npz': args.npz, 'tag': args.tag, 'n_rows': n,
           'overall': summarize(rows, 'all')}

    # 逐 k 的 MAE
    per_k = []
    for k in range(16):
        d = np.abs(z['model_chunk'][:, k, :7] - z['expert_chunk'][:, k, :7])
        per_k.append({'k': k, 'mae': float(d.mean()),
                      'model_absmax': float(np.abs(z['model_chunk'][:, k, :7]).max()),
                      'expert_absmax': float(np.abs(z['expert_chunk'][:, k, :7]).max())})
    out['per_k'] = per_k

    # 按专家首行幅度三分位分层
    ex = np.array([r['expert_absmax'] for r in rows])
    if len(rows) >= 9:
        qs = np.quantile(ex, [1 / 3, 2 / 3])
        out['by_expert_motion'] = {
            'low': summarize([r for r in rows if r['expert_absmax'] <= qs[0]], 'low'),
            'mid': summarize([r for r in rows if qs[0] < r['expert_absmax'] <= qs[1]], 'mid'),
            'high': summarize([r for r in rows if r['expert_absmax'] > qs[1]], 'high'),
            'thresholds': [float(qs[0]), float(qs[1])],
        }
    # 按夹爪开合分层
    out['by_gripper'] = {
        'closed': summarize([r for r in rows if r['grip_expert'] < 0.5], 'closed'),
        'open': summarize([r for r in rows if r['grip_expert'] >= 0.5], 'open'),
    }
    out['rows'] = rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                              encoding='utf-8')
    o = out['overall']
    print(f"n={o['n']} | MAE {o['mae_mean']:.4f} | mag_ratio {o['mag_ratio_median']:.2f} | "
          f"cos mean {o['cos_mean']:.3f} median {o['cos_median']:.3f} "
          f"q10 {o['cos_q10']:.2f} q90 {o['cos_q90']:.2f} | "
          f"cos>0.5 {o['frac_cos_gt_0.5']:.2f} cos<0 {o['frac_cos_lt_0']:.2f}")
    for name, grp in (out.get('by_expert_motion') or {}).items():
        if isinstance(grp, dict) and grp.get('cos_mean') is not None:
            print(f"  motion {name:5s} n={grp['n']:4d} cos {grp['cos_mean']:+.2f} "
                  f"MAE {grp['mae_mean']:.4f} mag {grp['mag_ratio_median']:.2f}")
    for name, grp in out['by_gripper'].items():
        if grp:
            print(f"  gripper {name:6s} n={grp['n']:4d} cos "
                  f"{grp['cos_mean'] if grp['cos_mean'] is None else round(grp['cos_mean'],2)} "
                  f"MAE {grp['mae_mean']:.4f}")
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
