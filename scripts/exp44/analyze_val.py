"""实验 4.4 离线评测分析：在 val 样本上做分层统计（不依赖人工阶段标签）。

输入：`results/experiment_4_4/offline/{GROUP}_step{STEP}_val.npz`
（由 exp42/expert_state_query.py 生成：q, expert_first, model_first, expert_chunk,
 model_chunk, seed, skill, episode, tick, expert_gripper, model_gripper 等）

输出：`{GROUP}_step{STEP}_val_stats.json`，包含
  * 每技能总体：joint MAE、方向余弦（中位/均值/q10/q90）、幅值比、夹爪开度误差；
  * 分层：夹爪闭合/张开、专家运动幅度三分位、B90 的 EE x 位置分箱（接触前/接触区/墙附近）；
  * 采样方差：同状态不同 seed 的 MAE 标准差（均值）；
  * Δz：EE 高度方向的模型 vs 专家位移差（首行）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EE_LINK = 11


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def fk_batch(qs):
    """对 (N,7) 关节角做 FK，返回 EE (N,3)。"""
    import pybullet as p

    from exp41 import scene41

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2, robot_q=None)
    sim = scene.sim
    out = []
    try:
        for q in qs:
            for j in range(7):
                p.resetJointState(sim.robot, j, float(q[j]), targetVelocity=0.0,
                                  physicsClientId=sim.client)
            out.append(p.getLinkState(sim.robot, EE_LINK, computeForwardKinematics=True,
                                      physicsClientId=sim.client)[4])
    finally:
        scene.close()
    return np.asarray(out, dtype=float)


def summarize(rows, name):
    if not rows:
        return None
    cos = np.array([r['cos'] for r in rows if r['cos'] is not None])
    out = {
        'name': name, 'n_rows': len(rows),
        'n_states': len({(r['skill'], r['episode'], r['tick']) for r in rows}),
        'joint_mae_mean': float(np.mean([r['mae'] for r in rows])),
        'joint_mae_median': float(np.median([r['mae'] for r in rows])),
        'mag_ratio_median': float(np.median([r['mag'] for r in rows])),
        'grip_err_mean': float(np.mean([r['grip_err'] for r in rows])),
        'dz_model_mean_m': float(np.mean([r['dz_model'] for r in rows])),
        'dz_expert_mean_m': float(np.mean([r['dz_expert'] for r in rows])),
        'seed_mae_std_mean': float(np.mean([r['state_seed_std'] for r in rows])),
    }
    if len(cos):
        out.update({'cos_mean': float(cos.mean()), 'cos_median': float(np.median(cos)),
                    'cos_q10': float(np.quantile(cos, .1)),
                    'cos_q90': float(np.quantile(cos, .9)),
                    'frac_cos_gt_0.5': float((cos > 0.5).mean())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=False)
    n = len(z['tick'])
    qs = np.stack([z['q'][i] for i in range(n)]).astype(float)
    ef = np.stack([z['expert_first'][i] for i in range(n)]).astype(float)
    mf = np.stack([z['model_first'][i] for i in range(n)]).astype(float)
    ee_q = fk_batch(qs)
    ee_e = fk_batch(qs + ef[:, :7])
    ee_m = fk_batch(qs + mf[:, :7])

    rows = []
    for i in range(n):
        expert_absmax = float(np.abs(ef[i, :7]).max())
        rows.append({
            'skill': str(z['skill'][i]), 'episode': int(z['episode'][i]),
            'tick': int(z['tick'][i]), 'seed': int(z['seed'][i]),
            'expert_absmax': expert_absmax,
            'mae': float(np.abs(mf[i, :7] - ef[i, :7]).mean()),
            'cos': (float(unit(mf[i, :7]) @ unit(ef[i, :7])) if expert_absmax > 0.01 else None),
            'mag': float(np.linalg.norm(mf[i, :7]) / max(np.linalg.norm(ef[i, :7]), 1e-9)),
            'grip_err': float(abs(float(z['model_gripper'][i]) - float(z['expert_gripper'][i]))),
            'dz_model': float(ee_m[i, 2] - ee_q[i, 2]),
            'dz_expert': float(ee_e[i, 2] - ee_q[i, 2]),
            'grip_closed': bool(float(z['expert_gripper'][i]) < 0.03),
            'ee_x': float(ee_q[i, 0]),
        })
    # 同状态跨 seed 的 MAE 标准差
    groups = {}
    for r in rows:
        groups.setdefault((r['skill'], r['episode'], r['tick']), []).append(r['mae'])
    for r in rows:
        v = groups[(r['skill'], r['episode'], r['tick'])]
        r['state_seed_std'] = float(np.std(v)) if len(v) > 1 else 0.0

    out = {'npz': args.npz, 'n_rows': n,
           'overall': summarize(rows, 'all'),
           'by_skill': {}, 'strata': {}}
    for skill in ('A90', 'B90'):
        sel = [r for r in rows if r['skill'] == skill]
        out['by_skill'][skill] = summarize(sel, skill)
        out['strata'][f'{skill}:gripper_closed'] = summarize(
            [r for r in sel if r['grip_closed']], 'closed')
        out['strata'][f'{skill}:gripper_open'] = summarize(
            [r for r in sel if not r['grip_closed']], 'open')
        ex = np.array([r['expert_absmax'] for r in sel]) if sel else np.array([0.0])
        if len(sel) >= 9:
            qs_ = np.quantile(ex, [1 / 3, 2 / 3])
            out['strata'][f'{skill}:motion_low'] = summarize(
                [r for r in sel if r['expert_absmax'] <= qs_[0]], 'low')
            out['strata'][f'{skill}:motion_high'] = summarize(
                [r for r in sel if r['expert_absmax'] > qs_[1]], 'high')
            out['strata'][f'{skill}:motion_thresholds'] = [float(qs_[0]), float(qs_[1])]
        if skill == 'B90' and sel:
            xs = np.array([r['ee_x'] for r in sel])
            qs2 = np.quantile(xs, [1 / 3, 2 / 3])
            out['strata']['B90:ee_x_low'] = summarize([r for r in sel if r['ee_x'] <= qs2[0]], 'xlow')
            out['strata']['B90:ee_x_high'] = summarize([r for r in sel if r['ee_x'] > qs2[1]],
                                                       'xhigh')
            out['strata']['B90:ee_x_thresholds'] = [float(qs2[0]), float(qs2[1])]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                              encoding='utf-8')
    for k in ('overall',):
        o = out[k]
        print(f"{k}: n={o['n_rows']} states={o['n_states']} MAE={o['joint_mae_mean']:.4f} "
              f"cos={o.get('cos_median', float('nan')):.3f} mag={o['mag_ratio_median']:.2f} "
              f"dz_m={o['dz_model_mean_m']:+.4f} dz_e={o['dz_expert_mean_m']:+.4f} "
              f"seedstd={o['seed_mae_std_mean']:.4f}")
    for k, v in out['by_skill'].items():
        print(f"{k}: MAE={v['joint_mae_mean']:.4f} cos={v.get('cos_median', float('nan')):.3f} "
              f"mag={v['mag_ratio_median']:.2f} dz_m={v['dz_model_mean_m']:+.4f} "
              f"dz_e={v['dz_expert_mean_m']:+.4f} grip={v['grip_err_mean']:.3f} "
              f"seedstd={v['seed_mae_std_mean']:.4f}")
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
