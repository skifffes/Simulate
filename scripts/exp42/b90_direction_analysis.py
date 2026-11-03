"""Gate 2：分离 B90 的三层运动方向。

  L1 预测：模型 chunk 第一行（7 维关节偏移）经当前构型 Jacobian 映射成 EE 位移趋势
  L2 实际：rollout 日志里的实测 EE 位移
  L3 物体：rollout 日志里的物体位移（含接触阶段）

同时给出专家参考：数据集中同一阶段（tick 0..20）的 EE 位移（FK(A[t]) − FK(S[t])）。

用法（simulate venv）：
  python scripts/exp42/b90_direction_analysis.py --rollout-json results/experiment_4_2/rollouts_fixed_contract2/B90_ep00.json
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

DATA = Path('/scratch/lijh/exp42/data')
EE_LINK = 6


def fk_ee(q):
    import pybullet as p

    from exp41 import scene41

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2,
                            robot_q=None)
    try:
        for j in range(7):
            p.resetJointState(scene.sim.robot, j, float(q[j]), physicsClientId=scene.sim.client)
        for _ in range(2):
            scene.sim.step()
        pos, quat = scene.sim.pose()
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
    finally:
        scene.close()


def jacobian_linear(q):
    import pybullet as p

    from exp41 import scene41

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2,
                            robot_q=None)
    try:
        for j in range(7):
            p.resetJointState(scene.sim.robot, j, float(q[j]), physicsClientId=scene.sim.client)
        for _ in range(2):
            scene.sim.step()
        ee_pos, _ = scene.sim.pose()
        qs = [float(p.getJointState(scene.sim.robot, j,
                                    physicsClientId=scene.sim.client)[0]) for j in range(7)]
        qs += [0.0, 0.0]
        zeros = [0.0] * len(qs)
        lin, _ang = p.calculateJacobian(scene.sim.robot, EE_LINK, [0.0, 0.0, 0.0],
                                        qs, zeros, zeros, physicsClientId=scene.sim.client)
        J = np.asarray(lin, dtype=float).reshape(3, -1)[:, :7]
        return J, np.asarray(ee_pos, dtype=float)
    finally:
        scene.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollout-json', required=True)
    ap.add_argument('--expert-episodes', default='0,1')
    ap.add_argument('--stage-ticks', type=int, default=21)
    ap.add_argument('--out', default='results/experiment_4_2/contract_replay_p0c')
    args = ap.parse_args(argv)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    roll = json.loads(Path(args.rollout_json).read_text(encoding='utf-8'))
    log = roll['action_log'][:args.stage_ticks]
    pred = []
    for r in log:
        q = np.array([float(x) for x in r['anchor_q'].split(';')])
        dq = np.array([float(x) for x in r['chunk_first_row'].split(';')])[:7]
        J, ee_pos = jacobian_linear(q)
        pred.append((J @ dq)[:3])
    pred = np.asarray(pred)
    actual = np.asarray([[r['ee_dx'], r['ee_dy'], r['ee_dz']] for r in log])
    objs = np.asarray([[r['obj_dx'], r['obj_dy'], 0.0] for r in log])

    # 专家参考：同阶段（tick 0..20）的 FK(A[t]) - FK(S[t])
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    expert = []
    for ep in [int(x) for x in args.expert_episodes.split(',')]:
        row = next(m for m in meta if m['skill'] == 'B90' and m['episode_index'] == ep)
        sa = np.load(DATA / row['state_path'])
        S, A = sa['state'].astype(float), sa['action'].astype(float)
        for t in range(min(args.stage_ticks, S.shape[0])):
            e0, _ = fk_ee(S[t, :7])
            e1, _ = fk_ee(A[t, :7])
            expert.append(e1 - e0)
    expert = np.asarray(expert)

    def summ(x, name):
        mean = x.mean(axis=0)
        return {'name': name, 'n': int(x.shape[0]),
                'mean_dx': float(mean[0]), 'mean_dy': float(mean[1]), 'mean_dz': float(mean[2]),
                'sum_dx': float(x[:, 0].sum()), 'sum_dy': float(x[:, 1].sum()),
                'frac_positive_x': float((x[:, 0] > 0).mean()),
                'abs_dx': float(np.abs(x[:, 0]).sum()), 'abs_dy': float(np.abs(x[:, 1]).sum()),
                'y_over_x_ratio': float(x[:, 1].sum() / max(abs(x[:, 0].sum()), 1e-9))}

    res = {
        'rollout': str(args.rollout_json), 'skill': roll['skill'], 'episode': roll['episode'],
        'strict_success': roll['strict_success'], 'failure_reason': roll['failure_reason'],
        'n_ticks_analyzed': len(log),
        'L1_predicted_EE': summ(pred, 'predicted (Jacobian @ chunk row 0)'),
        'L2_actual_EE': summ(actual, 'actual measured EE'),
        'L3_object': summ(objs, 'object displacement'),
        'expert_reference_EE': summ(expert, 'expert FK(A)-FK(S), episodes '
                                         + args.expert_episodes),
        'contact_ticks': int(sum(1 for r in log if r.get('robot_block_contacts', 0) > 0)),
        'verdict': None,
    }
    p_dir = np.array([res['L1_predicted_EE']['mean_dx'], res['L1_predicted_EE']['mean_dy']])
    e_dir = np.array([res['expert_reference_EE']['mean_dx'], res['expert_reference_EE']['mean_dy']])
    a_dir = np.array([res['L2_actual_EE']['mean_dx'], res['L2_actual_EE']['mean_dy']])
    o_dir = np.array([res['L3_object']['mean_dx'], res['L3_object']['mean_dy']])

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def cos(u, v):
        return float(unit(u) @ unit(v))

    res['cos'] = {'pred_vs_expert': cos(p_dir, e_dir),
                  'actual_vs_predicted': cos(a_dir, p_dir),
                  'actual_vs_expert': cos(a_dir, e_dir),
                  'object_vs_expert': cos(o_dir, e_dir)}
    if cos(p_dir, e_dir) < 0.5:
        res['verdict'] = '模型自身预测的 EE 方向就偏离专家（优先查条件动作/标签/归一化）'
    elif cos(a_dir, e_dir) < 0.5 and cos(p_dir, e_dir) >= 0.5:
        res['verdict'] = '预测方向正确但执行/接触把它带偏（优先查接触几何与控制器跟踪）'
    else:
        res['verdict'] = '预测与实际都接近专家；物体偏离来自接触过程'
    (out / f"b90_direction_{roll['skill']}_ep{roll['episode']:02d}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
