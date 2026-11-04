"""P0（修订版，回应 gate4_review_0e7bb26.md 的两处 P0）：

原版把 `FK(q_target_i) − FK(q_before_i)` 求和当作"命令轨迹位移"，这是**目标偏差**（target error），
不是命令路径；接触标记又是"子步执行之后"测的，与生成该 chunk 的时刻不同。
本版并列报告三种量，并按「chunk 生成时机 × 执行时接触」做 2×2 分层：

  target_error_i        = FK(q_target_i) − FK(q_before_i)      目标相对当前实际末端的偏差
  command_path_delta_i  = FK(q_target_i) − FK(q_target_{i-1})  相邻命令目标之间的路径（跨 chunk 连续）
  realized_path_delta_i = FK(q_after_i) − FK(q_before_i)       实际执行的路径

分层：
  planned_before_contact × contact_now     接近阶段（开环、尚未接触）
  planned_before_contact × contact_now=1   **chunk 生成时还没接触，执行中才接触**（开环无法反应）
  planned_after_contact  × contact_now=0   接触后重新推理、但本步没接触
  planned_after_contact  × contact_now=1   接触后重新推理、本步接触（模型看到过接触状态）

用法（simulate venv）：
  python scripts/exp42/substep_fk_pairing.py --dir results/experiment_4_2/rollouts_p0_logged
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


def vec(s):
    return np.array([float(x) for x in s.split(';')], dtype=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(ROOT / 'results' / 'experiment_4_2' / 'rollouts_p0_logged'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_2' / 'contract_replay_p0c'))
    args = ap.parse_args(argv)
    from exp41 import scene41

    d = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2, robot_q=None)
    sim = scene.sim

    def fk(q):
        import pybullet as p

        for j in range(7):
            p.resetJointState(sim.robot, j, float(q[j]), physicsClientId=sim.client)
        for _ in range(1):
            sim.step()
        pos, _ = sim.pose()
        return np.asarray(pos, dtype=float)

    res = {'dir': str(d), 'definitions': {
        'target_error': 'FK(q_target_i) - FK(q_before_i)（目标偏差，不是命令路径）',
        'command_path_delta': 'FK(q_target_i) - FK(q_target_{i-1})（相邻命令之间的路径）',
        'realized_path_delta': 'FK(q_after_i) - FK(q_before_i)（实际执行路径）',
        'stratification': 'chunk 生成时机（planned_after_first_contact）× 执行时是否有接触',
    }, 'files': []}
    try:
        for f in sorted(d.glob('*_substeps.csv')):
            rows = list(csv.DictReader(open(f, encoding='utf-8')))
            if not rows:
                continue
            touches = [int(float(r['robot_block_contacts'])) > 0
                       or int(float(r['block_wall_contacts'])) > 0 for r in rows]
            first_idx = next((i for i, v in enumerate(touches) if v), None)
            first_tick = int(rows[first_idx]['tick']) if first_idx is not None else None

            rec = []
            prev_target = vec(rows[0]['q_before'])   # 第一个子步之前没有更早的命令目标
            for i, r in enumerate(rows):
                qb, qt, qa = vec(r['q_before']), vec(r['q_target']), vec(r['q_after'])
                ee_b, ee_t, ee_a = fk(qb), fk(qt), fk(qa)
                ee_prev = fk(prev_target)
                tick = int(r['tick'])
                touch = touches[i]
                planned_after = first_tick is not None and tick > first_tick
                group = ('planned_after' if planned_after else 'planned_before') + \
                        ('_contact' if touch else '_free')
                rec.append({
                    'tick': tick, 'k': int(r['k']), 'group': group, 'contact': touch,
                    'rate_limited': r['rate_limited'] == 'True',
                    'target_err': (ee_t - ee_b)[:2].tolist(),
                    'cmd_path': (ee_t - ee_prev)[:2].tolist(),
                    'real_path': (ee_a - ee_b)[:2].tolist(),
                    'obj_d': [float(r['obj_dx']), float(r['obj_dy'])],
                })
                prev_target = qt

            def agg(sel):
                if not sel:
                    return None
                te = np.array([x['target_err'] for x in sel])
                cp = np.array([x['cmd_path'] for x in sel])
                rp = np.array([x['real_path'] for x in sel])
                ob = np.array([x['obj_d'] for x in sel])

                def d(v, axis=0):
                    return [float(v[:, 0].sum()), float(v[:, 1].sum())]
                return {'n': len(sel),
                        'target_error_sum': d(te), 'target_error_abs_mean': float(np.abs(te).mean()),
                        'command_path_sum': d(cp), 'command_path_abs_mean': float(np.abs(cp).mean()),
                        'command_path_frac_positive_x': float((cp[:, 0] > 0).mean()),
                        'realized_path_sum': d(rp), 'realized_path_abs_mean': float(np.abs(rp).mean()),
                        'object_sum': d(ob),
                        'n_rate_limited': int(sum(1 for x in sel if x['rate_limited']))}

            groups = {g: agg([x for x in rec if x['group'] == g])
                      for g in ('planned_before_free', 'planned_before_contact',
                                'planned_after_free', 'planned_after_contact')}
            res['files'].append({
                'file': f.name, 'n_substeps': len(rec),
                'first_contact': None if first_idx is None else {
                    'index': first_idx, 'tick': first_tick, 'k': int(rows[first_idx]['k']),
                    'note': '接触在执行子步之后测量；该 chunk 的生成时刻可能早于接触'},
                'n_contact_substeps': int(sum(touches)),
                'n_contact_bursts': int(sum(1 for i in range(len(touches))
                                            if touches[i] and (i == 0 or not touches[i - 1]))),
                'groups': groups,
            })
            print(f.name)
            for g, v in groups.items():
                if v:
                    print(f"  {g:24s} n={v['n']:4d} "
                          f"cmd_path=({v['command_path_sum'][0]:+.3f},{v['command_path_sum'][1]:+.3f}) "
                          f"real=({v['realized_path_sum'][0]:+.3f},{v['realized_path_sum'][1]:+.3f}) "
                          f"obj=({v['object_sum'][0]:+.3f},{v['object_sum'][1]:+.3f}) "
                          f"frac+x={v['command_path_frac_positive_x']:.2f}")
    finally:
        scene.close()

    (out / 'substep_fk_pairing.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print('wrote', out / 'substep_fk_pairing.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
