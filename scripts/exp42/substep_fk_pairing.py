"""P0：逐执行子步配对（命令 EE vs 实际 EE vs 物体），并按阶段切开。

输入：`eval_rollouts.py` 生成的 `*_substeps.csv`（每 tick 前 `execute_k` 行）。
输出：`substep_fk_pairing.json` —— 每个阶段（接触前 / 接触中 / 接触后）的
      命令 EE 位移（FK(q_target) − FK(q_before)）、实际 EE 位移（FK(q_after) − FK(q_before)）、
      物体位移、方向余弦、限速/拒绝计数。

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

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2,
                            robot_q=None)
    sim = scene.sim

    def fk(q):
        for j in range(7):
            sim.command(np.asarray(q[:7], dtype=float))
        for j in range(7):
            import pybullet as p

            p.resetJointState(sim.robot, j, float(q[j]), physicsClientId=sim.client)
        pos, _ = sim.pose()
        return np.asarray(pos, dtype=float)

    res = {'dir': str(d), 'files': []}
    try:
        for f in sorted(d.glob('*_substeps.csv')):
            rows = list(csv.DictReader(open(f, encoding='utf-8')))
            if not rows:
                continue
            rec = []
            seen_contact = False
            contact_ended = False
            for r in rows:
                qb = vec(r['q_before'])
                qt = vec(r['q_target'])
                qa = vec(r['q_after'])
                ee_b = fk(qb)
                ee_c = fk(qt)
                ee_a = fk(qa)
                touch = (r['robot_block_contacts'] not in ('0', '')) or \
                        (r['block_wall_contacts'] not in ('0', ''))
                if touch:
                    seen_contact = True
                elif seen_contact:
                    contact_ended = True
                phase = 'contact' if touch else ('post_contact' if contact_ended else 'pre_contact')
                rec.append({
                    'tick': int(r['tick']), 'k': int(r['k']), 'phase': phase,
                    'accepted': r['accepted'] == 'True',
                    'rate_limited': r['rate_limited'] == 'True',
                    'cmd_jump_rad': float(r['cmd_jump_rad']),
                    'q_track_err': float(r['q_track_err']),
                    'cmd_dx': float(ee_c[0] - ee_b[0]), 'cmd_dy': float(ee_c[1] - ee_b[1]),
                    'cmd_dz': float(ee_c[2] - ee_b[2]),
                    'act_dx': float(ee_a[0] - ee_b[0]), 'act_dy': float(ee_a[1] - ee_b[1]),
                    'act_dz': float(ee_a[2] - ee_b[2]),
                    'obj_dx': float(r['obj_dx']), 'obj_dy': float(r['obj_dy']),
                    'contacts': r['robot_block_contacts'],
                })

            def agg(sel):
                if not sel:
                    return None
                cmd = np.array([[x['cmd_dx'], x['cmd_dy']] for x in sel])
                act = np.array([[x['act_dx'], x['act_dy']] for x in sel])
                obj = np.array([[x['obj_dx'], x['obj_dy']] for x in sel])
                def unit(v):
                    n = np.linalg.norm(v)
                    return v / n if n > 1e-12 else v
                cos_ca = float(np.mean([unit(c) @ unit(a) if np.linalg.norm(c) > 1e-9
                                        and np.linalg.norm(a) > 1e-9 else np.nan
                                        for c, a in zip(cmd, act)]))
                return {
                    'n': len(sel),
                    'cmd_sum_dx': float(cmd[:, 0].sum()), 'cmd_sum_dy': float(cmd[:, 1].sum()),
                    'act_sum_dx': float(act[:, 0].sum()), 'act_sum_dy': float(act[:, 1].sum()),
                    'obj_sum_dx': float(obj[:, 0].sum()), 'obj_sum_dy': float(obj[:, 1].sum()),
                    'cmd_frac_positive_x': float((cmd[:, 0] > 0).mean()),
                    'act_frac_positive_x': float((act[:, 0] > 0).mean()),
                    'cos_cmd_vs_act': cos_ca,
                    'mean_q_track_err': float(np.mean([x['q_track_err'] for x in sel])),
                    'max_cmd_jump_rad': float(max(x['cmd_jump_rad'] for x in sel)),
                    'n_rate_limited': int(sum(1 for x in sel if x['rate_limited'])),
                    'n_rejected': int(sum(1 for x in sel if not x['accepted'])),
                }
            phases = {ph: agg([x for x in rec if x['phase'] == ph])
                      for ph in ('pre_contact', 'contact', 'post_contact')}
            res['files'].append({'file': f.name, 'n_substeps': len(rec), 'phases': phases})
            print(f'{f.name}: ' + ' | '.join(
                f"{ph}: n={phases[ph]['n']} cmd=({phases[ph]['cmd_sum_dx']:+.3f},"
                f"{phases[ph]['cmd_sum_dy']:+.3f}) act=({phases[ph]['act_sum_dx']:+.3f},"
                f"{phases[ph]['act_sum_dy']:+.3f}) obj=({phases[ph]['obj_sum_dx']:+.3f},"
                f"{phases[ph]['obj_sum_dy']:+.3f})" if phases[ph] else f'{ph}: n=0'
                for ph in ('pre_contact', 'contact')))
    finally:
        scene.close()

    (out / 'substep_fk_pairing.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print('wrote', out / 'substep_fk_pairing.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
