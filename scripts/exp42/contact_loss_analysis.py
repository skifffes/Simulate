"""B90 接触丢失 + A90 接近/对齐诊断（纯 CPU，simulate venv 的 numpy 即可）。

输入：`rollouts_curve1800/*_substeps.csv`（逐子步日志）与同名 JSON 的 action_log。
输出：`contact_loss_analysis.json`，回答：
  * B90 ep0：方块从哪一 tick 起不再前进？那一刻 EE 相对方块的位置/姿态如何？
    接触是否消失？之后命令方向是否仍然 +X？
  * A90：末端是否到达可抓取位姿（3D 距离、X/Y/Z 分量、最近距离出现在哪一 tick）？
    夹爪目标时序如何（是否在到达前就闭合）？

用法：
  python scripts/exp42/contact_loss_analysis.py --dir results/experiment_4_2/rollouts_curve1800 \
      --out results/experiment_4_2/contract_replay_p0c/contact_loss_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def vec(s):
    return np.array([float(x) for x in s.split(';')], dtype=float)


def load(sub_csv: Path):
    rows = list(csv.DictReader(open(sub_csv, encoding='utf-8')))
    ticks = {}
    for r in rows:
        t = int(r['tick'])
        ticks.setdefault(t, []).append(r)
    return ticks


def analyse_b90(ticks):
    seq = []
    for t in sorted(ticks):
        rs = ticks[t]
        first, last = rs[0], rs[-1]
        ee0 = vec(first['ee_before'])
        obj0 = vec(first['obj_before'])
        obj1 = vec(last['obj_after'])
        cont = sum(int(r['robot_block_contacts']) > 0 for r in rs)
        seq.append({
            'tick': t, 'ee': ee0.tolist(), 'obj': obj0.tolist(),
            'obj_end': obj1.tolist(),
            'obj_dx': float(obj1[0] - obj0[0]), 'obj_dy': float(obj1[1] - obj0[1]),
            'rel': (ee0 - obj0).tolist(), 'xy_dist': float(np.linalg.norm((ee0 - obj0)[:2])),
            'dz': float(ee0[2] - obj0[2]), 'contact_substeps': cont,
            'ee_dx': float(sum(float(r['ee_dx']) for r in rs)),
            'ee_dy': float(sum(float(r['ee_dy']) for r in rs)),
            'grip': float(first['gripper_target_m']),
        })
    # 方块最后一次明显前进的 tick
    moving = [s for s in seq if abs(s['obj_dx']) > 0.002]
    last_move = moving[-1]['tick'] if moving else None
    # 停止前的参考姿态（最后 3 个仍在推进的 tick 的平均相对位置）
    ref = [s for s in seq if last_move is not None and s['tick'] <= last_move][-3:]
    ref_rel = np.mean([np.asarray(s['rel'], float) for s in ref], axis=0) if ref else None
    stop = [s for s in seq if last_move is not None and s['tick'] > last_move]
    after_rel = np.mean([np.asarray(s['rel'], float) for s in stop], axis=0) if stop else None
    return {
        'ticks': seq, 'last_push_tick': last_move,
        'ref_rel_when_pushing': ref_rel.tolist() if ref_rel is not None else None,
        'mean_rel_after_stop': after_rel.tolist() if after_rel is not None else None,
        'contact_substeps_after_stop': int(sum(s['contact_substeps'] for s in stop)),
        'ticks_after_stop': len(stop),
        'obj_total_dx': float(seq[-1]['obj_end'][0] - seq[0]['obj'][0]),
    }


def analyse_a90(ticks):
    seq = []
    for t in sorted(ticks):
        rs = ticks[t]
        first = rs[0]
        ee = vec(first['ee_before'])
        obj = vec(first['obj_before'])
        d = ee - obj
        seq.append({'tick': t, 'dist3d': float(np.linalg.norm(d)),
                    'dx': float(d[0]), 'dy': float(d[1]), 'dz': float(d[2]),
                    'grip': float(first['gripper_target_m']),
                    'ee': ee.tolist(), 'obj': obj.tolist(),
                    'contact_substeps': sum(int(r['robot_block_contacts']) > 0 for r in rs)})
    best = min(seq, key=lambda s: s['dist3d'])
    return {
        'ticks': seq,
        'min_dist3d': best['dist3d'], 'min_dist_tick': best['tick'],
        'min_dx_dy_dz': [best['dx'], best['dy'], best['dz']],
        'final_dist3d': seq[-1]['dist3d'],
        'grip_min': float(min(s['grip'] for s in seq)),
        'grip_max': float(max(s['grip'] for s in seq)),
        'grip_at_min_dist': best['grip'],
        'ever_contact': any(s['contact_substeps'] > 0 for s in seq),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/experiment_4_2/rollouts_curve1800')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    out = {'b90': {}, 'a90': {}}
    for ep in (0, 1):
        p = d / f'B90_ep{ep:02d}_substeps.csv'
        if p.exists():
            out['b90'][f'ep{ep}'] = analyse_b90(load(p))
    for ep in (0, 1):
        p = d / f'A90_ep{ep:02d}_substeps.csv'
        if p.exists():
            out['a90'][f'ep{ep}'] = analyse_a90(load(p))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                              encoding='utf-8')

    for k, v in out['b90'].items():
        print(f"B90 {k}: last_push_tick={v['last_push_tick']} obj_total_dx={v['obj_total_dx']:.3f} "
              f"rel_when_pushing={np.round(v['ref_rel_when_pushing'], 3).tolist()} "
              f"rel_after={np.round(v['mean_rel_after_stop'], 3).tolist() if v['mean_rel_after_stop'] else None} "
              f"contact_after={v['contact_substeps_after_stop']}/{v['ticks_after_stop']}")
    for k, v in out['a90'].items():
        print(f"A90 {k}: min_dist3d={v['min_dist3d']:.3f} @tick {v['min_dist_tick']} "
              f"(dx,dy,dz)={np.round(v['min_dx_dy_dz'], 3).tolist()} final={v['final_dist3d']:.3f} "
              f"grip[{v['grip_min']:.3f},{v['grip_max']:.3f}] ever_contact={v['ever_contact']}")
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
