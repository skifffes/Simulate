"""实验 4.6：首次不可恢复偏离分析。

输入：
  * results/experiment_4_6/rollouts/step{STEP}/off*/{skill}_ep{NN}_substeps.csv
  * /scratch/lijh/exp46/export_overfit 的专家 state（FK 得到专家 EE 轨迹）
输出：
  * results/experiment_4_6/first_deviation.json / .csv
判据：在第 t 个控制 tick，|EE_rollout(t) − EE_expert(t)| > 3 cm，
且此后至少 80% 的 tick 仍 > 1 cm → 记为首次不可恢复偏离。
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

DATA = Path('/scratch/lijh/exp46/export_overfit')
L = Path('/scratch/lijh/exp40/repo/results/experiment_4_6')
EE_LINK = 11


RECORDS = {}


def load_records():
    import glob
    import json as _json
    for f in glob.glob('/scratch/lijh/exp40/repo/results/experiment_4_1/dataset/records/*.json'):
        r = _json.load(open(f, encoding='utf-8'))
        if r.get('stored_episode_index') is not None:
            RECORDS[(r['skill'], int(r['stored_episode_index']))] = r


def robot_start(rec):
    from exp41 import scene41
    cid = rec.get('config_id') or ''
    parts = cid.split('|')
    rid = parts[2] if len(parts) == 3 else scene41.B_ROBOT_STARTS[0]['robot_start_id']
    return next(r for r in scene41.B_ROBOT_STARTS if r['robot_start_id'] == rid)


def fk_batch(qs, robot=None, obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2):
    import pybullet as p

    from exp41 import scene41

    scene = scene41.Scene41('B90', obj_pos=obj_pos, obj_yaw=obj_yaw,
                            robot_q=(robot['q'] if robot else None),
                            gripper=(robot['gripper'] if robot else None))
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


def expert_ee(skill, ep):
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    row = next(r for r in meta if r['skill'] == skill and int(r['episode_index']) == ep)
    z = np.load(DATA / row['state_path'])
    rec = RECORDS.get((skill, ep))
    robot = robot_start(rec) if rec else None
    obj = row.get('start_object') or {}
    pos = tuple(obj['pos']) if 'pos' in obj else (0.55, 0.21, 0.0201)
    yaw = float(obj.get('yaw', 3.141592653589793))
    return fk_batch(z['state'][:, :7], robot=robot, obj_pos=pos, obj_yaw=yaw)


def main() -> int:
    load_records()
    rows = []
    for d in sorted((L / 'rollouts').glob('step*')):
        step = int(d.name.replace('step', ''))
        meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
                if l.strip()]
        for f in sorted(d.glob('off*/*.json')):
            if f.name.endswith('_summary.json'):
                continue
            j = json.loads(f.read_text(encoding='utf-8'))
            skill, ep = j['skill'], int(j['episode'])
            sub = f.with_name(f.stem + '_substeps.csv')
            if not sub.exists():
                continue
            ee_exp = expert_ee(skill, ep)
            sub_rows = list(csv.DictReader(open(sub, encoding='utf-8')))
            by_tick = {}
            for r in sub_rows:
                by_tick.setdefault(int(r['tick']), []).append(r)
            devs = []
            for t in sorted(by_tick):
                if t >= len(ee_exp):
                    break
                ee = [float(x) for x in by_tick[t][0]['ee_before'].split(';')]
                devs.append((t, float(np.linalg.norm(np.asarray(ee) - ee_exp[t]))))
            first = None
            for i, (t, dv) in enumerate(devs):
                if dv > 0.03:
                    rest = [x[1] for x in devs[i:]]
                    if rest and sum(1 for x in rest if x > 0.01) / len(rest) >= 0.8:
                        first = t
                        break
            rows.append({'step': step, 'skill': skill, 'episode': ep,
                         'seed_offset': int(re.search(r'off(\d+)', str(f)).group(1)),
                         'success': j.get('strict_success'),
                         'failure_reason': j.get('failure_reason'),
                         'max_dev_m': max((x[1] for x in devs), default=None),
                         'first_unrecoverable_dev_tick': first,
                         'n_ticks': len(devs)})
    keys = ['step', 'skill', 'episode', 'seed_offset', 'success', 'failure_reason',
            'max_dev_m', 'first_unrecoverable_dev_tick', 'n_ticks']
    with open(L / 'first_deviation.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (L / 'first_deviation.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=1)[:1500])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
