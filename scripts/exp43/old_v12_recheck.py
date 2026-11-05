"""旧 90 条示范在 **exp4.1-strict-v1.2** 下的复核（专家目标直发，不经过 Adapter）。

对 export meta 里所有 `episode_index < 1000` 的旧 train episode：
  用 records 恢复场景（start_object、goal、机器人构型），把数据集的绝对关节目标 A[t]
  与夹爪米值逐帧下发（20 Hz × 12 physics steps），最后用当前严格评价器判定。

输出：results/experiment_4_3/data_expansion/old_v12_recheck.csv / .json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path('/scratch/lijh/exp40/repo')
DATA = Path('/scratch/lijh/exp42/data')
OUT = ROOT / 'results/experiment_4_3/data_expansion'
STEPS_PER_TICK = 12


def main() -> int:
    import pybullet as p

    from exp41 import scene41
    from exp41.strict_eval import (DEFAULT_CONFIG, PhysicsMonitor, evaluate_A, evaluate_B,
                                   evaluate_initial_B)

    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    recs = {}
    for f in sorted((ROOT / 'results' / 'experiment_4_1' / 'dataset' / 'records').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('stored_episode_index') is not None:
            recs[(r['skill'], int(r['stored_episode_index']))] = r

    rows = []
    for m in meta:
        if m.get('split') != 'train' or int(m['episode_index']) >= 1000:
            continue
        skill, ep = m['skill'], int(m['episode_index'])
        rec = recs.get((skill, ep))
        if rec is None:
            rows.append({'skill': skill, 'episode': ep, 'status': 'no_record'})
            continue
        start = rec['start_object']
        if skill == 'A90':
            gx, gy = rec['goal_point']
            goal = scene41.AGoal(rec['goal']['goal_id'], gx, gy,
                                 rec['goal'].get('target_axis', 'x'),
                                 rec['goal'].get('region_id', 'A_place'))
            scene = scene41.Scene41('A90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                                    robot_q=None)
            initial = None
        else:
            goal = next(g for g in scene41.B_GOALS if g.goal_id == rec['goal']['goal_id'])
            rs = next(r for r in scene41.B_ROBOT_STARTS
                      if r['robot_start_id'] == (rec.get('config_id') or '').split('|')[-1])
            scene = scene41.Scene41('B90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                                    robot_q=rs['q'], gripper=rs['gripper'])
            initial = evaluate_initial_B(scene, DEFAULT_CONFIG)
        mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
        start_obj = scene.object_state()
        z = np.load(DATA / m['state_path'])
        A = z['action'].astype(float)
        try:
            for t in range(len(A)):
                scene.sim.command(A[t, :7])
                for j in (9, 10):
                    p.setJointMotorControl2(scene.sim.robot, j, p.POSITION_CONTROL,
                                            float(A[t, 7]), force=30,
                                            physicsClientId=scene.sim.client)
                for _ in range(STEPS_PER_TICK):
                    scene.sim.step()
            for _ in range(int(0.4 / scene.sim.dt)):
                scene.sim.step()
            if skill == 'A90':
                ev = evaluate_A(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start_obj['pos'])
            else:
                ev = evaluate_B(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start_obj['pos'],
                                initial={**initial, 'yaw': start['yaw']})
        finally:
            mon.detach()
            scene.close()
        rows.append({'skill': skill, 'episode': ep, 'status': 'ok',
                     'strict_success': bool(ev['strict_success']),
                     'failure_reason': ev['failure_reason'],
                     'n_frames': int(len(A))})
        print(f'{skill} ep{ep}: success={ev["strict_success"]} reason={ev["failure_reason"]}',
              flush=True)

    with open(OUT / 'old_v12_recheck.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {'n': len(rows),
               'by_skill': {s: {'n': sum(1 for r in rows if r['skill'] == s),
                                 'success': sum(1 for r in rows
                                                if r['skill'] == s and r.get('strict_success'))}
                            for s in ('A90', 'B90')},
               'failures': {}}
    for r in rows:
        if r.get('status') == 'ok' and not r.get('strict_success'):
            summary['failures'][r['failure_reason']] = \
                summary['failures'].get(r['failure_reason'], 0) + 1
    (OUT / 'old_v12_recheck.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
