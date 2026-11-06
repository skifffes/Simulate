"""实验 4.7 Gate 3：四层执行的物理一致性。

模式 A  ：逐周期原始专家回放（直接执行 A[t]）。
模式 B1 ：Oracle 绝对目标 chunk（用当前真实 q 重锚定 A[t+k]），每 chunk 执行 8 步。
模式 B2 ：Oracle 原始训练标签（A[t+k] − S_recorded[t]）在当前 q 上重锚定，执行 8 步。

每条 episode 记录：q_before、target、controller command、q_after、跟踪误差、夹爪、EE、物体、
机器人-方块接触、方块-墙接触，并进行严格 v1.2 判定。
安全消融（模式 A，关闭限速/裁剪）作为对照单独标记。
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/scratch/lijh/exp40/repo')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

OUT = REPO / 'results' / 'experiment_4_7'
DATA = Path('/scratch/lijh/exp46/export_overfit')
STEPS_PER_TICK = 12
HORIZON = 16
EXEC_K = 8


def load_records():
    recs = {}
    for f in (REPO / 'results' / 'experiment_4_1' / 'dataset' / 'records').glob('*.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('stored_episode_index') is not None:
            recs[(r['skill'], int(r['stored_episode_index']))] = r
    return recs


def build_scene(rec, scene41):
    start = rec['start_object']
    if rec['skill'] == 'A90':
        gx, gy = rec['goal_point']
        goal = scene41.AGoal(rec['goal']['goal_id'], gx, gy,
                             rec['goal'].get('target_axis', 'x'),
                             rec['goal'].get('region_id', 'A_place'))
        scene = scene41.Scene41('A90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                                robot_q=None)
        return scene, goal, None
    goal = next(g for g in scene41.B_GOALS if g.goal_id == rec['goal']['goal_id'])
    rid = (rec.get('config_id') or '').split('|')[-1]
    rs = next(r for r in scene41.B_ROBOT_STARTS if r['robot_start_id'] == rid)
    scene = scene41.Scene41('B90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                            robot_q=rs['q'], gripper=rs['gripper'])
    return scene, goal, None


def evaluate(scene, mon, rec, goal, start_pos, strict_eval, initial=None):
    if rec['skill'] == 'A90':
        return strict_eval.evaluate_A(scene, mon, goal, strict_eval.DEFAULT_CONFIG,
                                      start_obj_pos=start_pos)
    return strict_eval.evaluate_B(scene, mon, goal, strict_eval.DEFAULT_CONFIG,
                                   start_obj_pos=start_pos,
                                   initial={**(initial or {}),
                                            'yaw': rec['start_object']['yaw']})


def run_mode(rec, mode, safe=True):
    import pybullet as p

    from exp40.action_adapter import ActionAdapterConfig, chunk_to_targets, limit_joint_step
    from exp41 import scene41, strict_eval

    scene, goal, _ = build_scene(rec, scene41)
    S = np.load(DATA / f"{(rec['skill'])}/ep{int(rec['stored_episode_index']):03d}_sa.npz")
    A = np.load(DATA / f"{(rec['skill'])}/ep{int(rec['stored_episode_index']):03d}_sa.npz")['action']
    S_all = np.load(DATA / f"{(rec['skill'])}/ep{int(rec['stored_episode_index']):03d}_sa.npz")['state']
    A = A.astype(float)
    S_all = S_all.astype(float)
    initial = strict_eval.evaluate_initial_B(scene, strict_eval.DEFAULT_CONFIG) \
        if rec['skill'] == 'B90' else None
    mon = strict_eval.PhysicsMonitor(scene, wall_body=scene.wall).attach()
    start = scene.object_state()
    aa = ActionAdapterConfig(gripper_mode='continuous')
    lo, hi = scene.sim.lower, scene.sim.upper
    rows = []
    t = 0
    t0 = time.time()
    try:
        while t < len(A):
            q_before = np.asarray(scene.sim.q(), dtype=np.float64)
            ee_b, _ = scene.sim.pose()
            obj_b = scene.object_state()['pos']
            if mode == 'A':
                q_t = A[t, :7].copy()
                grip = A[t, 7]
                accepted, reasons = True, []
                exec_rows = [(q_t, grip)]
            else:
                k_end = min(t + HORIZON, len(A))
                seg = A[t:k_end]
                if mode == 'B1':
                    labels = np.zeros((len(seg), 8))
                    labels[:, :7] = seg[:, :7] - q_before[None, :7]
                    labels[:, 7] = np.clip(seg[:, 7] / 0.04, 0, 1)
                else:  # B2：原始训练标签，在当前 q 上重锚定
                    labels = np.zeros((len(seg), 8))
                    labels[:, :7] = seg[:, :7] - S_all[t, :7][None, :]
                    labels[:, 7] = np.clip(seg[:, 7] / 0.04, 0, 1)
                adapted = chunk_to_targets(labels, q_before, lo, hi, aa,
                                           anchor_reference='fixed')
                # 只执行 chunk 的前 EXEC_K 行（每行一个 20 Hz 控制周期），然后重新锚定
                exec_rows = [(np.asarray(a.q_target, dtype=np.float64), a.gripper_target_m)
                             for a in adapted[:EXEC_K]]
                q_t = exec_rows[0][0]
                grip = exec_rows[0][1]
                accepted = bool(adapted[0].accepted)
                reasons = list(adapted[0].reasons)
            for sub_i, (q_t, grip) in enumerate(exec_rows):
                q_before_i = np.asarray(scene.sim.q(), dtype=np.float64)
                q_cmd = q_t.copy()
                if safe:
                    q_cmd, rate_limited, jump = limit_joint_step(q_t, q_before_i, 0.30)
                else:
                    rate_limited, jump = False, 0.0
                scene.sim.command(q_cmd)
                for j in (9, 10):
                    p.setJointMotorControl2(scene.sim.robot, j, p.POSITION_CONTROL,
                                            float(grip), force=30,
                                            physicsClientId=scene.sim.client)
                for _ in range(STEPS_PER_TICK):
                    scene.sim.step()
                q_after = np.asarray(scene.sim.q(), dtype=np.float64)
                ee_a, _ = scene.sim.pose()
                obj_a = scene.object_state()['pos']
                ee_b = ee_b
                obj_b = obj_b
                rows.append({
                'mode': mode, 'safe': safe, 'tick': t + sub_i,
                'q_before': ';'.join(f'{v:.6f}' for v in q_before_i),
                'q_expert_target': ';'.join(f'{v:.6f}' for v in q_t),
                'q_controller_command': ';'.join(f'{v:.6f}' for v in q_cmd),
                'q_after': ';'.join(f'{v:.6f}' for v in q_after),
                'track_err_max': float(np.abs(q_after - q_cmd).max()),
                'adapter_target_vs_expert': float(np.abs(q_t - A[min(t + sub_i, len(A) - 1),
                                                               :7]).max()),
                'rate_limited': rate_limited, 'jump': float(jump),
                'accepted': accepted, 'reasons': ';'.join(reasons),
                'gripper_cmd': float(grip),
                'gripper_actual': float(np.mean([p.getJointState(scene.sim.robot, j,
                                                                 physicsClientId=scene.sim.client)[0]
                                                  for j in (9, 10)])),
                'ee_before': ';'.join(f'{v:.6f}' for v in ee_b),
                'obj_before': ';'.join(f'{v:.6f}' for v in obj_b),
                'ee_dx': float(ee_a[0] - ee_b[0]), 'ee_dy': float(ee_a[1] - ee_b[1]),
                'ee_dz': float(ee_a[2] - ee_b[2]),
                'obj_after_x': float(obj_a[0]),
                'robot_block_contacts': len(p.getContactPoints(bodyA=scene.block,
                                                               bodyB=scene.sim.robot,
                                                               physicsClientId=scene.sim.client)),
                'block_wall_contacts': len(p.getContactPoints(bodyA=scene.block, bodyB=scene.wall,
                                                              physicsClientId=scene.sim.client)),
            })
            t += EXEC_K if mode != 'A' else 1
        for _ in range(int(0.4 / scene.sim.dt)):
            scene.sim.step()
        ev = evaluate(scene, mon, rec, goal, start['pos'], strict_eval, initial)
    finally:
        mon.detach()
        scene.close()
    summary = {'skill': rec['skill'], 'episode': int(rec['stored_episode_index']),
               'mode': mode, 'safe': safe,
               'strict_success': bool(ev['strict_success']),
               'failure_reason': ev['failure_reason'],
               'metrics': {k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in (ev['metrics'] or {}).items()
                           if isinstance(v, (int, float, str, bool))},
               'n_ticks': len(rows), 'seconds': round(time.time() - t0, 1)}
    return summary, rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_records()
    targets = [('A90', 0), ('A90', 5), ('B90', 0), ('B90', 2)]
    all_rows = []
    summaries = []
    for skill, ep in targets:
        rec = dict(recs[(skill, ep)])
        for mode, safe in (('A', True), ('B1', True), ('B2', True), ('A', False)):
            try:
                s, rows = run_mode(rec, mode=mode, safe=safe)
            except Exception as exc:  # noqa: BLE001
                summaries.append({'skill': skill, 'episode': ep, 'mode': mode, 'safe': safe,
                                  'error': repr(exc)[:200]})
                print('ERR', skill, ep, mode, safe, repr(exc)[:120], flush=True)
                continue
            s.update({'skill': skill, 'episode': ep, 'mode': mode, 'safe': safe})
            summaries.append(s)
            all_rows.extend([{**r, 'skill': skill, 'episode': ep} for r in rows])
            print(f"{skill} ep{ep} mode={mode} safe={safe} success={s['strict_success']} "
                  f"reason={s['failure_reason']} ticks={s['n_ticks']}", flush=True)
    keys = list(all_rows[0].keys())
    with open(OUT / 'expert_execution_trace.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    with open(OUT / 'oracle_chunk_results.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['skill', 'episode', 'mode', 'safe',
                                            'strict_success', 'failure_reason', 'n_ticks',
                                            'seconds', 'metrics'])
        w.writeheader()
        for s in summaries:
            w.writerow({**{k: s.get(k) for k in
                           ('skill', 'episode', 'mode', 'safe', 'strict_success',
                            'failure_reason', 'n_ticks', 'seconds')},
                        'metrics': json.dumps(s.get('metrics'), ensure_ascii=False)[:400]})
    (OUT / 'oracle_chunk_results.json').write_text(json.dumps(summaries, ensure_ascii=False,
                                                              indent=2, default=float),
                                                   encoding='utf-8')
    print('rows', len(all_rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
