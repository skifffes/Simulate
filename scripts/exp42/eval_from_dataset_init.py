"""从**原始数据集的初始状态**跑一次闭环（真正回答"能不能完成一次任务"）。

与 `eval_rollouts.py` 的区别：
  * 场景来自 4.1 采集记录（`start_object` 的 pos/yaw、机器人启动构型、goal），**不加 seed 抖动**；
  * prompt 用数据集里该 episode 的原始 prompt（不是脚本现拼的）；
  * 其余（Adapter、夹爪单位、execute_k、限速、严格评价器 v1.2、逐子步日志）与 eval_rollouts 完全一致。

用法（服务器 simulate venv，先起策略服务器）：
  python scripts/exp42/eval_from_dataset_init.py --skill B90 --episodes 0,1,2,3 \
      --port 8191 --out results/experiment_4_2/rollouts_dataset_init
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path('/scratch/lijh/exp42/data')
FINGER_MAX_M = 0.04
STEPS_PER_TICK = 12
HORIZON = 16


def load_meta():
    return [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]


def load_records():
    out = {}
    rec_dir = ROOT / 'results' / 'experiment_4_1' / 'dataset' / 'records'
    for f in sorted(rec_dir.glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('stored_episode_index') is not None:
            out[(r['skill'], int(r['stored_episode_index']))] = r
    return out


def robot_start_of(rec, scene41):
    cid = rec.get('config_id') or ''
    parts = cid.split('|')
    rid = parts[2] if len(parts) == 3 else scene41.B_ROBOT_STARTS[0]['robot_start_id']
    return next(r for r in scene41.B_ROBOT_STARTS if r['robot_start_id'] == rid)


def build_scene(skill, rec, scene41):
    start = rec['start_object']
    if skill == 'A90':
        gx, gy = rec['goal_point']
        goal = scene41.AGoal(rec['goal']['goal_id'], gx, gy,
                             rec['goal'].get('target_axis', 'x'),
                             rec['goal'].get('region_id', 'A_place'))
        scene = scene41.Scene41('A90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                                robot_q=None)
        return scene, goal
    goal = next(g for g in scene41.B_GOALS if g.goal_id == rec['goal']['goal_id'])
    rs = robot_start_of(rec, scene41)
    scene = scene41.Scene41('B90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                            robot_q=rs['q'], gripper=rs['gripper'])
    return scene, goal


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skill', choices=['A90', 'B90'], required=True)
    ap.add_argument('--episodes', default='0,1,2,3')
    ap.add_argument('--ticks', type=int, default=45)
    ap.add_argument('--execute-k', type=int, default=8)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8191)
    ap.add_argument('--max-joint-step-rad', type=float, default=0.30)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    import pybullet as p

    from exp40.action_adapter import ActionAdapterConfig, chunk_to_targets, limit_joint_step
    from exp41 import scene41
    from exp41.strict_eval import (DEFAULT_CONFIG, PhysicsMonitor, evaluate_A, evaluate_B,
                                   evaluate_initial_B)
    from scripts.exp40._common import make_policy_client
    from vla.cameras import CameraRig

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    client = make_policy_client(args.host, args.port)
    print('ping:', client.ping() if hasattr(client, 'ping') else 'n/a', flush=True)

    meta = load_meta()
    recs = load_records()
    aa_cfg = ActionAdapterConfig(gripper_mode='continuous')
    episodes = [int(x) for x in args.episodes.split(',')]
    rows = []

    for ep in episodes:
        row = next((m for m in meta if m['skill'] == args.skill and m['episode_index'] == ep), None)
        rec = recs.get((args.skill, ep))
        if row is None or rec is None:
            print(f'skip {args.skill} ep{ep}: 缺少 meta 或采集记录')
            continue
        scene, goal = build_scene(args.skill, rec, scene41)
        initial = None
        if args.skill == 'B90':
            initial = evaluate_initial_B(scene, DEFAULT_CONFIG)
        rig = CameraRig(scene.sim, scene.cameras, (256, 256), supersample=1)
        mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
        start = scene.object_state()
        prompt = row['prompt']
        print(f'[{args.skill} ep{ep}] prompt="{prompt}" start_obj={np.round(start["pos"],4).tolist()}',
              flush=True)
        substeps = []
        t0 = time.time()
        n_infer = 0
        try:
            for tick in range(args.ticks):
                frames = rig.capture()
                q = np.asarray(scene.sim.q(), dtype=np.float64)
                fingers = np.array([p.getJointState(scene.sim.robot, j,
                                                   physicsClientId=scene.sim.client)[0]
                                    for j in (9, 10)])
                obs = {
                    'observation/exterior_image_1_left': np.asarray(frames['top'], dtype=np.uint8),
                    'observation/wrist_image_left': np.asarray(frames['wrist'], dtype=np.uint8),
                    'observation/joint_position': q,
                    'observation/gripper_position': np.asarray(
                        [np.clip(fingers.mean() / FINGER_MAX_M, 0, 1)], dtype=np.float64),
                    'prompt': prompt,
                }
                res = client.infer(obs, episode_seed=(9100 + ep) if tick == 0 else None)
                n_infer += 1
                chunk = np.asarray(res['actions'], dtype=np.float64).copy()
                adapted = chunk_to_targets(chunk, q, scene.sim.lower, scene.sim.upper, aa_cfg,
                                           anchor_reference='fixed')
                prev_cmd = q.copy()
                for i in range(min(args.execute_k, len(adapted))):
                    a = adapted[i]
                    if not a.accepted:
                        print('  adapter reject', a.reasons, flush=True)
                        break
                    q_t, rate_limited, jump = limit_joint_step(a.q_target, prev_cmd,
                                                               args.max_joint_step_rad)
                    q_before = np.asarray(scene.sim.q(), dtype=np.float64)
                    ee_b, _ = scene.sim.pose()
                    obj_b = scene.object_state()['pos']
                    scene.sim.command(q_t)
                    for j in (9, 10):
                        p.setJointMotorControl2(scene.sim.robot, j, p.POSITION_CONTROL,
                                                float(a.gripper_target_m), force=30,
                                                physicsClientId=scene.sim.client)
                    for _ in range(STEPS_PER_TICK):
                        scene.sim.step()
                    q_after = np.asarray(scene.sim.q(), dtype=np.float64)
                    ee_a, _ = scene.sim.pose()
                    obj_a = scene.object_state()['pos']
                    substeps.append({
                        'tick': tick, 'k': i, 'rate_limited': rate_limited, 'jump': float(jump),
                        'q_track_err': float(np.abs(q_after - q_t).max()),
                        'gripper_target_m': float(a.gripper_target_m),
                        'ee_before': ';'.join(f'{v:.5f}' for v in ee_b),
                        'obj_before': ';'.join(f'{v:.5f}' for v in obj_b),
                        'ee_dx': float(ee_a[0] - ee_b[0]), 'ee_dy': float(ee_a[1] - ee_b[1]),
                        'ee_dz': float(ee_a[2] - ee_b[2]),
                        'obj_dx': float(obj_a[0] - obj_b[0]), 'obj_dy': float(obj_a[1] - obj_b[1]),
                        'robot_block_contacts': len(p.getContactPoints(
                            bodyA=scene.block, bodyB=scene.sim.robot,
                            physicsClientId=scene.sim.client)),
                        'block_wall_contacts': len(p.getContactPoints(
                            bodyA=scene.block, bodyB=scene.wall,
                            physicsClientId=scene.sim.client)),
                    })
                    prev_cmd = q_t
            for _ in range(int(0.4 / scene.sim.dt)):
                scene.sim.step()
            if args.skill == 'A90':
                ev = evaluate_A(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'])
            else:
                ev = evaluate_B(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'],
                                initial={**initial, 'yaw': start['yaw']})
        finally:
            mon.detach()
            scene.close()

        rec_out = {
            'skill': args.skill, 'episode': ep, 'init': 'dataset_original',
            'prompt': prompt, 'start_obj': [float(v) for v in start['pos']],
            'strict_success': bool(ev['strict_success']), 'failure_reason': ev['failure_reason'],
            'checks': ev['checks'], 'metrics': ev['metrics'],
            'n_infer': n_infer, 'seconds': round(time.time() - t0, 1),
        }
        rows.append(rec_out)
        (out / f'{args.skill}_ep{ep:02d}.json').write_text(
            json.dumps(rec_out, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
        if substeps:
            with open(out / f'{args.skill}_ep{ep:02d}_substeps.csv', 'w', newline='',
                      encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=list(substeps[0].keys()))
                w.writeheader()
                w.writerows(substeps)
        print(f'[{args.skill} ep{ep}] success={rec_out["strict_success"]} '
              f'reason={rec_out["failure_reason"]} '
              f'adv={ev["metrics"].get("advance_x_m")} '
              f'wall={ev["metrics"].get("wall_contact_steps")}', flush=True)

    summary = {'skill': args.skill, 'init': 'dataset_original', 'n': len(rows),
               'n_success': sum(1 for r in rows if r['strict_success']),
               'rows': [{k: r[k] for k in ('episode', 'strict_success', 'failure_reason',
                                           'metrics')} for r in rows]}
    (out / f'{args.skill}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps({'skill': args.skill, 'n': summary['n'],
                      'n_success': summary['n_success'],
                      'reasons': [r['failure_reason'] for r in rows]}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
