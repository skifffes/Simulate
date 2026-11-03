"""Gate 5：π₀.₅ 自主闭环 + 严格物理评价（A90 / B90 原子技能）。

与实验 4.0 的 level4 闭环的区别：
  * 这里**必须**给出严格成功判定，因此把实验 4.1 的 `PhysicsMonitor`（每个 240 Hz 物理步记录接触）
    接在仿真步上，并用 `exp4.1-strict-v1.2` 的 A90/B90 判据评价；
  * 场景、相机、动作适配器、RNG reset 全部复用 4.0/4.1 已验证组件。

流程（每个 episode）：
    scene41 场景（权威 body id） → PhysicsMonitor.attach
    → 20 Hz 闭环：capture → client.infer（首帧带 episode_seed）→ chunk → adapt_action
      → 执行 execute_k 步（每步 12 个物理步）→ 记录
    → 稳定窗口 0.8 s → strict_eval → 落盘 JSON

用法：
    python scripts/exp42/eval_rollouts.py --skill B90 --episodes 3 --ticks 60 --port 8000 \
        --out results/experiment_4_2/finetuned_rollouts --tag ft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FINGER_MAX_M = 0.04
POLICY_HZ = 20
STEPS_PER_TICK = 12


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skill', choices=['A90', 'B90'], required=True)
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--ticks', type=int, default=60)
    ap.add_argument('--seed0', type=int, default=9000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--goal-id', default='')
    ap.add_argument('--execute-k', type=int, default=8)
    ap.add_argument('--tag', default='ft')
    ap.add_argument('--gripper-mode', choices=['continuous', 'binary'], default='continuous',
                    help='continuous=g*0.04（与训练标签 0.425->0.017m 一致）；binary=官方 client 的 0.5 二值化')
    ap.add_argument('--action-gain', type=float, default=1.0,
                    help='诊断用：反归一化后、安全裁剪前缩放关节增量（只用于增益实验）')
    ap.add_argument('--out', default='results/experiment_4_2/finetuned_rollouts')
    args = ap.parse_args(argv)

    import pybullet as p

    from exp40 import camera_v2
    from exp40.action_adapter import ActionAdapterConfig, chunk_to_targets
    from exp41 import scene41
    from exp41.strict_eval import (
        DEFAULT_CONFIG, PhysicsMonitor, evaluate_A, evaluate_B, evaluate_initial_B,
    )
    from scripts.exp40._common import make_policy_client
    from vla.cameras import CameraRig

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    client = make_policy_client(args.host, args.port)
    print('ping:', client.ping() if hasattr(client, 'ping') else 'n/a', flush=True)

    aa_cfg = ActionAdapterConfig(gripper_mode=args.gripper_mode)
    rows = []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        if args.skill == 'A90':
            goal = next((g for g in scene41.A_GOALS if g.goal_id == args.goal_id), scene41.A_GOALS[4])
            scene = scene41.build_a_scene(goal, seed=seed)
            initial = None
        else:
            obj_id, goal_id = scene41.B_ALIGNED_PAIRS[ep % len(scene41.B_ALIGNED_PAIRS)]
            goal = next(g for g in scene41.B_GOALS if g.goal_id == (args.goal_id or goal_id))
            obj = next(o for o in scene41.B_OBJECT_STARTS if o['start_id'] == obj_id)
            robot = scene41.B_ROBOT_STARTS[0]
            scene = scene41.build_b_scene(obj, robot, seed=seed)
            initial = evaluate_initial_B(scene, DEFAULT_CONFIG)
        rig = CameraRig(scene.sim, scene.cameras, (256, 256), supersample=1)
        mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
        start = scene.object_state()
        t0 = time.time()
        n_infer = 0
        infer_s = []
        act_log = []
        try:
            for tick in range(args.ticks):
                frames = rig.capture()
                q = np.asarray(scene.sim.q(), dtype=np.float64)
                fingers = np.array([p.getJointState(scene.sim.robot, j,
                                                   physicsClientId=scene.sim.client)[0]
                                    for j in (9, 10)])
                prompt = (f'push the orange block along the positive X direction to the wall '
                          f'region y in [{goal.y_lo:.2f}, {goal.y_hi:.2f}]'
                          if args.skill == 'B90' else
                          f'pick up the orange block and place it in the target area at '
                          f'({goal.x:.3f}, {goal.y:.3f}) with its long side aligned with X')
                obs = {
                    'observation/exterior_image_1_left': np.asarray(frames['top'], dtype=np.uint8),
                    'observation/wrist_image_left': np.asarray(frames['wrist'], dtype=np.uint8),
                    'observation/joint_position': q,
                    'observation/gripper_position': np.asarray(
                        [np.clip(fingers.mean() / FINGER_MAX_M, 0, 1)], dtype=np.float64),
                    'prompt': prompt,
                }
                ti = time.time()
                res = client.infer(obs, episode_seed=seed if tick == 0 else None)
                infer_s.append(time.time() - ti)
                n_infer += 1
                chunk = np.asarray(res['actions'], dtype=np.float64).copy()
                if args.action_gain != 1.0:
                    chunk[:, :7] *= args.action_gain
                adapted = chunk_to_targets(chunk, q, scene.sim.lower, scene.sim.upper, aa_cfg,
                                           anchor_reference='fixed')
                act_log.append({'tick': tick, 'action_absmax': float(np.abs(chunk[:, :8]).max()),
                                'gripper_cmd': float(chunk[0, 7]),
                                'gripper_target_m': float(adapted[0].gripper_target_m),
                                'delta_applied_absmax': float(max(np.abs(a.delta_applied).max()
                                                                  for a in adapted))})
                k = min(args.execute_k, len(adapted))
                for i in range(k):
                    scene.sim.command(np.asarray(adapted[i].q_target, dtype=np.float64))
                    for j in (9, 10):
                        p.setJointMotorControl2(scene.sim.robot, j, p.POSITION_CONTROL,
                                                float(adapted[i].gripper_target_m), force=30,
                                                physicsClientId=scene.sim.client)
                    for _ in range(STEPS_PER_TICK):
                        scene.sim.step()
                if tick % 10 == 0:
                    st = scene.object_state()
                    print(f'  [{args.skill} ep{ep} tick {tick}] obj={np.round(st["pos"], 4).tolist()} '
                          f'grip={float(fingers.mean()):.3f}', flush=True)
            # 稳定窗口（继续记录接触）
            for _ in range(int(0.8 / scene.sim.dt)):
                scene.sim.step()
            if args.skill == 'A90':
                ev = evaluate_A(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'])
            else:
                ev = evaluate_B(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start['pos'],
                                initial={**initial, 'yaw': start['yaw']})
        finally:
            mon.detach()
            scene.close()
        rec = {
            'eval_version': ev.get('eval_version'), 'skill': args.skill, 'episode': ep,
            'seed': seed, 'tag': args.tag, 'goal_id': goal.goal_id,
            'camera_version': camera_v2.VERSION, 'camera_sha256': camera_v2.CAMERA_CONFIG_SHA256,
            'strict_success': ev['strict_success'], 'failure_reason': ev['failure_reason'],
            'checks': ev['checks'], 'metrics': ev['metrics'],
            'n_infer': n_infer, 'seconds': round(time.time() - t0, 1),
            'anchor_reference': 'fixed', 'gripper_mode': args.gripper_mode,
            'action_gain': args.action_gain, 'execute_k': args.execute_k,
            'mean_infer_s': float(np.mean(infer_s)) if infer_s else None,
            'p95_infer_s': float(np.percentile(infer_s, 95)) if infer_s else None,
            'action_log': act_log,
            'contact_steps': {'wall': ev['contact_summary']['block_wall']['n_steps'],
                              'robot_block': ev['contact_summary']['robot_block']['n_steps']},
        }
        rows.append(rec)
        (out / f'{args.skill}_ep{ep:02d}.json').write_text(
            json.dumps(rec, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
        print(f'[{args.skill} ep{ep}] strict_success={rec["strict_success"]} '
              f'reason={rec["failure_reason"]} infer={rec["mean_infer_s"] and round(rec["mean_infer_s"], 3)}s',
              flush=True)

    n = len(rows)
    ok = sum(1 for r in rows if r['strict_success'])
    summary = {'skill': args.skill, 'tag': args.tag, 'n': n, 'n_success': ok,
               'success_rate': ok / max(n, 1),
               'failure_reasons': {k: sum(1 for r in rows if r['failure_reason'] == k)
                                   for k in {r['failure_reason'] for r in rows}},
               'mean_infer_s': float(np.mean([r['mean_infer_s'] for r in rows
                                              if r['mean_infer_s']])),
               'rows': rows}
    (out / f'{args.skill}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('skill', 'tag', 'n', 'n_success',
                                              'failure_reasons', 'mean_infer_s')},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
