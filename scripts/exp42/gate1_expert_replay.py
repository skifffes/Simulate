"""Gate 1：把**专家动作**经过修复后的 Action Adapter 真实回放，检验执行链路是否保真。

三个对象（同一 episode、同一初始状态、同一物理参数、同一评价器）：
  Reference  : 数据集归档的专家轨迹（`observation.state` 的逐帧 q）
  Replay A   : 原始专家控制目标（数据集 `action[:7]` 绝对关节目标 + `action[7]` 米）直接下发
  Replay B   : 训练标签（`a_k = A[t0+k] - S[t0]`）→ `chunk_to_targets(anchor_reference='fixed',
               gripper continuous)` → 只执行前 execute_k=8 步，然后**重新取实测 q 作锚点**

Replay B 的语义与 `scripts/exp42/eval_rollouts.py` 完全一致（同一 Adapter、同一 execute_k、
同一频率、同一夹爪单位），只是把模型输出换成专家标签——因此 Gate 1 检验的是
"执行链路 + 标签语义"是否保真，不涉及模型。

用法（simulate venv，无 GPU）：
  python scripts/exp42/gate1_expert_replay.py --skill A90 --episodes 5,6
  python scripts/exp42/gate1_expert_replay.py --skill B90 --episodes 0,1
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

DATA = Path('/scratch/lijh/exp42/data')
FINGER_MAX_M = 0.04
POLICY_HZ = 20
STEPS_PER_TICK = 240 // POLICY_HZ
EXECUTE_K = 8
HORIZON = 16


def load_meta() -> list[dict]:
    return [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]


def load_records() -> dict:
    """stored_episode_index → 采集记录（含 start_object / goal_point / config_id）。"""
    out = {}
    rec_dir = ROOT / 'results' / 'experiment_4_1' / 'dataset' / 'records'
    for f in sorted(rec_dir.glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('stored_episode_index') is not None:
            out[(r['skill'], int(r['stored_episode_index']))] = r
    return out


def robot_start_of(rec: dict):
    """B90 的机器人构型由 config_id 的第三段给出（goal|obj|robot）。"""
    from exp41 import scene41

    cid = rec.get('config_id') or ''
    parts = cid.split('|')
    rid = parts[2] if len(parts) == 3 else scene41.B_ROBOT_STARTS[0]['robot_start_id']
    return next(r for r in scene41.B_ROBOT_STARTS if r['robot_start_id'] == rid)


def build_scene(skill: str, rec: dict):
    """按采集记录恢复**完全相同**的初始场景（物体位姿、机器人构型、相机、墙体）。"""
    from exp41 import scene41

    start = rec['start_object']
    if skill == 'A90':
        gx, gy = rec['goal_point']
        goal = scene41.AGoal(rec['goal']['goal_id'], gx, gy,
                             rec['goal'].get('target_axis', 'x'),
                             rec['goal'].get('region_id', 'A_place'))
        # 与 4.1 采集一致：A90 用场景默认 HOME（build_a_scene 未传 robot_q）
        scene = scene41.Scene41('A90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                                robot_q=None)
        return scene, goal
    goal = next(g for g in scene41.B_GOALS if g.goal_id == rec['goal']['goal_id'])
    rs = robot_start_of(rec)
    scene = scene41.Scene41('B90', obj_pos=tuple(start['pos']), obj_yaw=start['yaw'],
                            robot_q=rs['q'], gripper=rs['gripper'])
    return scene, goal


def run_episode(skill: str, ep: int, mode: str, meta_row: dict, rec: dict,
                out_rows: list[dict], initial: dict | None,
                label_anchor: str = 'measured') -> dict:
    import pybullet as p

    from exp40.action_adapter import ActionAdapterConfig, chunk_to_targets
    from exp41.strict_eval import DEFAULT_CONFIG, PhysicsMonitor, evaluate_A, evaluate_B

    sa = np.load(DATA / meta_row['state_path'])
    S = sa['state'].astype(np.float64)     # (T,9) 记录状态
    A = sa['action'].astype(np.float64)    # (T,8) 专家绝对目标 + 夹爪米
    T = S.shape[0]

    scene, goal = build_scene(skill, rec)
    mon = PhysicsMonitor(scene, wall_body=scene.wall).attach()
    start_obj = scene.object_state()
    cfg = ActionAdapterConfig(gripper_mode='continuous')
    chunk_id = 0
    k_in_chunk = 0
    anchor_q = None
    adapted_chunk = None
    teacher_err: list[float] = []
    try:
        for t in range(T):
            q_before = np.asarray(scene.sim.q(), dtype=np.float64)
            if mode == 'expert':
                target_q = A[t, :7].copy()
                grip_m = float(A[t, 7])
                anchor = q_before
            else:
                if t % EXECUTE_K == 0 or adapted_chunk is None or k_in_chunk >= len(adapted_chunk):
                    anchor_q = q_before.copy()
                    chunk_id += 1
                    k_in_chunk = 0
                    n = int(min(HORIZON, T - t))
                    labels = np.zeros((n, 8), dtype=np.float64)
                    ref_q = q_before if label_anchor == 'measured' else S[t, :7]
                    labels[:, :7] = A[t:t + n, :7] - ref_q[None, :]
                    labels[:, 7] = np.clip(A[t:t + n, 7] / FINGER_MAX_M, 0, 1)
                    adapted_chunk = chunk_to_targets(labels, anchor_q, scene.sim.lower,
                                                     scene.sim.upper, cfg,
                                                     anchor_reference='fixed')
                    for k in range(n):
                        teacher_err.append(float(np.abs(adapted_chunk[k].q_target
                                                        - A[t + k, :7]).max()))
                k = k_in_chunk
                target_q = np.asarray(adapted_chunk[k].q_target, dtype=np.float64)
                grip_m = float(adapted_chunk[k].gripper_target_m)
                anchor = anchor_q
                k_in_chunk += 1
            scene.sim.command(target_q)
            for j in (9, 10):
                p.setJointMotorControl2(scene.sim.robot, j, p.POSITION_CONTROL, grip_m, force=30,
                                        physicsClientId=scene.sim.client)
            for _ in range(STEPS_PER_TICK):
                scene.sim.step()
            q_after = np.asarray(scene.sim.q(), dtype=np.float64)
            ee_pos, ee_quat = scene.sim.pose()
            fingers = np.array([p.getJointState(scene.sim.robot, j,
                                                physicsClientId=scene.sim.client)[0]
                                for j in (9, 10)])
            obj = scene.object_state()
            rbc = len(p.getContactPoints(bodyA=scene.block, bodyB=scene.robot,
                                         physicsClientId=scene.sim.client))
            wbc = len(p.getContactPoints(bodyA=scene.block, bodyB=scene.wall,
                                         physicsClientId=scene.sim.client))
            out_rows.append({
                'skill': skill, 'episode': ep, 'mode': mode, 'tick': t,
                'chunk_id': chunk_id if mode == 'adapter' else 0,
                'k_in_chunk': (k_in_chunk - 1) if mode == 'adapter' else 0,
                'anchor_q': ';'.join(f'{v:.6f}' for v in np.asarray(anchor).ravel()),
                'expert_target_q': ';'.join(f'{v:.6f}' for v in A[t, :7]),
                'adapter_target_q': ';'.join(f'{v:.6f}' for v in target_q),
                'actual_q': ';'.join(f'{v:.6f}' for v in q_after),
                'q_track_err': float(np.abs(q_after - target_q).max()),
                'q_vs_expert_state_err': float(np.abs(q_after - S[t, :7]).max()),
                'ee_x': float(ee_pos[0]), 'ee_y': float(ee_pos[1]), 'ee_z': float(ee_pos[2]),
                'ee_quat': ';'.join(f'{v:.6f}' for v in ee_quat),
                'gripper_cmd_m': grip_m,
                'finger9': float(fingers[0]), 'finger10': float(fingers[1]),
                'obj_x': float(obj['pos'][0]), 'obj_y': float(obj['pos'][1]),
                'obj_z': float(obj['pos'][2]),
                'obj_yaw': float(obj['yaw']),
                'obj_tilt_deg': abs(obj['roll_deg']) + abs(obj['pitch_deg']),
                'robot_block_contacts': rbc, 'block_wall_contacts': wbc,
            })
        for _ in range(int(0.4 / scene.sim.dt)):
            scene.sim.step()
        if skill == 'A90':
            ev = evaluate_A(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start_obj['pos'])
        else:
            ev = evaluate_B(scene, mon, goal, DEFAULT_CONFIG, start_obj_pos=start_obj['pos'],
                            initial=initial or {'task_valid': True, 'yaw': start_obj['yaw']})
        summ = mon.summary()
    finally:
        mon.detach()
        scene.close()

    rows = [r for r in out_rows if r['skill'] == skill and r['episode'] == ep and r['mode'] == mode]
    obj_x = np.array([r['obj_x'] for r in rows])
    obj_y = np.array([r['obj_y'] for r in rows])
    ee_x = np.array([r['ee_x'] for r in rows])
    ee_y = np.array([r['ee_y'] for r in rows])
    return {
        'skill': skill, 'episode': ep, 'mode': mode, 'label_anchor': label_anchor,
        'ticks': len(rows),
        'adapter_target_vs_expert_max_err_rad': (max(teacher_err) if teacher_err else None),
        'final_q_vs_expert_state_max_err_rad': rows[-1]['q_vs_expert_state_err'],
        'max_q_track_err_rad': float(np.abs([r['q_track_err'] for r in rows]).max()),
        'object_dx': float(obj_x[-1] - obj_x[0]), 'object_dy': float(obj_y[-1] - obj_y[0]),
        'ee_dx': float(ee_x[-1] - ee_x[0]), 'ee_dy': float(ee_y[-1] - ee_y[0]),
        'robot_block_contact_steps': int(summ['robot_block']['n_steps']),
        'block_wall_contact_steps': int(summ['block_wall']['n_steps']),
        'terminal_obj': [round(float(v), 5) for v in (obj_x[-1], obj_y[-1], rows[-1]['obj_z'])],
        'strict_success': bool(ev['strict_success']), 'failure_reason': ev['failure_reason'],
        'checks': ev['checks'],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skill', choices=['A90', 'B90'], required=True)
    ap.add_argument('--episodes', required=True, help='逗号分隔的数据集 episode 索引（train split）')
    ap.add_argument('--modes', default='expert,adapter')
    ap.add_argument('--label-anchor', choices=['recorded', 'measured'], default='measured',
                    help='recorded=训练标签的锚点（复现训练语义）；'
                         'measured=用实测 q 重算标签（检验执行链路能否自纠正）')
    ap.add_argument('--out', default='results/experiment_4_2/contract_replay_p0c')
    args = ap.parse_args(argv)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    recs = load_records()
    eps = [int(x) for x in args.episodes.split(',')]
    modes = [m.strip() for m in args.modes.split(',')]

    rows: list[dict] = []
    summaries = []
    for ep in eps:
        row = next((m for m in meta if m['skill'] == args.skill and m['episode_index'] == ep), None)
        if row is None or (args.skill, ep) not in recs:
            print(f'skip {args.skill} ep{ep}: 缺少导出数据或采集记录')
            continue
        rec = recs[(args.skill, ep)]
        initial = None
        if args.skill == 'B90':
            from exp41.strict_eval import DEFAULT_CONFIG, evaluate_initial_B
            probe, _ = build_scene(args.skill, rec)
            initial = evaluate_initial_B(probe, DEFAULT_CONFIG)
            initial['yaw'] = rec['start_object']['yaw']
            probe.close()
        for mode in modes:
            print(f'[{args.skill} ep{ep}] replay {mode} ...', flush=True)
            res = run_episode(args.skill, ep, mode, row, rec, rows, initial,
                              label_anchor=args.label_anchor)
            summaries.append(res)
            print(f"  -> success={res['strict_success']} reason={res['failure_reason']} "
                  f"q_err={res['final_q_vs_expert_state_max_err_rad']:.4f} "
                  f"obj_dx={res['object_dx']:+.4f} obj_dy={res['object_dy']:+.4f} "
                  f"contact={res['robot_block_contact_steps']}", flush=True)

    tag = args.skill
    if rows:
        with open(out / f'expert_vs_adapter_trace_{tag}.csv', 'w', newline='',
                  encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / f'expert_replay_results_{tag}.json').write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps([{k: s[k] for k in ('skill', 'episode', 'mode', 'strict_success',
                                         'failure_reason', 'final_q_vs_expert_state_max_err_rad',
                                         'object_dx', 'object_dy', 'robot_block_contact_steps',
                                         'adapter_target_vs_expert_max_err_rad')}
                      for s in summaries], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
