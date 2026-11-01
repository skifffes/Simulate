"""Level 4：有限物理闭环 Smoke Test。

链路：PyBullet Observation → π₀.₅ → Action Chunk → Action Adapter → Controller → PyBullet

安全设计（prompt Step 7.4）：
  * 关节限位 / EE 工作空间 / 单步幅度 / 不合法动作拒绝 / NaN·Inf / 非预期碰撞 /
    物体异常运动 / 控制器跟踪误差 / 超时 / 最大步数 —— 全部由 exp40.safety.SafetyMonitor 判定；
  * 触发任一条件立即停止，保存最后的输入、chunk、目标与实际轨迹；
  * 不做“无限制 clipping 把错误动作变成有效控制”。

执行频率：策略 15 Hz（默认，与官方 DROID 一致；可用 --policy-hz 20 回到实验 1–3 的旧时序），
每 tick 执行 240/15 = 16 个 240 Hz 物理步。推理期间仿真暂停 → 这是**仿真时间闭环**，
不代表满足真实机器人 15 Hz 的墙钟约束（见 hotfix 报告 §D）。

用法：
    python scripts/exp40/level4_closed_loop.py --scene B90 --ticks 25 --port 8000
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import droid_contract as dc  # noqa: E402
from exp40.action_adapter import ActionAdapterConfig, adapt_action, summarize_chunk  # noqa: E402
from exp40.chunk import ActionChunker  # noqa: E402
from exp40.observation_adapter import build_droid_observation, finger_widths, gripper_scalar  # noqa: E402
from exp40.safety import SafetyLimits, SafetyMonitor  # noqa: E402
from scripts.exp40._common import (  # noqa: E402
    build_sim_scene, capture, make_policy_client, model_input_summary, save_json,
)

STEPS_PER_TICK = 12  # 20 Hz 策略 / 240 Hz 物理（实验 1–3 的旧默认，保留兼容）
SIM_HZ = 240
DROID_HZ = dc.DROID_CONTROL_HZ  # 15 Hz：官方 DROID 采集/执行频率


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='B90', choices=['A90', 'B90'])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ticks', type=int, default=20, help='最多执行多少个策略控制步')
    parser.add_argument('--policy-hz', type=float, default=DROID_HZ,
                        help='策略控制频率（默认 15 Hz，与官方 DROID 一致；实验 1–3 为 20 Hz）')
    parser.add_argument('--execute-k', type=int, default=8, help='每个 chunk 执行多少步后重推理')
    parser.add_argument('--max-delta-rad', type=float, default=0.15)
    parser.add_argument('--wall-time', type=float, default=90.0)
    parser.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_0' / 'closed_loop_smoke'))
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / f'trace_{args.scene}_seed{args.seed}.csv'

    limits = SafetyLimits(max_ticks=args.ticks, wall_time_max_s=args.wall_time,
                          max_joint_step_rad=0.30)
    a_cfg = ActionAdapterConfig(max_delta_rad=args.max_delta_rad)
    # 控制时序：策略频率 -> 每个控制步内推进多少个 240 Hz 物理步
    steps_per_tick = max(1, int(round(SIM_HZ / float(args.policy_hz))))

    sim, block, rig, prompt, info = build_sim_scene(args.scene, seed=args.seed)
    monitor = SafetyMonitor(sim, limits, object_ids=(block,))
    chunker = ActionChunker(execute_k=args.execute_k)
    chunker.set_prompt(prompt)

    rows = []
    last_rt = ''  # 当前 chunk 的推理往返时间（s），写进每一行 trace
    chunks = []
    client = None
    error = None
    try:
        client = make_policy_client(args.host, args.port)
        meta = client.get_server_metadata()
        q_prev = np.asarray(sim.q(), dtype=float)

        with open(trace_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['tick', 'phase', 'q1..q7', 'q_target1..7', 'ee_x', 'ee_y', 'ee_z',
                             'finger1', 'finger2', 'gripper_scalar', 'action_gripper',
                             'block_x', 'block_y', 'block_z', 'block_yaw',
                             'chunk_roundtrip_s', 'accepted', 'reasons', 'stopped', 'stop_reason'])

            for _ in range(args.ticks + 1):
                if not monitor.tick_once():
                    break
                if not monitor.check_time():
                    break
                if monitor.stopped:
                    break

                # ---- 观测 ----
                obs = capture(sim, rig)
                g = gripper_scalar(sim)
                pkt = build_droid_observation(obs['frames'], obs['q'], g, prompt)

                # ---- 需要时推理 ----
                if chunker.needs_inference():
                    t0 = time.perf_counter()
                    try:
                        result = client.infer(pkt.payload)
                    except Exception as exc:  # 服务器异常必须安全停止
                        monitor.record('server_error', error=f'{type(exc).__name__}: {exc}')
                        error = str(exc)
                        break
                    infer_s = time.perf_counter() - t0
                    actions = np.asarray(result['actions'], dtype=np.float64)
                    chunker.push(actions)
                    last_rt = round(infer_s, 4)
                    chunks.append({'tick': monitor.tick, 'roundtrip_s': round(infer_s, 4),
                                   'summary': summarize_chunk(actions),
                                   'raw': actions.round(6).tolist(),
                                   'first_action': actions[0].round(6).tolist(),
                                   'request': model_input_summary(pkt.payload)})
                    if not np.all(np.isfinite(actions)):
                        monitor.record('nan_or_inf_action')
                        break
                action = chunker.pop()
                if action is None:
                    continue  # 理论上不会发生（上面已补货）
                # ---- 适配 ----
                q_now = np.asarray(sim.q(), dtype=float)
                adapted = adapt_action(action, q_now, sim.lower, sim.upper, a_cfg)
                if not monitor.check_action(action, adapted.reasons):
                    adapted.accepted = False
                if not adapted.accepted:
                    writer.writerow([monitor.tick, 'adapt_rejected', *np.round(q_now, 5),
                                     *np.round(adapted.q_target, 5), *np.round(obs['ee_position'], 5),
                                     *np.round(finger_widths(sim), 5), round(g, 5), round(adapted.gripper_value, 5),
                                     *np.round(_block_state(block, sim)[:3], 5), round(_block_state(block, sim)[3], 5),
                                     last_rt, int(adapted.accepted), ';'.join(adapted.reasons), 1,
                                     monitor.why_stopped()])
                    break

                # ---- 控制 ----
                for _ in range(steps_per_tick):
                    sim.command(adapted.q_target)
                    _set_gripper(sim, adapted.gripper_target_m)
                    sim.step()

                q_after = np.asarray(sim.q(), dtype=float)
                ee = np.asarray(sim.pose()[0], dtype=float)
                ok = (monitor.check_state(q_prev, q_after, sim.lower, sim.upper)
                      and monitor.check_ee(ee)
                      and monitor.check_collisions()
                      and monitor.check_objects((block,))
                      and monitor.check_tracking(adapted.q_target, q_after))
                q_prev = q_after
                bs = _block_state(block, sim)
                writer.writerow([monitor.tick, 'execute', *np.round(q_after, 5),
                                 *np.round(adapted.q_target, 5), *np.round(ee, 5),
                                 *np.round(finger_widths(sim), 5), round(gripper_scalar(sim), 5),
                                 round(adapted.gripper_value, 5),
                                 *np.round(bs[:3], 5), round(bs[3], 5),
                                 last_rt, int(adapted.accepted), '', int(monitor.stopped),
                                 monitor.why_stopped() or ''])
                if not ok:
                    break

        summary = {
            'scene': args.scene,
            'seed': args.seed,
            'prompt': prompt,
            'scene_info': info,
            'server_metadata': meta,
            'safety': monitor.summary(),
            'chunker': chunker.summary(),
            'adapter_config': {
                'max_delta_rad': a_cfg.max_delta_rad,
                'official_clip_rad': a_cfg.official_clip_rad,
                'gripper_mode': a_cfg.gripper_mode,
                'gripper_threshold': a_cfg.gripper_threshold,
            },
            'steps_per_tick': steps_per_tick,
            'policy_hz': float(args.policy_hz),
            'sim_hz': SIM_HZ,
            'droid_control_hz': dc.DROID_CONTROL_HZ,
            'timing_note': ('同步闭环：每个策略控制步先推理（必要时）再推进 '
                            f'{steps_per_tick} 个物理步；推理期间仿真暂停，'
                            '因此这是仿真时间闭环，不代表满足实时机器人 15 Hz 的墙钟约束'),
            'n_chunks': len(chunks),
            'chunks': chunks[:5],
            'final_block_state': _block_state(block, sim),
            'trace_csv': str(trace_path.relative_to(ROOT)) if trace_path.is_relative_to(ROOT) else str(trace_path),
            'error': error,
            'completed_ticks': monitor.tick,
        }
        save_json(out / f'closed_loop_{args.scene}_seed{args.seed}.json', summary)
        if chunks:
            np.savez_compressed(out / f'chunks_{args.scene}_seed{args.seed}.npz',
                                **{f'chunk_{i:03d}': np.asarray(c['raw']) for i, c in enumerate(chunks)})
        print(json.dumps({'scene': args.scene, 'ticks': summary['completed_ticks'],
                          'stop_reason': summary['safety']['stop_reason'],
                          'n_chunks': summary['n_chunks'],
                          'final_block': summary['final_block_state']}, ensure_ascii=False, indent=2))
        return 0
    finally:
        sim.close()


def _set_gripper(sim, width_m: float) -> None:
    import pybullet as p

    for j in (9, 10):
        p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(width_m), force=30,
                                physicsClientId=sim.client)


def _block_state(block, sim):
    import pybullet as p

    pos, quat = p.getBasePositionAndOrientation(block, physicsClientId=sim.client)
    euler = p.getEulerFromQuaternion(quat)
    yaw = float(np.arctan2(np.sin(euler[2]), np.cos(euler[2])))
    return [float(pos[0]), float(pos[1]), float(pos[2]), yaw]


if __name__ == '__main__':
    raise SystemExit(main())
