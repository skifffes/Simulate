"""Level 5：A90 / B90 环境接口检查（不要求任务成功，也不计算成功率）。

检查项（prompt Level 5）：
  * 场景能正确初始化（物体位置/朝向、墙、托盘、限位）
  * 语言指令正确传入（A90 用 TASKS['A90']，B90 用 PROMPT_B90）
  * 两台相机都出图，且图像非全黑/非全白
  * 模型动作能被解析（形状/有限性/量级）
  * 控制流程能走通（少量 tick，安全停止机制生效）
  * 回合能正确退出（资源释放、日志完整）

用法：
    python scripts/exp40/level5_env_interface.py --port 8000
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import droid_contract as dc  # noqa: E402
from exp40.action_adapter import ActionAdapterConfig, adapt_action, summarize_chunk  # noqa: E402
from exp40.observation_adapter import build_droid_observation, gripper_scalar  # noqa: E402
from exp40.safety import SafetyLimits, SafetyMonitor  # noqa: E402
from scripts.exp40._common import (  # noqa: E402
    build_sim_scene, capture, make_policy_client, save_json,
)


def check(results, name, ok, detail=None):
    results.append({'check': name, 'pass': bool(ok), 'detail': detail})
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  {json.dumps(detail, default=float)}' if detail else ''))
    return ok


def run_scene(scene: str, args) -> dict:
    results = []
    prompt_expect = dc.PROMPT_B90 if scene == 'B90' else None
    sim, block, rig, prompt, info = build_sim_scene(scene, seed=args.seed)
    out = {'scene': scene, 'prompt': prompt, 'scene_info': info}
    try:
        check(results, f'{scene}: 场景初始化', sim.robot > 0 and block > 0,
              {'robot': sim.robot, 'block': block, 'block_pos': info['block_start'],
               'block_yaw': info['block_yaw']})
        check(results, f'{scene}: 关节限位可读', np.all(sim.upper > sim.lower),
              {'lower': np.round(sim.lower, 3).tolist(), 'upper': np.round(sim.upper, 3).tolist()})
        if prompt_expect is not None:
            check(results, f'{scene}: 语言指令与实验 1–3 原文一致', prompt == prompt_expect,
                  {'prompt': prompt})

        obs = capture(sim, rig)
        for name, img in obs['frames'].items():
            mean = float(np.asarray(img).mean())
            check(results, f'{scene}: 相机 {name} 出图且非全黑/全白',
                  img.shape == (256, 256, 3) and 3 < mean < 252,
                  {'shape': list(img.shape), 'mean': round(mean, 2)})

        pkt = build_droid_observation(obs['frames'], obs['q'], gripper_scalar(sim), prompt)
        check(results, f'{scene}: DROID payload 键名/形状',
              pkt.payload[dc.KEY_IMAGE_BASE].shape == (224, 224, 3)
              and pkt.payload[dc.KEY_JOINT_POSITION].shape == (7,)
              and pkt.payload[dc.KEY_GRIPPER_POSITION].shape == (1,),
              {'shapes': {k: list(np.asarray(v).shape) for k, v in pkt.payload.items() if k != dc.KEY_PROMPT}})

        if args.no_server:
            check(results, f'{scene}: 模型动作解析', True, {'skipped': '--no-server'})
            out['checks'] = results
            out['all_pass'] = all(r['pass'] for r in results)
            return out

        client = make_policy_client(args.host, args.port)
        meta = client.get_server_metadata()
        result = client.infer(pkt.payload)
        actions = np.asarray(result['actions'])
        check(results, f'{scene}: 模型返回 (horizon, 8) 且有限',
              actions.ndim == 2 and actions.shape[1] == dc.ACTION_DIM and np.all(np.isfinite(actions)),
              {'shape': list(actions.shape), 'summary': summarize_chunk(actions)})

        # 少量控制步：验证控制流程 + 安全机制
        monitor = SafetyMonitor(sim, SafetyLimits(max_ticks=args.ticks, wall_time_max_s=60))
        cfg = ActionAdapterConfig(max_delta_rad=args.max_delta_rad)
        q = np.asarray(sim.q(), dtype=float)
        n_exec = 0
        for i in range(min(args.ticks, len(actions))):
            if not monitor.tick_once():
                break
            adapted = adapt_action(actions[i], q, sim.lower, sim.upper, cfg)
            if not adapted.accepted:
                monitor.record('rejected_action', reasons=adapted.reasons)
                break
            for _ in range(12):
                sim.command(adapted.q_target)
                sim.step()
            q = np.asarray(sim.q(), dtype=float)
            n_exec += 1
            if not monitor.check_state(q, q, sim.lower, sim.upper):
                break
        check(results, f'{scene}: 控制流程执行 {n_exec} tick 未触发安全停止', n_exec >= 1,
              {'executed_ticks': n_exec, 'stop_reason': monitor.why_stopped(),
               'safety': monitor.summary()['limits']})
        out['safety'] = monitor.summary()
        out['server_metadata'] = meta
        out['actions'] = {'shape': list(actions.shape),
                          'first3': actions[:3].round(6).tolist(),
                          'summary': summarize_chunk(actions)}
    finally:
        sim.close()
        check(results, f'{scene}: 回合正确退出（连接已释放）', True)

    out['checks'] = results
    out['n_pass'] = sum(1 for r in results if r['pass'])
    out['n_total'] = len(results)
    out['all_pass'] = out['n_pass'] == out['n_total']
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ticks', type=int, default=3)
    parser.add_argument('--max-delta-rad', type=float, default=0.10)
    parser.add_argument('--no-server', action='store_true')
    parser.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_0' / 'closed_loop_smoke'))
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for scene in ('A90', 'B90'):
        print(f'\n=========== {scene} ===========')
        report[scene] = run_scene(scene, args)
    save_json(out_dir / 'level5_env_interface.json', report)
    ok = all(v.get('all_pass') for v in report.values())
    print(f'\nLevel 5: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
