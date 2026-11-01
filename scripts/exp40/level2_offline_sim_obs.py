"""Level 2：用真实 PyBullet 观测做离线推理（不执行模型动作）。

流程：
    仿真场景 → CameraRig 采图 → ObservationAdapter → DROID 键名 payload
    → （可选）websocket 策略服务器 → action chunk
并保存：原始图、实际发送图、state、prompt、sha256、动作 chunk、延迟。

网络不通或服务器未启动时，用 --no-server 只验证观测链路并落盘（明确标记为未推理）。

用法：
    python scripts/exp40/level2_offline_sim_obs.py --scene B90 --port 8000
"""
from __future__ import annotations

import argparse
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
from exp40.action_adapter import chunk_to_targets, summarize_chunk, ActionAdapterConfig  # noqa: E402
from exp40.observation_adapter import build_droid_observation, gripper_scalar  # noqa: E402
from scripts.exp40._common import (  # noqa: E402
    build_sim_scene, capture, make_policy_client, model_input_summary, out_dir, save_json,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='B90', choices=['A90', 'B90'])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-server', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_0' / 'observation_checks'))
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sim, block, rig, prompt, info = build_sim_scene(args.scene, seed=args.seed)
    record = {'scene': args.scene, 'seed': args.seed, 'scene_info': info, 'prompt': prompt}
    try:
        obs = capture(sim, rig)
        g = gripper_scalar(sim)
        pkt = build_droid_observation(obs['frames'], obs['q'], g, prompt)

        # 保存原始图与实际发送图（PNG 便于人工核对）
        from PIL import Image

        Image.fromarray(obs['frames']['top']).save(out / f'{args.scene}_raw_top.png')
        Image.fromarray(obs['frames']['wrist']).save(out / f'{args.scene}_raw_wrist.png')
        Image.fromarray(pkt.payload[dc.KEY_IMAGE_BASE]).save(out / f'{args.scene}_sent_base_224.png')
        Image.fromarray(pkt.payload[dc.KEY_IMAGE_WRIST]).save(out / f'{args.scene}_sent_wrist_224.png')

        record['observation'] = {
            'summary': model_input_summary(pkt.payload),
            'adapter_audit': pkt.audit,
            'q_rad': np.asarray(obs['q']).round(6).tolist(),
            'fingers_m': np.asarray(obs['fingers']).round(6).tolist(),
            'ee_position': np.asarray(obs['ee_position']).round(6).tolist(),
            'ee_quat': np.asarray(obs['ee_quat']).round(6).tolist(),
            'block_position': info['block_start'],
            'block_yaw': info['block_yaw'],
        }

        if args.no_server:
            record['inference'] = {'executed': False, 'reason': '--no-server'}
            save_json(out / f'level2_observation_{args.scene}.json', record)
            print('观测链路 OK（未调用模型）')
            print(json.dumps(record['observation']['summary']['shapes'], indent=2))
            return 0

        client = make_policy_client(args.host, args.port)
        meta = client.get_server_metadata()
        record['server_metadata'] = meta
        t0 = time.perf_counter()
        result = client.infer(pkt.payload)
        dt = time.perf_counter() - t0
        actions = np.asarray(result['actions'])
        record['inference'] = {
            'executed': True,
            'roundtrip_s': round(dt, 4),
            'weights': meta.get('weights', 'unknown'),
            'chunk_summary': summarize_chunk(actions),
            'chunk_first3': actions[:3].round(6).tolist(),
            'policy_timing': result.get('policy_timing'),
        }
        np.savez_compressed(out / f'level2_chunk_{args.scene}.npz', actions=actions,
                            q=np.asarray(obs['q']), gripper=np.array([g]))
        # 顺带演示适配（不执行）
        adapted = chunk_to_targets(actions, obs['q'], sim.lower, sim.upper, ActionAdapterConfig())
        record['adaptation_preview'] = {
            'n': len(adapted),
            'accepted_frac': float(np.mean([a.accepted for a in adapted])),
            'first_q_target': adapted[0].q_target.round(6).tolist(),
            'first_gripper_m': adapted[0].gripper_target_m,
            'reasons': sorted({r for a in adapted for r in a.reasons}),
        }
        save_json(out / f'level2_observation_{args.scene}.json', record)
        print(json.dumps(record['inference'], ensure_ascii=False, indent=2, default=float))
        return 0
    finally:
        sim.close()


if __name__ == '__main__':
    raise SystemExit(main())
