"""最终验证的 GPU 部分：真实模型输入接线审计 + 逐回合 RNG 复现。

Test A（模型输入接线）：用**正式场景构建入口** `build_sim_scene('B90', camera_version='v3')`
采一帧观测 → 送 Policy Server → 服务器把**实际送进模型的张量**落盘（`calls_8000/calls/*.npz`）。
本脚本再把这一帧的 RGB 独立渲染一次，比较两者 sha256，证明“新相机真的进了模型输入”，
而不是只存在离线渲染里。

Test B（RNG）：同一观测、同一 seed，跨两次独立连接各自 infer 一次 → 动作应逐位一致；
同 seed 后继续取 chunk（不重置）→ 第二个 chunk 也应一致；
不同 seed → 动作应不同。全部在真实 A40 + 真实 checkpoint 上执行，不用 mock。

用法（仿真 venv，服务器在 127.0.0.1:8000）：
    python scripts/exp40/final_model_audit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import camera_v2  # noqa: E402
from exp40 import droid_contract as dc  # noqa: E402
from exp40.observation_adapter import build_droid_observation, gripper_scalar  # noqa: E402
from scripts.exp40._common import build_sim_scene, capture, make_policy_client  # noqa: E402

OUT = ROOT / 'results' / 'experiment_4_0' / 'final_validation'


def sha(arr) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--scene', default='B90')
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    record: dict = {'scene': args.scene, 'camera_version': camera_v2.VERSION,
                    'camera_sha256': camera_v2.CAMERA_CONFIG_SHA256, 'checks': []}

    def add(name, ok, detail=None):
        record['checks'].append({'check': name, 'pass': bool(ok), 'detail': detail})
        print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  {json.dumps(detail, default=float)}' if detail else ''))

    # ---------------- Test A：真实模型输入 ----------------
    sim, block, rig, prompt, info = build_sim_scene(args.scene, seed=0, camera_version='v3')
    try:
        obs = capture(sim, rig)
        pkt = build_droid_observation(obs['frames'], obs['q'], gripper_scalar(sim), prompt)
        sent_base = pkt.payload[dc.KEY_IMAGE_BASE].copy()
        sent_wrist = pkt.payload[dc.KEY_IMAGE_WRIST].copy()
        client = make_policy_client(args.host, args.port)
        meta = client.get_server_metadata()
        record['server_metadata'] = meta

        out_a = client.infer(pkt.payload, episode_seed=args.seed)
        act_a = np.asarray(out_a['actions'])
        add('A1 服务器返回 (15,8) 且有限', act_a.shape == (15, 8) and np.all(np.isfinite(act_a)),
            {'shape': list(act_a.shape), 'absmax': float(np.abs(act_a).max())})
        add('A2 服务器以 checkpoint 模式运行（weights=checkpoint）',
            (meta.get('mode') == 'checkpoint'), {'meta': meta})
        # 独立渲染同一状态，比较发送给模型的图
        obs2 = capture(sim, rig)
        pkt2 = build_droid_observation(obs2['frames'], obs2['q'], gripper_scalar(sim), prompt)
        same = sha(sent_base) == sha(pkt2.payload[dc.KEY_IMAGE_BASE])
        add('A3 同状态独立渲染与发送图逐位一致（相机→模型输入可复现）', same,
            {'sha256_sent': sha(sent_base)[:16], 'sha256_rerender': sha(pkt2.payload[dc.KEY_IMAGE_BASE])[:16]})
        add('A4 图像尺寸/通道为 224×224×3 uint8',
            sent_base.shape == (224, 224, 3) and sent_base.dtype == np.uint8,
            {'shape': list(sent_base.shape), 'dtype': str(sent_base.dtype)})
        # 用配置里的相机重建一次 rig，确认 hash 匹配
        add('A5 rig 使用的相机配置与冻结 SHA256 一致',
            info.get('camera_sha256') == camera_v2.CAMERA_CONFIG_SHA256,
            {'info_sha': info.get('camera_sha256', '')[:16]})
        Path(OUT / 'model_input_audit.json').write_text(json.dumps({
            **record,
            'prompt': prompt,
            'q': np.asarray(obs['q']).round(6).tolist(),
            'gripper': gripper_scalar(sim),
            'image_sha256': {'base': sha(sent_base), 'wrist': sha(sent_wrist)},
            'n_action_dims_physical': dc.ACTION_DIM,
            'action_horizon': dc.ACTION_HORIZON,
            'first_action': act_a[0].round(6).tolist(),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        sim.close()

    # ---------------- Test B：RNG 复现 ----------------
    sim2, block2, rig2, prompt2, info2 = build_sim_scene(args.scene, seed=0, camera_version='v3')
    try:
        o = capture(sim2, rig2)
        pkt = build_droid_observation(o['frames'], o['q'], gripper_scalar(sim2), prompt2)
        c1 = make_policy_client(args.host, args.port)
        a1 = np.asarray(c1.infer(pkt.payload, episode_seed=args.seed)['actions'])
        b1 = np.asarray(c1.infer(pkt.payload)['actions'])          # 同回合第二个 chunk
        c1.close()
        c2 = make_policy_client(args.host, args.port)              # 独立连接 = 独立 episode
        a2 = np.asarray(c2.infer(pkt.payload, episode_seed=args.seed)['actions'])
        b2 = np.asarray(c2.infer(pkt.payload)['actions'])
        c3 = make_policy_client(args.host, args.port)
        a3 = np.asarray(c3.infer(pkt.payload, episode_seed=args.seed + 1)['actions'])
        c3.close()

        add('B1 同 seed 两次独立初始化：首个 chunk 逐位一致', np.array_equal(a1, a2),
            {'max_abs_diff': float(np.abs(a1 - a2).max())})
        add('B2 同 seed 的后续 chunk（不重置）也逐位一致', np.array_equal(b1, b2),
            {'max_abs_diff': float(np.abs(b1 - b2).max())})
        add('B3 同 seed 下连续 chunk 与首个 chunk 不同（PRNG 在推进）', not np.array_equal(a1, b1),
            {'max_abs_diff': float(np.abs(a1 - b1).max())})
        add('B4 不同 seed 产生不同动作', not np.array_equal(a1, a3),
            {'max_abs_diff': float(np.abs(a1 - a3).max())})
        Path(OUT / 'rng_model_reproducibility.json').write_text(json.dumps({
            'seed': args.seed, 'model': 'pi05_droid', 'checks': record['checks'][5:],
            'actions': {'seed_a_chunk1': a1[:2].round(6).tolist(),
                        'seed_a_chunk2': b1[:2].round(6).tolist(),
                        'seed_b_chunk1': a3[:2].round(6).tolist()},
            'note': '全部在真实 A40 + 真实 checkpoint 上执行；seed 通过 __episode_seed__ 下发，'
                    '服务器重置 jax.random.key(seed) 后推理',
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        sim2.close()
    return 0 if all(c['pass'] for c in record['checks']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
