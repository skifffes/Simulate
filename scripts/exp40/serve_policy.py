"""π₀.₅ (pi05_droid) 策略服务器：官方 websocket 协议 + 输入/输出审计。

两种模式：
  --mode checkpoint : 加载官方 gs://openpi-assets/checkpoints/pi05_droid（生产路径）
  --mode random     : 用官方配置与官方 norm_stats，但权重随机初始化
                      —— 只用于验证“观测→模型→动作→客户端”这段管线是否接对，
                         绝不能用来判断 π₀.₅ 的能力。每条记录都会写 weights="random_init"。

审计钩子位置：`policy._input_transform` 的返回值（即已经过 DroidInputs → Normalize →
ResizeImages(224) → TokenizePrompt → PadStatesAndActions(32) 之后、加 batch 维之前的真实模型输入），
以及 `infer()` 返回的动作 chunk。这样记录的是模型真正看到/输出的东西，而不是另建一套观测。

用法（在 openpi venv 中）：
    uv run --no-project python scripts/exp40/serve_policy.py --mode checkpoint --port 8000 --dump-dir ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_norm_stats(path: pathlib.Path):
    from openpi.shared import normalize as _normalize

    # 官方 loader 读的是 <dir>/norm_stats.json，这里直接把 droid.json 放进临时目录
    return _normalize.load(path)


def build_random_policy(train_config, norm_stats, seed: int = 0):
    """官方配置 + 官方 norm_stats + 随机权重。仅用于管线联调。"""
    import jax
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    model = train_config.model.create(jax.random.key(seed))
    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata={**(train_config.policy_metadata or {}), 'weights': 'random_init'},
    )


class AuditingPolicy:
    """在官方 Policy 外面加一层记录：不改变任何计算。"""

    def __init__(self, policy, dump_dir: pathlib.Path | None, weights: str, max_records: int = 200):
        self._policy = policy
        self._dump_dir = pathlib.Path(dump_dir) if dump_dir else None
        self._weights = weights
        self._n = 0
        self._max = max_records
        self.calls: list[dict] = []
        if self._dump_dir:
            (self._dump_dir / 'calls').mkdir(parents=True, exist_ok=True)
        self._orig_transform = policy._input_transform
        policy._input_transform = self._recording_transform

    def _recording_transform(self, data):
        out = self._orig_transform(data)
        self._last_inputs = {
            'state': np.asarray(out['state']),
            'image': {k: np.asarray(v) for k, v in out['image'].items()},
            'image_mask': {k: bool(np.asarray(v)) for k, v in out['image_mask'].items()},
            'prompt': str(out.get('prompt', '')),
            'tokenized_prompt_shape': list(np.asarray(out['tokenized_prompt']).shape)
            if 'tokenized_prompt' in out else None,
        }
        return out

    @property
    def metadata(self) -> dict:
        return {**self._policy.metadata, 'weights': self._weights}

    def infer(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        out = self._policy.infer(obs)
        dt = time.perf_counter() - t0
        self._n += 1
        record = {
            'call': self._n,
            'weights': self._weights,
            'wall_time_s': dt,
            'infer_ms_reported': out.get('policy_timing', {}).get('infer_ms'),
            'request_keys': sorted(k for k in obs),
            'request_shapes': {
                k: (list(np.asarray(v).shape) if k != 'prompt' else None)
                for k, v in obs.items()
            },
            'request_dtypes': {
                k: str(np.asarray(v).dtype) for k, v in obs.items() if k != 'prompt'
            },
            'prompt': obs.get('prompt'),
            'model_inputs': {
                'state': np.asarray(self._last_inputs['state']).round(6).tolist(),
                'state_shape': list(np.asarray(self._last_inputs['state']).shape),
                'image_shapes': {k: list(v.shape) for k, v in self._last_inputs['image'].items()},
                'image_dtypes': {k: str(v.dtype) for k, v in self._last_inputs['image'].items()},
                'image_masks': self._last_inputs['image_mask'],
                'image_pixel_range': {k: [float(v.min()), float(v.max())] for k, v in self._last_inputs['image'].items()},
                'tokenized_prompt_shape': self._last_inputs['tokenized_prompt_shape'],
            },
            'actions': {
                'shape': list(np.asarray(out['actions']).shape),
                'dtype': str(np.asarray(out['actions']).dtype),
                'finite': bool(np.all(np.isfinite(out['actions']))),
                'first_action': np.asarray(out['actions'])[0].round(6).tolist(),
                'absmax': float(np.abs(np.asarray(out['actions'])).max()),
            },
        }
        self.calls.append(record)
        if self._dump_dir and self._n <= self._max:
            np.savez_compressed(
                self._dump_dir / 'calls' / f'call_{self._n:04d}.npz',
                state=np.asarray(self._last_inputs['state']),
                **{f'image/{k}': v for k, v in self._last_inputs['image'].items()},
                actions=np.asarray(out['actions']),
            )
            with open(self._dump_dir / 'calls' / f'call_{self._n:04d}.json', 'w', encoding='utf-8') as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2, default=float)
        return out

    def reset(self) -> None:
        self.calls.clear()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='pi05_droid')
    parser.add_argument('--mode', choices=['checkpoint', 'random'], default='checkpoint')
    parser.add_argument('--checkpoint', default='gs://openpi-assets/checkpoints/pi05_droid')
    parser.add_argument('--norm-stats', default='', help='random 模式下用的 norm_stats.json 目录')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--dump-dir', default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    import jax
    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server

    logging.info('jax devices: %s', jax.devices())
    train_config = _config.get_config(args.config)

    if args.mode == 'checkpoint':
        policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
        wrapped = AuditingPolicy(policy, args.dump_dir, weights='checkpoint')
        ckpt_id = args.checkpoint
    else:
        if not args.norm_stats:
            raise SystemExit('random 模式必须提供 --norm-stats（官方 assets/droid 目录）')
        norm_stats = _build_norm_stats(pathlib.Path(args.norm_stats))
        policy = build_random_policy(train_config, norm_stats, seed=args.seed)
        wrapped = AuditingPolicy(policy, args.dump_dir, weights='random_init')
        ckpt_id = f'random_init(seed={args.seed})'

    server = websocket_policy_server.WebsocketPolicyServer(
        wrapped, host=args.host, port=args.port, metadata={
            'config': args.config,
            'mode': args.mode,
            'checkpoint': ckpt_id,
            'openpi_commit': os.environ.get('OPENPI_COMMIT', ''),
        },
    )
    logging.info('serving on %s:%d (mode=%s)', args.host, args.port, args.mode)
    server.serve_forever()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
