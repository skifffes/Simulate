"""Level 1：π₀.₅ 独立推理验证（在 openpi venv 里跑，不涉及 PyBullet）。

验证 prompt 5.3 要求的全部项目：
    模型加载 / 输入键名 / tensor shape / dtype / 视觉预处理 / normalization /
    action horizon 与 action dimension / 输出含义 / 无 NaN·Inf / 可重复 /
    GPU 显存峰值 / 首次推理时间 / 稳态推理时间 / 保存完整推理元数据与动作样本

两种权重来源：
    --mode checkpoint : 官方 pi05_droid
    --mode random     : 官方配置 + 官方 norm_stats + 随机权重（只验证管线，不验证能力）

用法：
    python scripts/exp40/level1_model_inference.py --mode checkpoint \
        --out results/experiment_4_0/model_smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import droid_contract as dc  # noqa: E402


def gpu_mem_used_mib(pid: int | None = None) -> float | None:
    """只统计**本进程**的显存占用（避免把同机其它进程算进来）。

    早期版本统计的是所有 compute apps 的总和，导致同一台机器上同时跑策略服务器时
    读数会翻倍（实测出现过 55016 MiB > 单卡 46068 MiB 的不可能值）。这里按 pid 过滤。
    """
    import os

    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout
        total = 0
        found = False
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            if int(parts[0]) == pid:
                total += int(parts[1])
                found = True
        return float(total) if found else 0.0
    except Exception:
        return None


def build_obs(seed: int, height: int = 224, width: int = 224) -> dict:
    """完全符合 DROID 官方 example 的输入（键名/形状/dtype 一致）。"""
    rng = np.random.default_rng(seed)
    return {
        dc.KEY_IMAGE_BASE: rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        dc.KEY_IMAGE_WRIST: rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        dc.KEY_JOINT_POSITION: rng.uniform(-1.5, 1.5, 7).astype(np.float32),
        dc.KEY_GRIPPER_POSITION: np.array([0.5], dtype=np.float32),
        dc.KEY_PROMPT: dc.PROMPT_B90,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='pi05_droid')
    parser.add_argument('--mode', choices=['checkpoint', 'random'], default='checkpoint')
    parser.add_argument('--checkpoint', default='gs://openpi-assets/checkpoints/pi05_droid')
    parser.add_argument('--norm-stats', default='')
    parser.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_0' / 'model_smoke'))
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(ROOT / 'scripts' / 'exp40'))

    record: dict = {
        'mode': args.mode,
        'config': args.config,
        'checkpoint': args.checkpoint if args.mode == 'checkpoint' else f'random_init',
        'openpi_commit': dc.OPENPI_COMMIT,
        'simulate_commit': _git_rev(),
        'python': sys.version.split()[0],
        'jax': jax.__version__,
        'callback_begin': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    print('=== jax devices ===')
    print(jax.devices())
    record['jax_devices'] = [str(d) for d in jax.devices()]
    record['jax_backend'] = jax.default_backend()

    # ---- 构建策略（与官方 serve_policy 使用同一条 create_trained_policy 路径） ----
    from serve_policy import build_random_policy  # 复用 random 模式的构造逻辑

    import openpi.training.config as _config
    from openpi.policies import policy_config as _policy_config

    train_config = _config.get_config(args.config)
    record['action_dim_model'] = train_config.model.action_dim
    record['action_horizon_model'] = train_config.model.action_horizon
    record['normalization'] = 'quantile' if train_config.data.create(
        train_config.assets_dirs, train_config.model).use_quantile_norm else 'zscore'

    t_load0 = time.perf_counter()
    if args.mode == 'checkpoint':
        policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
        weights = 'checkpoint'
    else:
        from openpi.shared import normalize as _normalize

        if not args.norm_stats:
            raise SystemExit('random 模式需要 --norm-stats <dir containing norm_stats.json>')
        norm_stats = _normalize.load(pathlib.Path(args.norm_stats))
        policy = build_random_policy(train_config, norm_stats)
        weights = 'random_init'
    load_s = time.perf_counter() - t_load0
    record['weights'] = weights
    record['load_seconds'] = round(load_s, 3)
    record['gpu_mem_after_load_mib'] = gpu_mem_used_mib()
    print(f'policy ready ({weights}) in {load_s:.1f}s, gpu_mem={record["gpu_mem_after_load_mib"]} MiB')

    # ---- 记录真实模型输入（后置 transform 的输出） ----
    captured = {}
    orig = policy._input_transform

    def spy(data):
        res = orig(data)
        captured['state'] = np.asarray(res['state'])
        captured['image'] = {k: np.asarray(v) for k, v in res['image'].items()}
        captured['image_mask'] = {k: bool(np.asarray(v)) for k, v in res['image_mask'].items()}
        captured['prompt'] = str(res.get('prompt', ''))
        for k in ('tokenized_prompt', 'tokenized_prompt_mask'):
            if k in res:
                captured[k] = np.asarray(res[k])
        return res

    policy._input_transform = spy

    obs = build_obs(0)
    record['request'] = {
        'keys': sorted(obs),
        'shapes': {k: list(np.asarray(v).shape) for k, v in obs.items() if k != dc.KEY_PROMPT},
        'dtypes': {k: str(np.asarray(v).dtype) for k, v in obs.items() if k != dc.KEY_PROMPT},
        'prompt': obs[dc.KEY_PROMPT],
    }

    # ---- 首次推理 ----
    t0 = time.perf_counter()
    out1 = policy.infer(obs)
    first_s = time.perf_counter() - t0
    actions1 = np.asarray(out1['actions'])
    record['first_inference_seconds'] = round(first_s, 3)
    record['gpu_mem_after_first_infer_mib'] = gpu_mem_used_mib()

    record['model_inputs'] = {
        'state': captured['state'].round(6).tolist(),
        'state_shape': list(captured['state'].shape),
        'state_dtype': str(captured['state'].dtype),
        'state_note': '归一化后的 8 维 state 再 pad 到 action_dim=32（后 24 维为 0）',
        'image_shapes': {k: list(v.shape) for k, v in captured['image'].items()},
        'image_dtypes': {k: str(v.dtype) for k, v in captured['image'].items()},
        'image_dtype_note': 'ResizeImages 之后是 float32，值域 [-1, 1]' ,
        'image_range': {k: [float(v.min()), float(v.max())] for k, v in captured['image'].items()},
        'image_masks': captured['image_mask'],
        'prompt': captured['prompt'],
    }
    for k in ('tokenized_prompt', 'tokenized_prompt_mask'):
        if k in captured:
            record['model_inputs'][k] = captured[k].tolist()

    record['output'] = {
        'shape': list(actions1.shape),
        'dtype': str(actions1.dtype),
        'finite': bool(np.all(np.isfinite(actions1))),
        'absmax': float(np.abs(actions1).max()),
        'first_action': actions1[0].round(6).tolist(),
        'per_dim_q01_q99': [[float(np.quantile(actions1[:, i], 0.01)),
                             float(np.quantile(actions1[:, i], 0.99))] for i in range(actions1.shape[1])],
        'policy_timing': out1.get('policy_timing'),
    }

    # ---- 可重复性：固定噪声后两次推理必须逐位一致 ----
    # 注意 noise 必须按模型内部动作宽度 action_dim(=32) 给，而不是 8（8 只是物理有效维度）
    noise = np.zeros((1, actions1.shape[0], train_config.model.action_dim), dtype=np.float32)
    try:
        a = np.asarray(policy.infer(obs, noise=noise.copy())['actions'])
        b = np.asarray(policy.infer(obs, noise=noise.copy())['actions'])
        record['determinism'] = {'identical': bool(np.array_equal(a, b)),
                                 'max_abs_diff': float(np.abs(a - b).max())}
    except Exception as exc:  # pragma: no cover
        record['determinism'] = {'error': f'{type(exc).__name__}: {exc}'}

    # ---- 稳态推理时间 ----
    times = []
    for _ in range(args.warmup):
        policy.infer(obs)
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        policy.infer(obs)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    record['steady_state'] = {
        'n': int(times.size),
        'mean_ms': round(float(times.mean() * 1000), 2),
        'p50_ms': round(float(np.percentile(times, 50) * 1000), 2),
        'p95_ms': round(float(np.percentile(times, 95) * 1000), 2),
        'max_ms': round(float(times.max() * 1000), 2),
        'hz': round(float(1.0 / times.mean()), 2),
    }
    record['gpu_mem_after_repeats_mib'] = gpu_mem_used_mib()
    record['gpu_mem_peak_mib'] = max(v for v in (
        record.get('gpu_mem_after_load_mib'), record.get('gpu_mem_after_first_infer_mib'),
        record.get('gpu_mem_after_repeats_mib')) if v is not None)

    # ---- 输出随机输入下的动作统计（明确标注不是有效控制策略） ----
    np.savez_compressed(out_dir / 'action_sample.npz',
                        actions=actions1,
                        state=np.asarray(obs[dc.KEY_JOINT_POSITION]),
                        gripper=np.asarray(obs[dc.KEY_GRIPPER_POSITION]),
                        image_base=obs[dc.KEY_IMAGE_BASE],
                        image_wrist=obs[dc.KEY_IMAGE_WRIST])
    (out_dir / 'model_inputs.npz').unlink(missing_ok=True)
    np.savez_compressed(out_dir / 'model_inputs.npz',
                        state=captured['state'],
                        **{f'image_{k}': v for k, v in captured['image'].items()})
    with open(out_dir / 'level1_inference.json', 'w', encoding='utf-8') as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2, default=float)

    print(json.dumps({k: record[k] for k in (
        'weights', 'load_seconds', 'first_inference_seconds', 'steady_state',
        'gpu_mem_peak_mib', 'normalization', 'action_dim_model', 'action_horizon_model')},
        ensure_ascii=False, indent=2))
    print('output shape', record['output']['shape'], 'finite', record['output']['finite'])
    print(f'saved -> {out_dir}')
    return 0 if record['output']['finite'] else 1


def _git_rev() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    raise SystemExit(main())
