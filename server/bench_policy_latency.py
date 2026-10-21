"""测量 SmolVLA 部署时的推理延迟。

区分两种调用：
  * 新 chunk 的首次调用：真正跑一次策略前向（视觉编码 + 动作专家）；
  * chunk 内后续调用：直接取队列里的动作，几乎无计算。
评测脚本记录的 per-tick 延迟是两者的平均值，这里把两者分开报告。

用法：
    PYTHONPATH=. venv/bin/python server/bench_policy_latency.py \
        --policy outputs/smolvla_panda/checkpoints/015000/pretrained_model \
        --n-action-steps 10 --chunks 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vla_policy import SmolVLARunner  # noqa: E402
from vla.config import IMAGE_SIZE, TASK  # noqa: E402


def pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-action-steps', type=int, default=10)
    parser.add_argument('--chunks', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)

    import torch
    runner = SmolVLARunner(args.policy, args.device, n_action_steps=args.n_action_steps)

    rng = np.random.default_rng(0)
    height, width = IMAGE_SIZE
    frames = {
        'top': rng.integers(0, 255, (height, width, 3), dtype=np.uint8),
        'wrist': rng.integers(0, 255, (height, width, 3), dtype=np.uint8),
    }
    state = rng.normal(0.0, 0.2, 9).astype(np.float32)

    def timed_call():
        import time
        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        t0 = time.perf_counter()
        runner.act(frames, state)
        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        return (time.perf_counter() - t0) * 1000.0

    runner.reset()
    for _ in range(args.warmup):
        for _ in range(args.n_action_steps):
            timed_call()

    runner.reset()
    infer_ms, cached_ms = [], []
    for _ in range(args.chunks):
        infer_ms.append(timed_call())
        for _ in range(args.n_action_steps - 1):
            cached_ms.append(timed_call())

    result = {
        'policy': args.policy,
        'device': args.device,
        'n_action_steps': args.n_action_steps,
        'chunks': args.chunks,
        'task': TASK,
        'inference_ms_mean': float(np.mean(infer_ms)),
        'inference_ms_p50': pct(infer_ms, 50),
        'inference_ms_p95': pct(infer_ms, 95),
        'cached_action_ms_mean': float(np.mean(cached_ms)),
        'cached_action_ms_p95': pct(cached_ms, 95),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
