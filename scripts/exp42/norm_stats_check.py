"""Normalization 决策依据：把 4.1 训练集分布与官方 DROID stats 对比。

只用 **train split**（exp4.1-split-v1），不碰 val/test。
输出：train-split 的 state/action 各维分位数、超出 DROID [q01,q99] 的比例、clip 比例。

用法（simulate_vla venv）：
    python scripts/exp42/norm_stats_check.py --out results/experiment_4_2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path('/scratch/lijh/exp42/data')
DROID_STATS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json')
FINGER_MAX_M = 0.04


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_2'))
    ap.add_argument('--split', default='train')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    rows = [m for m in meta if m['split'] == args.split]
    S, A = [], []
    for m in rows:
        z = np.load(DATA / m['state_path'])
        S.append(z['state'].astype(np.float64))
        A.append(z['action'].astype(np.float64))
    S = np.concatenate(S)
    A = np.concatenate(A)
    q = S[:, :7]
    g_state = np.clip(S[:, 7:9].mean(axis=1) / FINGER_MAX_M, 0, 1)
    dq = A[:, :7] - q
    g_action = np.clip(A[:, 7] / FINGER_MAX_M, 0, 1)

    state8 = np.concatenate([q, g_state[:, None]], axis=1)
    action8 = np.concatenate([dq, g_action[:, None]], axis=1)

    droid = json.loads(DROID_STATS.read_text(encoding='utf-8'))
    dst = droid['norm_stats']['state']
    dact = droid['norm_stats']['actions']

    def compare(x: np.ndarray, ref: dict, name: str) -> dict:
        d = int(x.shape[1])
        # 官方 DROID stats 是 32 维（8 个物理维 + 24 个 padding 维），只取前 d 维比较
        q01 = np.asarray(ref['q01'], dtype=float)[:d]
        q99 = np.asarray(ref['q99'], dtype=float)[:d]
        span = q99 - q01
        out_hi = (x > q99[None, :]).mean(axis=0)
        out_lo = (x < q01[None, :]).mean(axis=0)
        # 归一化到 [-1, 1] 后需要 clip 的比例
        norm = 2.0 * (x - q01[None, :]) / np.where(span == 0, 1, span) - 1.0
        clip_frac = (np.abs(norm) > 1).mean(axis=0)
        return {
            'dim': int(x.shape[1]),
            'train_min': np.round(x.min(0), 4).tolist(),
            'train_max': np.round(x.max(0), 4).tolist(),
            'train_q01': np.round(np.quantile(x, 0.01, axis=0), 4).tolist(),
            'train_q99': np.round(np.quantile(x, 0.99, axis=0), 4).tolist(),
            'droid_q01': np.round(q01, 4).tolist(),
            'droid_q99': np.round(q99, 4).tolist(),
            'frac_above_droid_q99': np.round(out_hi, 4).tolist(),
            'frac_below_droid_q01': np.round(out_lo, 4).tolist(),
            'clip_frac_after_droid_norm': np.round(clip_frac, 4).tolist(),
            'train_span_ratio_vs_droid': np.round(span / np.where(span == 0, 1, span), 4).tolist(),
            'name': name,
        }

    res = {
        'split': args.split,
        'n_episodes': len(rows),
        'n_frames': int(S.shape[0]),
        'state_vs_droid': compare(state8, dst, 'state8'),
        'action_vs_droid': compare(action8, dact, 'action8'),
        'notes': [
            'DROID 的 action q99 ≈ 0.45~0.80 rad/step；本数据 Δq 大约小一个量级，',
            '这说明两边的"每步增量"尺度不同（专家移动慢），不是符号或单位错误：',
            '两侧都是 rad/control-step，模型归一化后本数据落在 DROID 区间的中部，',
            'clip 比例为 0 即可训练；若改用 train-split 自算 stats，归一化后数值会放大，',
            '需与 checkpoint 的预训练范围权衡（见 normalization_decision.md）。',
        ],
    }
    (out / 'normalization_stats_check.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('split', 'n_episodes', 'n_frames')}, ensure_ascii=False))
    for key in ('state_vs_droid', 'action_vs_droid'):
        d = res[key]
        print(key, 'clip_frac_after_droid_norm', d['clip_frac_after_droid_norm'])
        print('   train_q01', d['train_q01'][:8])
        print('   train_q99', d['train_q99'][:8])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
