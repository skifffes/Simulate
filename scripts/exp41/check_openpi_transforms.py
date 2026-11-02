"""在 OpenPI 训练环境里验证 8 维 state/action 能通过真实 transforms（quantile 归一化）。

只做前向检查（不训练）：把一条 A90 episode 的导出数组喂给 openpi 的 Normalize/Unnormalize，
确认维度、dtype、量化范围都正确。

用法（openpi venv）：
    /scratch/lijh/exp40/openpi/.venv/bin/python scripts/exp41/check_openpi_transforms.py \
        --data data/exp41 --out results/experiment_4_1/preflight
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

from exp41.training_export import build_training_arrays  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/exp41')
    ap.add_argument('--out', default='results/experiment_4_1/preflight')
    ap.add_argument('--npz', default=None, help='预先导出的 state8/action8（openpi venv 无 pandas 时用）')
    args = ap.parse_args(argv)
    res = {'openpi_importable': False}
    try:
        import openpi.transforms as T
        res['openpi_importable'] = True
    except Exception as exc:  # noqa: BLE001
        res['import_error'] = str(exc)[:300]
        (Path(args.out) / 'openpi_transform_check.json').write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(res, ensure_ascii=False))
        return 0

    if args.npz:
        z = np.load(args.npz)
        state8, action8 = z['state'].astype(np.float64), z['action'].astype(np.float64)
    else:
        import pandas as pd

        data = Path(args.data)
        root = data / 'A90'
        df = pd.read_parquet(root / 'data' / 'chunk-000' / 'episode_000000.parquet',
                             columns=['observation.state', 'action'])
        S = np.stack(df['observation.state'].to_numpy()).astype(np.float64)
        A = np.stack(df['action'].to_numpy()).astype(np.float64)
        arr = build_training_arrays(S, A)
        state8, action8 = arr['state'], arr['action']

    # 用本数据集自身的 q01/q99 构造 norm_stats（形状与 openpi 期望一致：每维一串分位数）
    q01 = np.quantile(state8, 0.01, axis=0)
    q99 = np.quantile(state8, 0.99, axis=0)
    aq01 = np.quantile(action8, 0.01, axis=0)
    aq99 = np.quantile(action8, 0.99, axis=0)
    from openpi.shared.normalize import NormStats
    stats = {'state': NormStats(mean=state8.mean(0), std=state8.std(0) + 1e-6,
                                q01=q01, q99=q99),
             'actions': NormStats(mean=action8.mean(0), std=action8.std(0) + 1e-6,
                                  q01=aq01, q99=aq99)}

    res.update({
        'state_shape': list(state8.shape), 'action_shape': list(action8.shape),
        'state_dtype': str(state8.dtype), 'action_dtype': str(action8.dtype),
        'state8_gripper_minmax': [float(state8[:, 7].min()), float(state8[:, 7].max())],
        'action8_gripper_minmax': [float(action8[:, 7].min()), float(action8[:, 7].max())],
        'action8_dq_absmax': float(np.abs(action8[:, :7]).max()),
    })
    try:
        norm = T.Normalize(stats)
        unnorm = T.Unnormalize(stats)
        item = {'state': state8[0].copy(), 'actions': action8.copy()}
        n = norm(dict(item))
        res['normalized_state_range'] = [float(np.min(n['state'])), float(np.max(n['state']))]
        res['normalized_action_range'] = [float(np.min(n['actions'])), float(np.max(n['actions']))]
        back = unnorm({'state': n['state'].copy(), 'actions': n['actions'].copy()})
        res['roundtrip_state_max_abs_err'] = float(np.max(np.abs(back['state'] - state8[0])))
        res['roundtrip_action_max_abs_err'] = float(np.max(np.abs(back['actions'] - action8)))
        res['pass'] = bool(res['roundtrip_state_max_abs_err'] < 1e-5
                           and res['roundtrip_action_max_abs_err'] < 1e-5)
    except Exception as exc:  # noqa: BLE001
        res['transform_error'] = str(exc)[:400]
        res['pass'] = False
    (Path(args.out) / 'openpi_transform_check.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, default=float)[:600])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
