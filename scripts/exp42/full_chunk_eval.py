"""P0-b：16 步整段动作离线评价（不只是首行）。

用已有 `expert_state_diagnostic.npz`（step901）或 A/B 的 npz，逐行比较
模型 chunk 与专家 chunk，并把两种表示的标签解码成**物理关节目标**再比较
（审计 §二：首行两种表示相同，只有 1..15 行不同）。

指标：
  * 逐 k 关节 MAE / 方向余弦 / 幅值比；
  * 整段还原成物理目标后的 max/mean 误差（anchor: q0+a_k；increment: q0+cumsum(a)）；
  * 首行 vs 整段的误差占比（回答"失败只在首行还是整段"）。

用法（CPU）：
  python scripts/exp42/full_chunk_eval.py --npz results/.../expert_state_diagnostic.npz \
      --out results/.../full_chunk_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None,
                    help='与 --npz 对应的表示名（anchor / target_increment），默认 anchor')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    labels = args.labels or ['anchor'] * len(args.npz)

    out = {'files': []}
    for npz_path, label in zip(args.npz, labels):
        z = np.load(npz_path, allow_pickle=False)
        n = len(z['tick'])
        per_k = []
        for k in range(16):
            me = z['model_chunk'][:, k, :7]
            ex = z['expert_chunk'][:, k, :7]
            mae = float(np.abs(me - ex).mean())
            err = me - ex
            nz = np.abs(ex).max(axis=1) > 0.01
            cos = (float(np.mean([unit(a) @ unit(b) for a, b in zip(me[nz], ex[nz])]))
                   if nz.any() else None)
            mag = float(np.mean([np.linalg.norm(a) / max(np.linalg.norm(b), 1e-9)
                                 for a, b in zip(me, ex)]))
            per_k.append({'k': k, 'joint_mae': mae,
                          'cos': cos, 'mag_ratio': mag,
                          'model_absmax': float(np.abs(me).max()),
                          'expert_absmax': float(np.abs(ex).max())})

        # 物理目标还原误差（按表示）
        q = z['q']
        phys_model = np.zeros_like(z['model_chunk'][:, :, :7])
        phys_expert = np.zeros_like(z['expert_chunk'][:, :, :7])
        if label == 'target_increment':
            phys_model = q[:, None, :] + np.cumsum(z['model_chunk'][:, :, :7], axis=1)
            phys_expert = q[:, None, :] + np.cumsum(z['expert_chunk'][:, :, :7], axis=1)
        else:
            phys_model = q[:, None, :] + z['model_chunk'][:, :, :7]
            phys_expert = q[:, None, :] + z['expert_chunk'][:, :, :7]
        err = np.abs(phys_model - phys_expert)
        out['files'].append({
            'npz': str(npz_path), 'label': label, 'n_rows': int(n),
            'per_k': per_k,
            'first_row_mae': per_k[0]['joint_mae'],
            'rows_1_15_mae_mean': float(np.mean([p['joint_mae'] for p in per_k[1:]])),
            'phys_target_max_err_rad': float(err.max()),
            'phys_target_mean_err_rad': float(err.mean()),
            'phys_target_err_k0': float(err[:, 0].max()),
            'phys_target_err_k15': float(err[:, 15].max()),
            'model_gripper_absmax': float(np.abs(z['model_chunk'][:, :, 7]).max()),
        })
        print(f"{Path(npz_path).name} [{label}] rows={n} "
              f"MAE k0={per_k[0]['joint_mae']:.4f} rows1-15={out['files'][-1]['rows_1_15_mae_mean']:.4f} "
              f"phys_max={err.max():.3f} (k0 {err[:, 0].max():.3f}, k15 {err[:, 15].max():.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
