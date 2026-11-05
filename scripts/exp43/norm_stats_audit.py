"""train-only 动作统计诊断（**只读**，不改 DROID normalization）。

对合并导出（314 条 train）计算：
  * 单步 Δq 与 16 步 anchor 标签的分布；
  * 与 DROID q01/q99 量程的比值（按关节）；
  * 归一化后是否越界（clip 比例）；
  * A90/B90 分开统计。

用法（服务器）：
  python scripts/exp43/norm_stats_audit.py --data /scratch/lijh/exp43/export \
      --droid /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HORIZON = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/scratch/lijh/exp43/export')
    ap.add_argument('--droid', default='/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/'
                                        'pi05_droid/assets/droid/norm_stats.json')
    ap.add_argument('--out', default='results/experiment_4_3/data_expansion/'
                                      'train_norm_stats_audit.json')
    args = ap.parse_args()
    data = Path(args.data)
    meta = [json.loads(l) for l in (data / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    train = [m for m in meta if m.get('split') == 'train']
    droid = json.loads(Path(args.droid).read_text(encoding='utf-8'))
    q01 = np.asarray(droid['norm_stats']['actions']['q01'][:7], dtype=float)
    q99 = np.asarray(droid['norm_stats']['actions']['q99'][:7], dtype=float)
    denom = np.maximum(q99 - q01, 1e-9)

    res = {'n_train': len(train), 'by_skill': {}, 'anchor16': {}, 'droid': {
        'q01': q01.tolist(), 'q99': q99.tolist()}}
    for skill in ('A90', 'B90'):
        step, anchor, clip_frac = [], [], []
        for m in train:
            if m['skill'] != skill:
                continue
            z = np.load(data / m['state_path'])
            s = z['state'].astype(float)
            a = z['action'].astype(float)
            dq = a[:, :7] - s[:, :7]
            step.append(dq)
            T = len(s) - HORIZON + 1
            if T > 0:
                for t in range(0, T, 4):
                    anchor.append(a[t:t + HORIZON, :7] - s[t, :7][None, :])
        if not step:
            continue
        step = np.concatenate(step)
        anc = np.concatenate(anchor) if anchor else np.zeros((1, HORIZON, 7))
        norm = 2 * (anc - q01) / denom - 1
        res['by_skill'][skill] = {
            'n_episodes': sum(1 for m in train if m['skill'] == skill),
            'step_frames': int(step.shape[0]),
            'step_abs_mean': float(np.abs(step).mean()),
            'step_q01': np.quantile(step, .01, axis=0).tolist(),
            'step_q99': np.quantile(step, .99, axis=0).tolist(),
            'step_qspan_over_droid': ((np.quantile(step, .99, axis=0)
                                        - np.quantile(step, .01, axis=0)) / denom).tolist(),
        }
        res['anchor16'][skill] = {
            'windows': int(anc.shape[0]),
            'abs_mean_k0': float(np.abs(anc[:, 0]).mean()),
            'abs_mean_k15': float(np.abs(anc[:, 15]).mean()),
            'qspan_over_droid': ((np.quantile(anc, .99, axis=(0, 1))
                                   - np.quantile(anc, .01, axis=(0, 1))) / denom).tolist(),
            'normalized_frac_outside_pm1': float((np.abs(norm) > 1).mean()),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'n_train': res['n_train'],
                      'by_skill': {k: {kk: vv for kk, vv in v.items() if kk != 'step_qspan_over_droid'}
                                   for k, v in res['by_skill'].items()},
                      'anchor16': res['anchor16']}, ensure_ascii=False, indent=2)[:1800])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
