"""Gate 4 A/B 离线对照：把 A(anchor) 与 B(target_increment) 在**同一批专家状态**上的
首行动作误差/方向/幅值并排比较（输入来自 expert_state_diagnostic.npz 的同一 picks）。

用法：
  python scripts/exp42/ab_repr_compare.py --a results/.../expert_state_analysis_A.json \
      --b results/.../expert_state_analysis_B.json --out results/.../ab_repr_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    A = json.loads(Path(args.a).read_text(encoding='utf-8'))['by_phase']
    B = json.loads(Path(args.b).read_text(encoding='utf-8'))['by_phase']

    rows = []
    for key in sorted(set(A) | set(B)):
        a, b = A.get(key), B.get(key)
        if not a or not b:
            continue
        rows.append({
            'phase': key, 'n_states': a['n_states'],
            'expert_absmax': a['expert_absmax_median'],
            'A_model_absmax': a['model_absmax_median'],
            'B_model_absmax': b['model_absmax_median'],
            'A_mag_ratio': a['mag_ratio_median'],
            'B_mag_ratio': b['mag_ratio_median'],
            'A_joint_mae': a['joint_mae_median'],
            'B_joint_mae': b['joint_mae_median'],
            'A_cos': a['cos_joint_median'], 'B_cos': b['cos_joint_median'],
            'A_ee_cos': a['ee_cos_median'], 'B_ee_cos': b['ee_cos_median'],
            'A_grip_err': a['gripper_err_median'], 'B_grip_err': b['gripper_err_median'],
        })

    print(f"{'phase':22s} {'expert':>8s} {'A|dq|':>8s} {'B|dq|':>8s} {'A_mae':>7s} {'B_mae':>7s} "
          f"{'A_cos':>6s} {'B_cos':>6s} {'A_ee':>6s} {'B_ee':>6s}")
    for r in rows:
        print(f"{r['phase']:22s} {r['expert_absmax']:8.4f} {r['A_model_absmax']:8.4f} "
              f"{r['B_model_absmax']:8.4f} {r['A_joint_mae']:7.4f} {r['B_joint_mae']:7.4f} "
              f"{(r['A_cos'] if r['A_cos'] is not None else 0):6.3f} "
              f"{(r['B_cos'] if r['B_cos'] is not None else 0):6.3f} "
              f"{(r['A_ee_cos'] if r['A_ee_cos'] is not None else 0):6.3f} "
              f"{(r['B_ee_cos'] if r['B_ee_cos'] is not None else 0):6.3f}")

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        'n_phases': len(rows),
        'mean_absmax': {'expert': mean('expert_absmax'),
                         'A': mean('A_model_absmax'), 'B': mean('B_model_absmax')},
        'mean_mag_ratio': {'A': mean('A_mag_ratio'), 'B': mean('B_mag_ratio')},
        'mean_joint_mae': {'A': mean('A_joint_mae'), 'B': mean('B_joint_mae')},
        'mean_cos': {'A': mean('A_cos'), 'B': mean('B_cos')},
        'mean_ee_cos': {'A': mean('A_ee_cos'), 'B': mean('B_ee_cos')},
        'rows': rows,
    }
    print('\nsummary:', json.dumps({k: v for k, v in summary.items() if k != 'rows'},
                                   ensure_ascii=False))
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
