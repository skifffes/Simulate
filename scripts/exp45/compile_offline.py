"""实验 4.5：汇总 AE 与 LoRA 的离线评测 → offline_eval_results.json（含分阶段表）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

L45 = Path('/scratch/lijh/exp40/repo/results/experiment_4_5')
L44 = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')


def load(dirp, prefix):
    out = []
    for p in sorted(dirp.glob(f'{prefix}_step*_val_stats.json')):
        m = re.search(r'step(\d+)_val_stats', p.name)
        if not m:
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        closed = d['strata'].get('A90:gripper_closed') or {}
        hi = d['strata'].get('B90:motion_high') or {}
        out.append({
            'step': int(m.group(1)),
            'overall_mae': d['overall']['joint_mae_mean'],
            'overall_cos': (d['by_skill'].get('A90') or {}).get('cos_median'),
            'A90_mae': d['by_skill']['A90']['joint_mae_mean'],
            'A90_cos': (d['by_skill'].get('A90') or {}).get('cos_median'),
            'A90_grasp_mae': closed.get('joint_mae_mean'),
            'A90_grasp_grip': closed.get('grip_err_mean'),
            'A90_dz': d['by_skill']['A90']['dz_model_mean_m'],
            'B90_mae': d['by_skill']['B90']['joint_mae_mean'],
            'B90_cos': (d['by_skill'].get('B90') or {}).get('cos_median'),
            'B90_push_cos': hi.get('cos_median'),
            'B90_dz': d['by_skill']['B90']['dz_model_mean_m'],
            'seed_std': d['overall']['seed_mae_std_mean'],
        })
    return out


def main() -> int:
    ae = load(L45 / 'offline', 'ae')
    lora = load(L44 / 'offline', 'B')
    out = {'action_expert': ae, 'lora_group_b': lora,
           'note': 'val 30 状态 × 4 seeds；A90_grasp_* 是 gripper_closed 分层（专家运动≈0）'}
    (L45 / 'offline_eval_results.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'AE checkpoints {len(ae)}, LoRA checkpoints {len(lora)}')
    for r in ae:
        print('AE', r['step'], 'mae', round(r['overall_mae'], 4), 'A90', round(r['A90_grasp_mae'] or 0, 4),
              'B90cos', round(r['B90_push_cos'] or 0, 3), 'dz', round(r['B90_dz'], 4))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
