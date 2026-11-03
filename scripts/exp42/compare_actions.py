"""Gate 4 证据：同一观测下，原始 `pi05_droid` 与微调 checkpoint 的 action chunk 对比。

在**同一个进程**里用官方 `create_trained_policy` 分别构建：
  1) baseline：`pi05_droid`（全参数变体，官方 checkpoint）
  2) finetuned：`exp42_pi05_droid_lora`（LoRA 变体，训练导出的完整 params）
然后喂**完全相同的观测**（4.1 数据里的一帧 A90 与一帧 B90），比较输出。

判据：
  * 两者都能推理（shape (16,8)）；
  * 逐元素不同（不是同一个权重）；
  * 差异量级合理（不是随机数）；并记录动作幅值，检查是否被 clip。

用法（openpi venv）：
  python scripts/exp42/compare_actions.py --ckpt /scratch/lijh/exp42/run/ckpts/params_step00501
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path(os.environ.get('EXP42_DATA', '/scratch/lijh/exp42/data'))
OUT = Path(os.environ.get('EXP42_OUT', '/scratch/lijh/exp42/run'))
FINGER_MAX_M = 0.04


def load_obs(meta: list[dict], pick: dict) -> dict:
    row = next(m for m in meta if m['skill'] == pick['skill'] and m['episode_index'] == pick['episode'])
    sa = np.load(DATA / row['state_path'])
    top = np.load(DATA / row['top_path'], mmap_mode='r')
    wrist = np.load(DATA / row['wrist_path'], mmap_mode='r')
    t = pick['frame']
    st = sa['state'][t]
    return {
        'observation/exterior_image_1_left': np.asarray(top[t], dtype=np.uint8),
        'observation/wrist_image_left': np.asarray(wrist[t], dtype=np.uint8),
        'observation/joint_position': st[:7].astype(np.float64),
        'observation/gripper_position': np.asarray(
            [np.clip(st[7:9].mean() / FINGER_MAX_M, 0, 1)], dtype=np.float64),
        'prompt': row['prompt'],
    }, row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--picks',
                    default='[{"skill":"A90","episode":0,"frame":40},{"skill":"B90","episode":0,"frame":20}]')
    args = ap.parse_args(argv)
    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config

    from exp42 import registry

    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    picks = json.loads(args.picks)

    registry.register()
    base_cfg = _config.get_config('pi05_droid')
    ft_cfg = _config.get_config(registry.CONFIG_NAME)

    print('building baseline policy (pi05_droid) ...', flush=True)
    base_policy = _policy_config.create_trained_policy(
        base_cfg, '/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid')
    print('building finetuned policy ...', flush=True)
    ft_policy = _policy_config.create_trained_policy(ft_cfg, args.ckpt)

    res = {'ckpt': args.ckpt, 'picks': picks, 'items': []}
    for pick in picks:
        obs, row = load_obs(meta, pick)
        a_base = np.asarray(base_policy.infer(obs)['actions'], dtype=np.float64)
        a_ft = np.asarray(ft_policy.infer(obs)['actions'], dtype=np.float64)
        # baseline 用官方 pi05_droid 配置（horizon 15），微调用 horizon 16；比较公共前缀
        n = min(a_base.shape[0], a_ft.shape[0])
        diff = a_ft[:n] - a_base[:n]
        item = {
            'skill': row['skill'], 'episode': row['episode_index'], 'frame': pick['frame'],
            'prompt': row['prompt'],
            'shape_base': list(a_base.shape), 'shape_ft': list(a_ft.shape),
            'base_action_absmax': float(np.abs(a_base).max()),
            'ft_action_absmax': float(np.abs(a_ft).max()),
            'base_gripper': a_base[0, 7].round(4).tolist(),
            'ft_gripper': a_ft[0, 7].round(4).tolist(),
            'diff_absmax': float(np.abs(diff).max()),
            'diff_mean_abs': float(np.abs(diff).mean()),
            'identical': bool(np.allclose(a_base[:n], a_ft[:n], atol=1e-9)),
            'base_first_row': a_base[0].round(4).tolist(),
            'ft_first_row': a_ft[0].round(4).tolist(),
        }
        res['items'].append(item)
        print(json.dumps({k: item[k] for k in ('skill', 'shape_ft', 'base_action_absmax',
                                               'ft_action_absmax', 'diff_absmax', 'identical')},
                         ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'action_compare_baseline_vs_finetuned.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print('wrote', OUT / 'action_compare_baseline_vs_finetuned.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
