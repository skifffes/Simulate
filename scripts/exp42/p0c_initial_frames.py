"""P0c：**初始状态**（episode 第 0–2 帧）上的动作对比。

动机：闭环从 HOME 起步，而 P0/P0b 的离线抽样取的是"专家动作最大"的中段帧。
如果模型在初始 HOME 状态就给出错误动作，闭环第一步就会把机械臂带偏，
这比"中段方向不准"更能解释 rollout 的失败。

用法（openpi venv）：
  python scripts/exp42/p0c_initial_frames.py --out /scratch/lijh/exp42/diag
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
FINGER_MAX_M = 0.04
HORIZON = 16


def frame(meta_row, t):
    sa = np.load(DATA / meta_row['state_path'])
    top = np.load(DATA / meta_row['top_path'], mmap_mode='r')
    wrist = np.load(DATA / meta_row['wrist_path'], mmap_mode='r')
    st = sa['state'][t].astype(np.float64)
    act = sa['action'][t:t + HORIZON].astype(np.float64)
    dq = act[:, :7] - st[:7][None, :]
    g = np.clip(act[:, 7] / FINGER_MAX_M, 0, 1)
    obs = {
        'observation/exterior_image_1_left': np.asarray(top[t], dtype=np.uint8),
        'observation/wrist_image_left': np.asarray(wrist[t], dtype=np.uint8),
        'observation/joint_position': st[:7],
        'observation/gripper_position': np.asarray([np.clip(st[7:9].mean() / FINGER_MAX_M, 0, 1)]),
        'prompt': meta_row['prompt'],
    }
    return obs, dq, g, st


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float('nan')
    return float(a @ b / (na * nb))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/scratch/lijh/exp42/diag')
    ap.add_argument('--ckpt501', default='/scratch/lijh/exp42/run/ckpts/params_nested')
    ap.add_argument('--ckpt901', default='/scratch/lijh/exp42/run2/ckpts/params_nested')
    ap.add_argument('--episodes-per-skill', type=int, default=8)
    ap.add_argument('--frames', default='0,1,2')
    args = ap.parse_args(argv)
    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config

    from exp42 import registry

    ts = [int(x) for x in args.frames.split(',')]
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    picks = []
    for skill in ('A90', 'B90'):
        idxs = [i for i, m in enumerate(meta) if m['skill'] == skill][:args.episodes_per_skill]
        for i in idxs:
            for t in ts:
                if t < meta[i]['n_frames'] - HORIZON:
                    picks.append((i, t))
    print(f'initial-state frames: {len(picks)}', flush=True)

    registry.register()
    cfg_ft = _config.get_config(registry.CONFIG_NAME)
    pol = {}
    print('loading baseline ...', flush=True)
    pol['baseline'] = _policy_config.create_trained_policy(
        _config.get_config('pi05_droid'),
        '/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid')
    print('loading step501 ...', flush=True)
    pol['step501'] = _policy_config.create_trained_policy(cfg_ft, args.ckpt501)
    print('loading step901 ...', flush=True)
    pol['step901'] = _policy_config.create_trained_policy(cfg_ft, args.ckpt901)

    rows = []
    for i, t in picks:
        row = meta[i]
        obs, dq, g, st = frame(row, t)
        item = {'skill': row['skill'], 'episode': row['episode_index'], 'frame': t,
                'split': row['split'],
                'expert_absmax_first': float(np.abs(dq[0]).max()),
                'expert_norm_first': float(np.linalg.norm(dq[0])),
                'gripper_state': float(obs['observation/gripper_position'][0]),
                'models': {}}
        for name, p in pol.items():
            a = np.asarray(p.infer(dict(obs))['actions'], dtype=np.float64)
            n = min(a.shape[0], dq.shape[0])
            item['models'][name] = {
                'joint_absmax': float(np.abs(a[:n, :7]).max()),
                'magnitude_ratio': float(np.abs(a[:n, :7]).mean()
                                         / max(np.abs(dq[:n, :7]).mean(), 1e-9)),
                'cos_step0': cos(a[0, :7], dq[0, :7]),
                'cos_mean6': float(np.nanmean([cos(a[k, :7], dq[k, :7])
                                               for k in range(min(6, n))])),
                'joint_mae': float(np.abs(a[:n, :7] - dq[:n, :7]).mean()),
                'gripper_first': float(a[0, 7]),
            }
        rows.append(item)
        print(f"[{item['skill']} ep{row['episode_index']} f{t}] expert|dq|={item['expert_absmax_first']:.3f} "
              + ' | '.join(f"{nm}: |a|={mm['joint_absmax']:.2f} cos={mm['cos_step0'] if mm['cos_step0'] is not None else float('nan'):.2f}"
                           for nm, mm in item['models'].items()), flush=True)

    summ = {}
    for skill in ('A90', 'B90'):
        for name in ('baseline', 'step501', 'step901'):
            sel = [r['models'][name] for r in rows if r['skill'] == skill]
            if not sel:
                continue
            summ[f'{skill}/{name}'] = {
                'n': len(sel),
                'mean_joint_absmax': float(np.mean([m['joint_absmax'] for m in sel])),
                'expert_absmax_mean': float(np.mean([r['expert_absmax_first']
                                                     for r in rows if r['skill'] == skill])),
                'mean_magnitude_ratio': float(np.mean([m['magnitude_ratio'] for m in sel])),
                'mean_cos_step0': float(np.nanmean([m['cos_step0'] for m in sel])),
                'mean_cos_mean6': float(np.nanmean([m['cos_mean6'] for m in sel])),
                'mean_joint_mae': float(np.mean([m['joint_mae'] for m in sel])),
            }
    res = {'frames': rows, 'summary': summ,
           'note': 'episode 起始帧（HOME 附近），闭环第一步的状态；P0/P0b 的离线抽样是中段运动帧'}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'p0c_initial_state.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
