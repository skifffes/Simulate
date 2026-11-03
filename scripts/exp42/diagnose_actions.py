"""实验 4.2 P0 诊断：微调后的 π₀.₅ 在**专家观测**上能否预测接近专家的动作？

回答三个问题（review P0）：
  1. 同一批 train/val 观测上，专家动作 vs 原始模型 vs step501 vs step901 的**真实采样动作**差多少
     （幅度比、方向余弦、逐关节 MAE、夹爪命令）；
  2. Flow-Matching loss 里 **7D 关节 / 1D 夹爪 / 24D padding** 各占多少
     （用"把真实动作维度置零后再算 loss"的对照实验归因，不需要改模型内部）；
  3. 反归一化后每个关节有多少步被 action adapter 的 0.25 rad 上限裁剪。

用法（openpi venv，单卡）：
  python scripts/exp42/diagnose_actions.py --out /scratch/lijh/exp42/diag
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
CLIP_RAD = 0.25


def load_frame(meta: list[dict], row_idx: int, t: int):
    row = meta[row_idx]
    sa = np.load(DATA / row['state_path'])
    top = np.load(DATA / row['top_path'], mmap_mode='r')
    wrist = np.load(DATA / row['wrist_path'], mmap_mode='r')
    st = sa['state'][t].astype(np.float64)
    act = sa['action'][t:t + HORIZON].astype(np.float64)
    dq = act[:, :7] - st[:7][None, :]
    g = np.clip(act[:, 7] / FINGER_MAX_M, 0, 1)
    obs = {
        'observation/exterior_image_1_left': np.asarray(top[t], dtype=np.uint8),
        'observation/wrist_image_left': np.asarray(wrist[t], dtype=np.uint8),
        'observation/joint_position': st[:7],
        'observation/gripper_position': np.asarray([np.clip(st[7:9].mean() / FINGER_MAX_M, 0, 1)]),
        'prompt': row['prompt'],
    }
    return obs, st, dq, g, row


def compare(a_model: np.ndarray, dq: np.ndarray, g: np.ndarray) -> dict:
    n = min(a_model.shape[0], dq.shape[0])
    m = a_model[:n]
    e = np.concatenate([dq[:n], g[:n, None]], axis=1)
    jm = m[:, :7]
    je = e[:, :7]
    out = {
        'shape': list(a_model.shape),
        'joint_mae_rad': float(np.abs(jm - je).mean()),
        'joint_absmax_model': float(np.abs(jm).max()),
        'joint_absmax_expert': float(np.abs(je).max()),
        'magnitude_ratio': float(np.abs(jm).mean() / max(np.abs(je).mean(), 1e-9)),
        'per_joint_mae': np.abs(jm - je).mean(axis=0).round(4).tolist(),
        'per_joint_absmax_model': np.abs(jm).max(axis=0).round(4).tolist(),
        'gripper_model_first': float(m[0, 7]),
        'gripper_expert_first': float(e[0, 7]),
        'gripper_mae': float(np.abs(m[:n, 7] - e[:n, 7]).mean()),
        'clip_frac_per_joint_at_0.25': (np.abs(jm) > CLIP_RAD).mean(axis=0).round(3).tolist(),
        'clip_frac_overall': float((np.abs(jm) > CLIP_RAD).mean()),
    }
    # 方向：逐时间步的 7 维余弦相似度（绕开幅度）
    denom = (np.linalg.norm(jm, axis=1) * np.linalg.norm(je, axis=1))
    cos = np.where(denom > 1e-9, (jm * je).sum(axis=1) / np.maximum(denom, 1e-9), np.nan)
    out['direction_cosine_mean'] = float(np.nanmean(cos)) if np.isfinite(cos).any() else None
    out['direction_cosine_first'] = float(cos[0]) if np.isfinite(cos[0]) else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/scratch/lijh/exp42/diag')
    ap.add_argument('--ckpt501', default='/scratch/lijh/exp42/run/ckpts/params_nested')
    ap.add_argument('--ckpt901', default='/scratch/lijh/exp42/run2/ckpts/params_nested')
    ap.add_argument('--train-frames', type=int, default=6)
    ap.add_argument('--val-frames', type=int, default=4)
    args = ap.parse_args(argv)
    import jax
    import jax.numpy as jnp
    import openpi.models.model as _model
    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config

    from exp42 import registry

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    tr = [i for i, m in enumerate(meta) if m['split'] == 'train']
    va = [i for i, m in enumerate(meta) if m['split'] == 'val']
    picks = []
    for i in tr[:args.train_frames]:
        picks.append((i, int(meta[i]['n_frames'] // 3), 'train'))
    for i in va[:args.val_frames]:
        picks.append((i, int(meta[i]['n_frames'] // 3), 'val'))
    print(f'frames: {len(picks)}', flush=True)

    registry.register()
    policies = {}
    print('loading baseline pi05_droid ...', flush=True)
    policies['baseline'] = _policy_config.create_trained_policy(
        _config.get_config('pi05_droid'),
        '/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid')
    ft_cfg = _config.get_config(registry.CONFIG_NAME)
    print('loading step501 ...', flush=True)
    policies['step501'] = _policy_config.create_trained_policy(ft_cfg, args.ckpt501)
    print('loading step901 ...', flush=True)
    policies['step901'] = _policy_config.create_trained_policy(ft_cfg, args.ckpt901)

    rows = []
    for row_idx, t, split in picks:
        obs, st, dq, g, row = load_frame(meta, row_idx, t)
        item = {'skill': row['skill'], 'episode_index': row['episode_index'],
                'frame': t, 'split': split, 'prompt': row['prompt'], 'models': {}}
        item['expert'] = {'joint_mean_abs': float(np.abs(dq).mean()),
                          'joint_absmax': float(np.abs(dq).max()),
                          'gripper_first': float(g[0])}
        for name, pol in policies.items():
            a = np.asarray(pol.infer(dict(obs))['actions'], dtype=np.float64)
            item['models'][name] = compare(a, dq, g)
        rows.append(item)
        print(f"[{row['skill']} ep{row['episode_index']} {split} f{t}] "
              + ' | '.join(f"{n}: ratio={m['magnitude_ratio']:.2f} mae={m['joint_mae_rad']:.3f} "
                           f"cos={m['direction_cosine_first']}" for n, m in item['models'].items()),
              flush=True)

    # ---- loss 归因：真实目标 vs 全零目标 vs 只保留 padding ----
    import openpi.transforms as _transforms

    data_cfg = ft_cfg.data.create(assets_dirs=Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets'),
                                  model_config=ft_cfg.model)
    chain = list(data_cfg.data_transforms.inputs) + [
        _transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm)
    ] + list(data_cfg.model_transforms.inputs)
    transform = _transforms.compose(chain)

    def batch_of(picks_subset):
        items = []
        for row_idx, t, _ in picks_subset:
            obs, st, dq, g, row = load_frame(meta, row_idx, t)
            raw = dict(obs)
            raw['actions'] = np.concatenate([dq, g[:, None]], axis=1)
            items.append(transform(raw))
        return jax.tree.map(lambda *xs: np.stack(xs, 0), *items)

    loss_report = {}
    subset = picks[:4]
    bt = batch_of(subset)
    obs_b = _model.Observation.from_dict({k: v for k, v in bt.items() if not k.startswith('_')})
    obs_b = jax.tree.map(lambda x: jnp.asarray(x), obs_b)
    act = jnp.asarray(bt['actions'])
    act_zero = act.at[..., :8].set(0.0)          # 真实维度置零，padding 本来就是 0
    act_pad_only = act.at[..., :8].set(0.0)      # 与上一条等价（padding 目标原本为 0）
    for name in ('step501', 'step901'):
        pol = policies[name]
        model = pol._model
        model.eval()
        l_real = float(jax.device_get(jnp.mean(model.compute_loss(
            jax.random.key(0), obs_b, act, train=False))))
        l_zero = float(jax.device_get(jnp.mean(model.compute_loss(
            jax.random.key(0), obs_b, act_zero, train=False))))
        loss_report[name] = {
            'loss_with_real_targets': l_real,
            'loss_with_zero_real_dims': l_zero,
            'implied_real_dim_contribution': l_real - l_zero,
            'padding_fraction_of_action_dim': 24 / 32,
            'note': ('若 zero-real 的 loss 与真实 loss 接近，说明 loss 主要由 padding/零目标主导；'
                     '差值即真实 8 维动作对 loss 的贡献。'),
        }
        print(f'{name} loss real={l_real:.5f} zero-real={l_zero:.5f}', flush=True)

    res = {'date': None, 'clip_rad': CLIP_RAD, 'picks': len(picks),
           'frames': rows, 'loss_attribution': loss_report,
           'summary': {}}
    for name in ('baseline', 'step501', 'step901'):
        ms = [r['models'][name] for r in rows]
        res['summary'][name] = {
            'mean_magnitude_ratio': float(np.mean([m['magnitude_ratio'] for m in ms])),
            'mean_joint_mae_rad': float(np.mean([m['joint_mae_rad'] for m in ms])),
            'mean_direction_cosine': float(np.nanmean([m['direction_cosine_first'] for m in ms])),
            'mean_clip_frac': float(np.mean([m['clip_frac_overall'] for m in ms])),
            'mean_gripper_mae': float(np.mean([m['gripper_mae'] for m in ms])),
        }
    (out / 'action_diagnostics.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(res['summary'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
