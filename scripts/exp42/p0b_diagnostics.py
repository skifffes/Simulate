"""实验 4.2-P0b：扩大离线方向诊断 + 逐维 loss/梯度归因（**不训练**）。

回答 review 的三点质疑：
  1. 方向余弦 ≈ 0 是否只是因为"专家动作太小 / 帧太少 / A、B 混在一起"？
     → 只在**专家确实在动**的帧上算余弦，按技能分开，并逐 chunk 步报告；
  2. padding 到底吃掉多少 loss/梯度？
     → 用**同一组噪声与时间采样**复算 v_t，把平方误差拆成 7D 关节 / 1D 夹爪 / 24D padding，
       再分别对 LoRA 参数求梯度范数；
  3. action chunk 是否存在一步错位？
     → 比较 expert Δq(t) 与 Δq(t+1) 对模型动作的相关性（错位时后者更高）。

用法（openpi venv，单卡）：
  python scripts/exp42/p0b_diagnostics.py --out /scratch/lijh/exp42/diag
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
MOVE_THRESHOLD = 0.02      # rad/step：专家"确实在动"的阈值


def frame_arrays(meta_row, t):
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
    return obs, st, dq, g


def select_frames(meta, skill: str, n_episodes: int, per_episode: int) -> list[tuple[int, int]]:
    """每个技能挑若干 episode，在"专家动作范数最大"的若干帧上取样（只取确实在动的帧）。"""
    idxs = [i for i, m in enumerate(meta) if m['skill'] == skill]
    picks = []
    step = max(1, len(idxs) // n_episodes)
    for i in idxs[::step][:n_episodes]:
        row = meta[i]
        sa = np.load(DATA / row['state_path'])
        S = sa['state']
        A = sa['action']
        T = S.shape[0]
        norms = []
        for t in range(0, max(1, T - HORIZON), max(1, (T - HORIZON) // 40)):
            dq = A[t, :7] - S[t, :7]
            norms.append((float(np.abs(dq).max()), t))
        norms.sort(reverse=True)
        chosen = [t for _, t in norms[:per_episode]]
        picks += [(i, t) for t in chosen]
    return picks


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float('nan')
    return float(a @ b / (na * nb))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/scratch/lijh/exp42/diag')
    ap.add_argument('--ckpt501', default='/scratch/lijh/exp42/run/ckpts/params_nested')
    ap.add_argument('--ckpt901', default='/scratch/lijh/exp42/run2/ckpts/params_nested')
    ap.add_argument('--episodes-per-skill', type=int, default=6)
    ap.add_argument('--frames-per-episode', type=int, default=2)
    args = ap.parse_args(argv)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import openpi.models.model as _model
    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config
    import openpi.transforms as _transforms
    from openpi.models.pi0 import make_attn_mask

    from exp42 import registry

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]

    picks = (select_frames(meta, 'A90', args.episodes_per_skill, args.frames_per_episode)
             + select_frames(meta, 'B90', args.episodes_per_skill, args.frames_per_episode))
    print(f'frames: {len(picks)} '
          f'({sum(1 for i, _ in picks if meta[i]["skill"] == "A90")} A90 / '
          f'{sum(1 for i, _ in picks if meta[i]["skill"] == "B90")} B90)', flush=True)

    registry.register()
    cfg_ft = _config.get_config(registry.CONFIG_NAME)
    policies = {}
    print('loading baseline ...', flush=True)
    policies['baseline'] = _policy_config.create_trained_policy(
        _config.get_config('pi05_droid'),
        '/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid')
    print('loading step501 ...', flush=True)
    policies['step501'] = _policy_config.create_trained_policy(cfg_ft, args.ckpt501)
    print('loading step901 ...', flush=True)
    policies['step901'] = _policy_config.create_trained_policy(cfg_ft, args.ckpt901)

    rows = []
    for row_idx, t in picks:
        row = meta[row_idx]
        obs, st, dq, g = frame_arrays(row, t)
        # 对齐检查：expert Δq(t) 与 Δq(t+1)
        obs_n, st_n, dq_n, g_n = frame_arrays(row, min(t + 1, row['n_frames'] - HORIZON - 1))
        item = {'skill': row['skill'], 'episode': row['episode_index'], 'frame': int(t),
                'split': row['split'],
                'expert_norm_first': float(np.linalg.norm(dq[0])),
                'expert_absmax_first': float(np.abs(dq[0]).max()),
                'moving': bool(np.abs(dq[0]).max() >= MOVE_THRESHOLD),
                'models': {}}
        for name, pol in policies.items():
            a = np.asarray(pol.infer(dict(obs))['actions'], dtype=np.float64)
            a_next = np.asarray(pol.infer(dict(obs_n))['actions'], dtype=np.float64)
            n = min(a.shape[0], dq.shape[0])
            jm, je = a[:n, :7], dq[:n, :7]
            per_step_cos = [cos(jm[k], je[k]) for k in range(min(6, n))]
            item['models'][name] = {
                'joint_mae': float(np.abs(jm - je).mean()),
                'joint_absmax': float(np.abs(jm).max()),
                'expert_absmax': float(np.abs(je).max()),
                'magnitude_ratio': float(np.abs(jm).mean() / max(np.abs(je).mean(), 1e-9)),
                'cos_step0': cos(jm[0], je[0]),
                'cos_mean6': float(np.nanmean(per_step_cos)),
                'cos_steps': [round(float(c), 3) if np.isfinite(c) else None
                              for c in per_step_cos],
                'gripper_first': float(a[0, 7]),
                'gripper_expert': float(g[0]),
                'cos_vs_next_frame_expert': cos(a_next[0, :7], dq_n[0]),
            }
        rows.append(item)
        print(f"[{item['skill']} ep{row['episode_index']} {item['split']} f{t} "
              f"|e|={item['expert_absmax_first']:.3f}] "
              + ' | '.join(f"{nm}: ratio={mm['magnitude_ratio']:.2f} cos0={mm['cos_step0']} "
                           f"cos6={mm['cos_mean6']:.2f}" for nm, mm in item['models'].items()),
              flush=True)

    # ---------------- 逐维 loss 与梯度 ----------------
    assets = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets')
    data_cfg = cfg_ft.data.create(assets_dirs=assets, model_config=cfg_ft.model)
    chain = list(data_cfg.data_transforms.inputs) + [
        _transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm)
    ] + list(data_cfg.model_transforms.inputs)
    transform = _transforms.compose(chain)

    def batch_of(subset):
        items = []
        for row_idx, t in subset:
            obs, st, dq, g = frame_arrays(meta[row_idx], t)
            raw = dict(obs)
            raw['actions'] = np.concatenate([dq, g[:, None]], axis=1)
            items.append(transform(raw))
        return jax.tree.map(lambda *xs: np.stack(xs, 0), *items)

    bt = batch_of(picks[:8])
    obs_b = _model.Observation.from_dict({k: v for k, v in bt.items() if not k.startswith('_')})
    obs_b = jax.tree.map(lambda x: jnp.asarray(x), obs_b)
    act_b = jnp.asarray(bt['actions'])

    def dim_loss(model, rng, obs, actions):
        pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        obs2 = _model.preprocess_observation(pre_rng, obs, train=False)
        bs = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        tm = jax.random.beta(time_rng, 1.5, 1, bs) * 0.999 + 0.001
        te = tm[..., None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions
        pre_tok, pre_mask, pre_ar = model.embed_prefix(obs2)
        suf_tok, suf_mask, suf_ar, adarms = model.embed_suffix(obs2, x_t, tm)
        input_mask = jnp.concatenate([pre_mask, suf_mask], axis=1)
        ar_mask = jnp.concatenate([pre_ar, suf_ar], axis=0)
        attn = make_attn_mask(input_mask, ar_mask)
        pos = jnp.cumsum(input_mask, axis=1) - 1
        (_, suf_out), _ = model.PaliGemma.llm([pre_tok, suf_tok], mask=attn, positions=pos,
                                              adarms_cond=[None, adarms])
        v_t = model.action_out_proj(suf_out[:, -model.action_horizon:])
        sq = jnp.square(v_t - u_t)
        return {'total': jnp.mean(sq), 'joint7': jnp.mean(sq[..., :7]),
                'gripper1': jnp.mean(sq[..., 7:8]), 'padding24': jnp.mean(sq[..., 8:])}

    model = policies['step901']._model
    model.eval()
    rng = jax.random.key(0)
    losses = {k: float(jax.device_get(v)) for k, v in dim_loss(model, rng, obs_b, act_b).items()}

    # 梯度范数：分别对 trainable 参数求导
    filt = cfg_ft.trainable_filter

    def grad_norm_of(key: str) -> float:
        def fn(m):
            return dim_loss(m, rng, obs_b, act_b)[key]

        _, grads = nnx.value_and_grad(fn, argnums=nnx.DiffState(0, filt))(model)
        return float(jax.device_get(optax.global_norm(grads)))

    grads = {k: grad_norm_of(k) for k in ('total', 'joint7', 'gripper1', 'padding24')}

    # 归一化尺度对比：DROID vs train-split（仅动作前 8 维）
    droid = json.loads((assets / 'droid' / 'norm_stats.json').read_text(encoding='utf-8'))
    dq01 = np.asarray(droid['norm_stats']['actions']['q01'], dtype=float)[:8]
    dq99 = np.asarray(droid['norm_stats']['actions']['q99'], dtype=float)[:8]
    tr = [m for m in meta if m['split'] == 'train']
    S = np.concatenate([np.load(DATA / m['state_path'])['state'] for m in tr])
    A = np.concatenate([np.load(DATA / m['state_path'])['action'] for m in tr])
    dq = A[:, :7] - S[:, :7]
    act8 = np.concatenate([dq, np.clip(A[:, 7:8] / FINGER_MAX_M, 0, 1)], axis=1)
    norm_scale = {
        'droid_span_action8': np.round(dq99 - dq01, 4).tolist(),
        'train_span_action8': np.round(np.quantile(act8, 0.99, axis=0)
                                       - np.quantile(act8, 0.01, axis=0), 4).tolist(),
        'ratio_train_over_droid': np.round((np.quantile(act8, 0.99, axis=0)
                                            - np.quantile(act8, 0.01, axis=0))
                                           / np.maximum(dq99 - dq01, 1e-9), 4).tolist(),
    }

    res = {'frames': rows, 'loss_split_step901': losses, 'grad_norm_step901': grads,
           'norm_scale': norm_scale,
           'config': {'move_threshold': MOVE_THRESHOLD, 'horizon': HORIZON,
                      'n_frames': len(picks), 'same_noise_and_time': True}}

    # 汇总：只统计"专家在动"的帧，按技能分组
    summ = {}
    for skill in ('A90', 'B90'):
        for name in ('baseline', 'step501', 'step901'):
            sel = [r['models'][name] for r in rows if r['skill'] == skill and r['moving']]
            if not sel:
                continue
            summ[f'{skill}/{name}'] = {
                'n_moving_frames': len(sel),
                'mean_magnitude_ratio': float(np.mean([m['magnitude_ratio'] for m in sel])),
                'mean_cos_step0': float(np.nanmean([m['cos_step0'] for m in sel])),
                'mean_cos_mean6': float(np.nanmean([m['cos_mean6'] for m in sel])),
                'mean_joint_mae': float(np.mean([m['joint_mae'] for m in sel])),
                'frac_cos_positive_step0': float(np.mean([m['cos_step0'] > 0 for m in sel])),
            }
    res['summary_moving_frames'] = summ
    (out / 'p0b_diagnostics.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps({'loss_split': losses, 'grad_norm': grads,
                      'norm_scale': norm_scale}, ensure_ascii=False, indent=2))
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
