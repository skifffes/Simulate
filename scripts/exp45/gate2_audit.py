"""实验 4.5 Gate 2/3：Action Expert-only 参数分区、梯度/更新审计、A40 可行性。

复用 exp42/train_lora.py 的数据与训练步实现，只替换：
  * 模型变体：paligemma='gemma_2b'、action_expert='gemma_300m'（**非 LoRA**）；
  * freeze_filter：冻结 PaliGemma（img + llm 非 _1 部分），只训练 action expert + 动作投影。

输出：
  /scratch/lijh/exp45/parameter_partition.json
  /scratch/lijh/exp45/gate2_audit.json
  /scratch/lijh/exp45/gate3_benchmark.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/scratch/lijh/exp40/repo')
sys.path.insert(0, str(REPO / 'scripts' / 'exp42'))
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

import flax.nnx as nnx  # noqa: E402
import flax.traverse_util as traverse_util  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.policies.droid_policy as droid_policy  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402
import train_lora as TL  # noqa: E402

OUT = Path('/scratch/lijh/exp45')
ASSETS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets')
PARAMS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params')
HORIZON = 16
DATA = Path('/scratch/lijh/exp43/export_v3')

# Action Expert（含 adaRMSNorm Dense）+ 动作投影 = 全部可训练参数
TRAINABLE_PATTERNS = [
    r'PaliGemma/llm/.*_1(/.*)?',      # action expert: layers/*_1, final_norm_1, ...
    r'action_in_proj(:?/.*)?',
    r'action_out_proj(:?/.*)?',
    r'time_mlp_in(:?/.*)?',
    r'time_mlp_out(:?/.*)?',
]


def build_config_ae(batch_size=4, num_steps=12000, lr_scale=1.0):
    import dataclasses
    base = _config.get_config('pi05_droid_finetune')
    model_cfg = pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=HORIZON,
        paligemma_variant='gemma_2b', action_expert_variant='gemma_300m')
    data_factory = _config.SimpleDataConfig(
        assets=_config.AssetsConfig(assets_dir=str(ASSETS), asset_id='droid'),
        data_transforms=lambda m: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=_model.ModelType.PI05)],
            outputs=[droid_policy.DroidOutputs()]),
        base_config=_config.DataConfig(prompt_from_task=True))
    trainable = nnx.All(*[nnx_utils.PathRegex(p) for p in TRAINABLE_PATTERNS]) \
        if len(TRAINABLE_PATTERNS) == 1 else nnx.Any(
            *[nnx_utils.PathRegex(p) for p in TRAINABLE_PATTERNS])
    freeze_filter = nnx.Not(trainable)
    cfg = dataclasses.replace(
        base, model=model_cfg, data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARAMS)),
        freeze_filter=freeze_filter, ema_decay=None,
        batch_size=int(batch_size), num_train_steps=int(num_steps))
    if lr_scale != 1.0:
        cfg = dataclasses.replace(cfg, lr_schedule=dataclasses.replace(
            cfg.lr_schedule, peak_lr=cfg.lr_schedule.peak_lr * lr_scale))
    return cfg


def partition_table(train_state):
    flat = traverse_util.flatten_dict(train_state.params.to_pure_dict())
    rows = []
    for path, val in flat.items():
        name = '/'.join(str(p) for p in path)
        n = int(np.prod(val.shape))
        trainable = not bool(train_state_should_freeze(name))
        rows.append({'path': name, 'shape': list(val.shape), 'dtype': str(val.dtype),
                     'n': n, 'trainable': trainable})
    return rows


_TRAINABLE_RE_CHECK = None


def train_state_should_freeze(name: str) -> bool:
    global _TRAINABLE_RE_CHECK
    import re
    if _TRAINABLE_RE_CHECK is None:
        _TRAINABLE_RE_CHECK = [re.compile(p) for p in TRAINABLE_PATTERNS]
    return not any(r.fullmatch(name) for r in _TRAINABLE_RE_CHECK)


def tensor_fingerprint(train_state) -> dict:
    flat = traverse_util.flatten_dict(train_state.params.to_pure_dict())
    out = {}
    for path, val in flat.items():
        a = np.asarray(jax.device_get(val), dtype=np.float32)
        out['/'.join(str(p) for p in path)] = [float(a.astype(np.float64).sum()),
                                               float(np.abs(a.astype(np.float64)).sum())]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--save-dir', default=str(OUT / 'gate_ckpt'))
    ap.add_argument('--max-hours', type=float, default=0.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = build_config_ae(batch_size=args.batch, num_steps=max(args.steps, 2))
    print('building model + loading weights ...', flush=True)
    train_state = TL.init_state(cfg)
    print(f'weights loaded in {time.time() - t0:.1f}s', flush=True)

    rows = partition_table(train_state)
    n_total = sum(r['n'] for r in rows)
    n_train = sum(r['n'] for r in rows if r['trainable'])
    print(f'params {len(rows)} tensors, total {n_total/1e6:.1f}M, '
          f'trainable {n_train/1e6:.1f}M ({100*n_train/n_total:.2f}%)')
    for r in rows:
        if r['trainable']:
            print(f"  TRAIN {r['path']:60s} {str(r['shape']):24s} {r['n']/1e6:8.2f}M")
    (OUT / 'parameter_partition.json').write_text(json.dumps({
        'n_tensors': len(rows), 'n_total': n_total, 'n_trainable': n_train,
        'trainable_patterns': TRAINABLE_PATTERNS, 'rows': rows,
    }, ensure_ascii=False, indent=1), encoding='utf-8')

    # opt_state 是否只覆盖 trainable
    opt_leaves = jax.tree_util.tree_leaves(train_state.opt_state)
    n_opt = sum(int(np.prod(x.shape)) for x in opt_leaves)
    print(f'opt_state leaves {len(opt_leaves)}, elements {n_opt/1e6:.1f}M '
          f'(expect ~2x trainable = {2*n_train/1e6:.1f}M)', flush=True)

    TL.DATA = DATA
    TL.OUT = OUT / 'gate_run'
    TL.OUT.mkdir(parents=True, exist_ok=True)
    meta = TL.load_meta()
    cache = TL.EpisodeCache()
    transform, data_cfg = TL.make_transforms(cfg)
    rng_np = np.random.default_rng(0)
    TL.WIN_IDX = TL.windows(meta, 'train')
    print(f'train windows {len(TL.WIN_IDX)} episodes {len(meta)}', flush=True)

    step_fn = TL.make_train_step(cfg, loss_dims=0)
    jax_rng = jax.random.key(0)
    batch = TL.build_batch(meta, cache, transform, rng_np, args.batch, label_mode='anchor')
    obs = _model.Observation.from_dict({k: v for k, v in batch.items()
                                        if not k.startswith('_')})
    print('first step (incl. JIT) ...', flush=True)
    t1 = time.time()
    train_state, info = step_fn(jax_rng, train_state, (obs, batch['actions']))
    jax.block_until_ready(train_state.params)
    print(f'first step {time.time()-t1:.1f}s loss={float(info["loss"]):.4f} '
          f'grad={float(info["grad_norm"]):.3f}', flush=True)

    before = tensor_fingerprint(train_state)
    losses = [float(info['loss'])]
    grads = [float(info['grad_norm'])]
    t2 = time.time()
    for k in range(2, args.steps + 1):
        batch = TL.build_batch(meta, cache, transform, rng_np, args.batch,
                               label_mode='anchor')
        obs_k = _model.Observation.from_dict({kk: v for kk, v in batch.items()
                                              if not kk.startswith('_')})
        train_state, info = step_fn(jax_rng, train_state, (obs_k, batch['actions']))
        if k % 5 == 0 or k == args.steps:
            jax.block_until_ready(train_state.params)
            losses.append(float(info['loss']))
            grads.append(float(info['grad_norm']))
            dt = time.time() - t2
            print(f'step {k} loss={float(info["loss"]):.4f} grad={float(info["grad_norm"]):.3f} '
                  f'{(k-1)/dt:.3f} step/s', flush=True)
    total_s = time.time() - t2
    after = tensor_fingerprint(train_state)

    frozen_changed = [p for p in before
                      if before[p] != after[p] and train_state_should_freeze(p)]
    expert_changed = [p for p in before
                      if before[p] != after[p] and not train_state_should_freeze(p)]
    expert_total = [p for p in before if not train_state_should_freeze(p)]
    audit = {
        'steps': args.steps, 'batch': args.batch,
        'n_total': n_total, 'n_trainable': n_train,
        'trainable_ratio': n_train / n_total,
        'opt_state_leaves': len(opt_leaves), 'opt_state_elements': n_opt,
        'losses': losses, 'grad_norms': grads,
        'seconds_per_step': total_s / max(args.steps - 1, 1),
        'frozen_tensors': len(before) - len(expert_total),
        'frozen_changed_count': len(frozen_changed),
        'frozen_changed_examples': frozen_changed[:5],
        'expert_tensors': len(expert_total),
        'expert_changed_count': len(expert_changed),
        'expert_unchanged': [p for p in expert_total if p not in expert_changed],
        'gate2_pass': len(frozen_changed) == 0 and len(expert_changed) == len(expert_total),
    }
    (OUT / 'gate2_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'rows'}, ensure_ascii=False,
                     indent=2)[:1500])

    # Gate3：保存 + 恢复（trainable 参数 + opt state + step + rng）
    ck = Path(args.save_dir)
    ck.mkdir(parents=True, exist_ok=True)
    TL.save_ckpt(train_state, cfg, ck, args.steps)
    TL.save_opt_state(train_state, ck / f'opt_step{args.steps:05d}.npz',
                      rng_jax=jax_rng, rng_np=rng_np)
    bench = {'seconds_per_step': audit['seconds_per_step'],
             'steps_per_second': 1.0 / audit['seconds_per_step'] if audit['seconds_per_step'] else None,
             'losses': losses, 'grad_norms': grads,
             'ckpt_files': [p.name for p in sorted(ck.glob('*'))],
             'compile_seconds': None}
    (OUT / 'gate3_benchmark.json').write_text(json.dumps(bench, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print('ckpt files:', bench['ckpt_files'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
