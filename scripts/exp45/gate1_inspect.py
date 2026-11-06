"""Gate 1：枚举 pi05_droid（**非 LoRA** 变体）的参数树并核对权重加载。

输出：
  * /scratch/lijh/exp45/params_tree.json：每个参数路径 + shape + dtype + 元素数
  * 控制台打印模块分组统计（PaliGemma/img/llm/action 投影等）与加载缺失统计
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import flax.traverse_util as traverse_util
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.weight_loaders as weight_loaders
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.droid_policy as droid_policy
import openpi.training.config as _config
import openpi.transforms as _transforms

OUT = Path('/scratch/lijh/exp45')
ASSETS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets')
PARAMS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params')
HORIZON = 16


def build_config(batch_size=4, num_steps=12000):
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
    import dataclasses
    cfg = dataclasses.replace(
        base, model=model_cfg, data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARAMS)),
        freeze_filter=nnx.Nothing, ema_decay=None,
        batch_size=batch_size, num_train_steps=num_steps)
    return cfg


def main() -> int:
    t0 = time.time()
    cfg = build_config()
    rng = jax.random.key(0)
    model = cfg.model.create(rng)
    graphdef, state = nnx.split(model)
    target = state.to_pure_dict()
    partial = cfg.weight_loader.load(target)
    state.replace_by_pure_dict(partial)
    params = nnx.state(nnx.merge(graphdef, state))
    flat = traverse_util.flatten_dict(params.to_pure_dict())

    rows = []
    for path, val in flat.items():
        name = '/'.join(str(p) for p in path)
        arr = np.asarray(jax.device_get(val)) if hasattr(val, 'shape') else None
        rows.append({'path': name,
                     'shape': list(val.shape) if hasattr(val, 'shape') else None,
                     'dtype': str(getattr(val, 'dtype', '')),
                     'n': int(np.prod(val.shape)) if hasattr(val, 'shape') else 0})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'params_tree.json').write_text(json.dumps({'n_params': len(rows), 'rows': rows},
                                                    ensure_ascii=False, indent=1), encoding='utf-8')

    total = sum(r['n'] for r in rows)
    groups = {}
    for r in rows:
        top = r['path'].split('/')[0]
        key = top
        if top == 'PaliGemma':
            seg = r['path'].split('/')
            key = 'PaliGemma/' + (seg[1] if len(seg) > 1 else '')
        groups[key] = groups.get(key, 0) + r['n']
    print(f'params {len(rows)} tensors, {total/1e6:.1f}M elements '
          f'({total*4/1e9:.1f} GB fp32), load {time.time()-t0:.1f}s')
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f'  {k:28s} {v/1e6:10.2f}M')
    ex = [r['path'] for r in rows if '_1' in r['path']]
    print(f'\npaths containing "_1": {len(ex)}')
    for p in ex[:6]:
        print('  ', p)
    print('\nsample action projection paths:')
    for r in rows:
        if any(k in r['path'] for k in ('action_in_proj', 'action_out_proj', 'time_mlp')):
            print('  ', r['path'], r['shape'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
