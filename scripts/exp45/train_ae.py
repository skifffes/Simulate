"""实验 4.5 正式训练：π₀.₅ **Action Expert-only** 全参数微调（非 LoRA）。

与 4.4 Group B 完全相同的：数据（export_v3，311 条）、batch 4、seed 0、Anchor-16、
官方 DROID normalization、32D flow matching、CosineDecay(warmup 1000, peak 2.5e-5, decay 30000)。

唯一变量：微调参数范围 = Action Expert（layers/*_1、final_norm_1）+ action_in/out_proj +
time_mlp_in/out，共 430.1M / 3353.4M（12.83%）；PaliGemma（img + llm 非 _1）全部冻结。

用法：
  python scripts/exp45/train_ae.py --steps 12000 --batch 4 --seed 0 \
      --save-at 1200,2400,4000,6000,8000,10000,12000 --save-opt-at 6000,12000 \
      --out /scratch/lijh/exp45/run --max-hours 9
"""
from __future__ import annotations

import argparse
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
import train_lora as TL  # noqa: E402
from exp45_common import build_config_ae, DATA, TRAINABLE_PATTERNS  # noqa: E402

HORIZON = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-at', default='1200,2400,4000,6000,8000,10000,12000')
    ap.add_argument('--save-opt-at', default='6000,12000')
    ap.add_argument('--out', default='/scratch/lijh/exp45/run')
    ap.add_argument('--resume-params', default='', help='trainable npz（lora_step*.npz 格式）')
    ap.add_argument('--resume-opt', default='')
    ap.add_argument('--max-hours', type=float, default=0.0)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--label-mode', default='anchor')
    args = ap.parse_args()

    out = Path(args.out)
    ck = out / 'ckpts'
    ck.mkdir(parents=True, exist_ok=True)
    TL.DATA = DATA
    TL.OUT = out
    TL.HORIZON = HORIZON

    cfg = build_config_ae(batch_size=args.batch, num_steps=args.steps)
    t0 = time.time()
    print(f'[{time.strftime("%H:%M:%S")}] building model + loading pi05_droid ...', flush=True)
    train_state = TL.init_state(cfg)
    print(f'[{time.strftime("%H:%M:%S")}] weights loaded in {time.time()-t0:.1f}s', flush=True)
    total, trainable = TL.count_params(train_state, cfg)
    print(f'params total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M '
          f'({100*trainable/total:.2f}%)', flush=True)

    def apply_params_from_npz(state, path):
        z = np.load(path)
        flat_in = {tuple(k.split('__')): z[k] for k in z.files if k != 'step'}
        st = nnx.state(nnx.merge(state.model_def, state.params))
        flat = traverse_util.flatten_dict(st.to_pure_dict())
        n = 0
        for k, v in flat_in.items():
            if k in flat:
                flat[k] = jnp.asarray(v)
                n += 1
        st.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
        st_ = __import__('dataclasses').replace(state, params=st)
        print(f'resumed params: {n}/{len(flat_in)} tensors from {path}', flush=True)
        return st_

    if args.resume_params:
        train_state = apply_params_from_npz(train_state, args.resume_params)

    meta = TL.load_meta()
    cache = TL.EpisodeCache()
    transform, _ = TL.make_transforms(cfg)
    rng_np = np.random.default_rng(args.seed)
    TL.WIN_IDX = TL.windows(meta, args.train_split if hasattr(args, 'train_split') else 'train')
    print(f'train windows {len(TL.WIN_IDX)} episodes {len(meta)}', flush=True)

    step_fn = TL.make_train_step(cfg, loss_dims=0)
    jax_rng = jax.random.key(args.seed)
    save_opt_at = {int(x) for x in args.save_opt_at.split(',') if x.strip()}
    start_step = 2

    batch = TL.build_batch(meta, cache, transform, rng_np, args.batch,
                           label_mode=args.label_mode)
    obs = _model.Observation.from_dict({k: v for k, v in batch.items()
                                        if not k.startswith('_')})
    print(f'[{time.strftime("%H:%M:%S")}] first step (incl. JIT) ...', flush=True)
    t1 = time.time()
    train_state, info = step_fn(jax_rng, train_state, (obs, batch['actions']))
    jax.block_until_ready(train_state.params)
    print(f'[{time.strftime("%H:%M:%S")}] first step {time.time()-t1:.1f}s '
          f'loss={float(info["loss"]):.4f} grad={float(info["grad_norm"]):.3f}', flush=True)
    # warm-up step 已经消耗了一次更新与一批数据；续训时必须在它**之后**恢复参数与优化器状态，
    # 否则会出现数据/RNG 前进一格、loss 序列整体错位（Gate 4 曾实测到该 off-by-one）。
    if args.resume_params:
        train_state = apply_params_from_npz(train_state, args.resume_params)
    if args.resume_opt:
        train_state, jax_rng, rng_np = TL.load_opt_state(train_state, args.resume_opt)
        start_step = int(train_state.step) + 1
        print(f'resumed optimizer state at step {int(train_state.step)}', flush=True)

    save_at = {int(x) for x in args.save_at.split(',') if x.strip()}
    deadline = (t1 + args.max_hours * 3600.0) if args.max_hours > 0 else None
    t2 = time.time()
    n_done = 1
    log = []
    for k in range(max(2, start_step), args.steps + 1):
        if deadline is not None and time.time() > deadline:
            print(f'[max-hours] stopping at step {k-1}', flush=True)
            TL.save_opt_state(train_state, ck / f'opt_step{k-1:05d}.npz',
                              rng_jax=jax_rng, rng_np=rng_np)
            break
        b = TL.build_batch(meta, cache, transform, rng_np, args.batch,
                           label_mode=args.label_mode)
        o = _model.Observation.from_dict({kk: v for kk, v in b.items()
                                          if not kk.startswith('_')})
        train_state, info = step_fn(jax_rng, train_state, (o, b['actions']))
        n_done = k
        if k % max(1, args.log_every) == 0 or k in save_at:
            jax.block_until_ready(train_state.params)
            el = time.time() - t2
            sps = (k - 1) / max(el, 1e-6)
            entry = {'step': k, 'loss': float(info['loss']),
                     'grad_norm': float(info['grad_norm']), 'step_per_s': sps}
            log.append(entry)
            print(f'[{time.strftime("%H:%M:%S")}] step {k}/{args.steps} '
                  f'loss={entry["loss"]:.4f} grad={entry["grad_norm"]:.3f} {sps:.3f} step/s',
                  flush=True)
        if k in save_at or k == args.steps:
            size = TL.save_ckpt(train_state, cfg, ck, k)
            (ck / 'manifest.json').write_text(json.dumps({
                'mode': 'action_expert_full_ft', 'steps_done': k, 'batch': args.batch,
                'seed': args.seed, 'label_mode': args.label_mode,
                'anchor_reference': 'fixed', 'horizon': HORIZON, 'loss_dims': 0,
                'base_config': 'pi05_droid_finetune',
                'trainable_patterns': TRAINABLE_PATTERNS,
                'n_trainable': int(trainable), 'n_total': int(total),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  saved ckpt step {k} ({size/1e6:.1f} MB)', flush=True)
            if k in save_opt_at:
                TL.save_opt_state(train_state, ck / f'opt_step{k:05d}.npz',
                                  rng_jax=jax_rng, rng_np=rng_np)
                print(f'  saved opt state step {k}', flush=True)

    total_s = time.time() - t2
    bench = {'steps_requested': args.steps, 'steps_done': n_done,
             'batch': args.batch, 'seed': args.seed,
             'seconds_per_step': total_s / max(n_done - 1, 1),
             'steps_per_second': max(n_done - 1, 1) / max(total_s, 1e-6),
             'n_trainable': int(trainable), 'n_total': int(total),
             'train_windows': len(TL.WIN_IDX),
             'epochs_equivalent': args.batch * n_done / max(len(TL.WIN_IDX), 1),
             'log': log}
    (out / 'training_benchmark.json').write_text(json.dumps(bench, ensure_ascii=False,
                                                            indent=2), encoding='utf-8')
    with open(out / 'training_curve.csv', 'w', encoding='utf-8') as fh:
        fh.write('step,loss,grad_norm,step_per_s\n')
        for e in log:
            fh.write(f"{e['step']},{e['loss']},{e['grad_norm']},{e['step_per_s']}\n")
    print(json.dumps({k: v for k, v in bench.items() if k != 'log'}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
