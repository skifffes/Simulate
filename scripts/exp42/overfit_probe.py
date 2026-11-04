"""P1：小样本过拟合测试（8 个固定训练窗口，anchor-16 表示）。

问题（审计 §五）：现有 LoRA 方案**能不能**学会少量确定的"状态→动作"映射？
如果连 8 个固定窗口都拟合不了，就不该扩训练预算，而应先查 warmup/监督维度/权重更新。

做法：
  * 取 train split 里 8 个固定窗口（每步都用全部 8 个，batch=8，loss=32D flow-matching）；
  * 每 10 步记录一次 loss；
  * 训练结束后导出完整 params（nested），交给 expert_state_query 做采样对比。

用法（服务器 openpi venv，GPU）：
  python scripts/exp42/overfit_probe.py --steps 150 --out /scratch/lijh/exp42/overfit/run1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_trainer():
    spec = importlib.util.spec_from_file_location('train_lora_mod',
                                                  ROOT / 'scripts' / 'exp42' / 'train_lora.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--n-windows', type=int, default=8)
    ap.add_argument('--label-mode', default='anchor')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/scratch/lijh/exp42/overfit/run1')
    ap.add_argument('--warmup-steps', type=int, default=10,
                    help='官方默认 warmup=1000；过拟合诊断用短 warmup 以观察可学性')
    ap.add_argument('--lr-scale', type=float, default=4.0,
                    help='warmup 内（1000 步）学习率很小，这里放大以便观察可学性；记录在 json')
    args = ap.parse_args(argv)

    import jax

    T = load_trainer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = T.load_meta()
    all_windows = T.windows(meta, 'train')
    # 取**分散**的 8 个窗口（覆盖 A90/B90 与不同阶段），确定性选择
    idx = np.linspace(0, len(all_windows) - 1, args.n_windows).astype(int)
    picks = [all_windows[int(i)] for i in idx]
    T.WIN_IDX = picks
    cache = T.EpisodeCache()
    cfg = T.build_config(batch_size=args.n_windows, num_steps=args.steps,
                         lr_scale=args.lr_scale, warmup_steps=args.warmup_steps)
    transform, data_cfg = T.make_transforms(cfg)
    state = T.init_state(cfg)
    step_fn = T.make_train_step(cfg, loss_dims=0)
    rng = np.random.default_rng(args.seed)

    raw_batch = T.build_batch(meta, cache, transform, rng, args.n_windows,
                              label_mode=args.label_mode)
    import openpi.models.model as _model
    obs = _model.Observation.from_dict({k: v for k, v in raw_batch.items()
                                        if not k.startswith('_')})
    batch = (obs, raw_batch['actions'])
    jax_rng = jax.random.key(args.seed)
    log = []
    t0 = time.time()
    state, info = step_fn(jax_rng, state, batch)
    jax.block_until_ready(state.params)
    print(f'first step {time.time() - t0:.1f}s loss={float(info["loss"]):.4f}', flush=True)
    log.append({'step': 1, 'loss': float(info['loss'])})
    for k in range(2, args.steps + 1):
        state, info = step_fn(jax_rng, state, batch)
        if k % 10 == 0 or k == args.steps:
            loss = float(info['loss'])
            log.append({'step': k, 'loss': loss, 'sps': k / (time.time() - t0)})
            print(f'step {k}/{args.steps} loss={loss:.5f} {k / (time.time() - t0):.2f} step/s',
                  flush=True)

    params_dir = T.export_full_params(state, out / f'params_step{args.steps:05d}')
    print('exported', params_dir, flush=True)

    # 同 8 窗口的 picks，供 expert_state_query 采样
    picks_json = {'items': []}
    for ep_i, start in picks:
        row = meta[ep_i]
        picks_json['items'].append({'skill': row['skill'], 'episode': row['episode_index'],
                                    'tick': int(start), 'phase': 'overfit',
                                    'robot_block': 0, 'block_wall': 0, 'frame': int(start)})
    (out / 'overfit_picks.json').write_text(json.dumps(picks_json, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    (out / 'overfit_log.json').write_text(json.dumps({
        'steps': args.steps, 'n_windows': args.n_windows, 'label_mode': args.label_mode,
        'lr_scale': args.lr_scale, 'warmup_steps': args.warmup_steps,
        'seed': args.seed,
        'windows': [{'skill': meta[i]['skill'], 'episode': meta[i]['episode_index'],
                     'start': int(s)} for i, s in picks],
        'log': log, 'final_loss': log[-1]['loss'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', out / 'overfit_log.json', 'final loss', log[-1]['loss'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
