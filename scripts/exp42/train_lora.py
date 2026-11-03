"""实验 4.2：π₀.₅ DROID 联合 LoRA 微调（A90 + B90 单一模型）。

因为 CentOS 7 / glibc 2.17 上无法安装 `lerobot`（`openpi.training.data_loader` 导入失败，
见 environment_manifest.json），本脚本：
  * **复用官方** `pi05_droid_finetune` 的模型配置（pi05=True, action_dim=32, horizon=16）、
    `droid_policy.DroidInputs/DroidOutputs`、`Normalize`（官方 DROID norm stats）、
    `TokenizePrompt`、`PadStatesAndActions`、`pi0.compute_loss`、官方 optimizer/lr schedule；
  * **只替换数据迭代器**：直接读 4.1 导出的 npy/npz（与 LeRobotDataset 等价的逐帧窗口采样）。

模式：
  --mode probe   构造一个 batch 并前向 loss（Gate 1），打印 shape/dtype/范围/mask 统计
  --mode train   真正反向传播 + optimizer 更新，保存 LoRA checkpoint 与 benchmark

用法（openpi venv）：
  python scripts/exp42/train_lora.py --mode probe
  python scripts/exp42/train_lora.py --mode train --steps 50 --batch 4 --accum 4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path(os.environ.get('EXP42_DATA', '/scratch/lijh/exp42/data'))
OUT = Path(os.environ.get('EXP42_OUT', '/scratch/lijh/exp42/run'))
ASSETS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets')
PARAMS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params')
FINGER_MAX_M = 0.04
HORIZON = 16


# ----------------------------------------------------------------- 配置
def build_config(*, batch_size: int, num_steps: int, lr_scale: float = 1.0):
    import openpi.models.model as _model
    import openpi.models.pi0_config as pi0_config
    import openpi.policies.droid_policy as droid_policy
    import openpi.training.config as _config
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as _transforms

    base = _config.get_config('pi05_droid_finetune')
    model_cfg = pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=HORIZON,
        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    data_factory = _config.SimpleDataConfig(
        assets=_config.AssetsConfig(assets_dir=str(ASSETS), asset_id='droid'),
        data_transforms=lambda m: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=_model.ModelType.PI05)],
            outputs=[droid_policy.DroidOutputs()]),
        base_config=_config.DataConfig(prompt_from_task=True))
    freeze = pi0_config.Pi0Config(
        paligemma_variant='gemma_2b_lora',
        action_expert_variant='gemma_300m_lora').get_freeze_filter()
    cfg = dataclasses.replace(
        base, model=model_cfg, data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARAMS)),
        freeze_filter=freeze, ema_decay=None,
        batch_size=int(batch_size), num_train_steps=int(num_steps))
    if lr_scale != 1.0:
        cfg = dataclasses.replace(cfg, lr_schedule=dataclasses.replace(
            cfg.lr_schedule, peak_value=cfg.lr_schedule.peak_value * lr_scale))
    return cfg


def make_transforms(cfg):
    import openpi.transforms as _transforms

    data_cfg = cfg.data.create(assets_dirs=ASSETS, model_config=cfg.model)
    chain = list(data_cfg.data_transforms.inputs) + [
        _transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm)
    ] + list(data_cfg.model_transforms.inputs)
    return _transforms.compose(chain), data_cfg


# ----------------------------------------------------------------- 数据
def load_meta() -> list[dict]:
    rows = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    return rows


def windows(meta: list[dict], split: str) -> list[tuple[int, int]]:
    out = []
    for i, m in enumerate(meta):
        if m['split'] != split:
            continue
        n = int(m['n_frames'])
        for t in range(0, max(1, n - HORIZON + 1), 4):
            out.append((i, t))
    return out


class EpisodeCache:
    def __init__(self, limit: int = 64):
        self.limit = limit
        self.cache = {}

    def get(self, meta_row: dict):
        key = (meta_row['skill'], meta_row['episode_index'])
        if key in self.cache:
            return self.cache[key]
        sa = np.load(DATA / meta_row['state_path'])
        top = np.load(DATA / meta_row['top_path'], mmap_mode='r')
        wrist = np.load(DATA / meta_row['wrist_path'], mmap_mode='r')
        item = {'state': sa['state'], 'action': sa['action'], 'top': top, 'wrist': wrist}
        if len(self.cache) >= self.limit:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = item
        return item


def make_item(meta_row: dict, ep: dict, t: int) -> dict:
    S = ep['state']
    A = ep['action']
    T = S.shape[0]
    last = min(t + HORIZON, T)
    idx = np.arange(t, last)
    st = S[t]
    q = st[:7].astype(np.float64)
    g_state = float(np.clip(st[7:9].mean() / FINGER_MAX_M, 0, 1))
    chunk = A[idx].astype(np.float64)
    dq = chunk[:, :7] - q[None, :]
    g_cmd = np.clip(chunk[:, 7] / FINGER_MAX_M, 0, 1)
    actions = np.concatenate([dq, g_cmd[:, None]], axis=1)          # (<=16, 8)
    mask = np.zeros((HORIZON,), dtype=bool)
    mask[:actions.shape[0]] = True
    if actions.shape[0] < HORIZON:                                   # episode 尾部：重复最后一帧
        pad = np.repeat(actions[-1:], HORIZON - actions.shape[0], axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    return {
        'observation/exterior_image_1_left': np.asarray(ep['top'][t], dtype=np.uint8),
        'observation/wrist_image_left': np.asarray(ep['wrist'][t], dtype=np.uint8),
        'observation/joint_position': q,
        'observation/gripper_position': np.asarray([g_state]),
        'prompt': meta_row['prompt'],
        'actions': actions,
        '_valid_mask': mask,
    }


def build_batch(meta, cache, transform, rng, batch_size: int):
    import jax

    picks = []
    for _ in range(batch_size):
        ep_i, start = WIN_IDX[int(rng.integers(len(WIN_IDX)))]
        picks.append((meta[ep_i], start))
    items, masks, skills = [], [], []
    for row, t in picks:
        ep = cache.get(row)
        raw = make_item(row, ep, t)
        masks.append(raw.pop('_valid_mask'))
        skills.append(row['skill'])
        items.append(transform(raw))
    batch = jax.tree.map(lambda *xs: np.stack(xs, 0), *items)
    batch['_action_mask'] = np.stack(masks, 0)
    batch['_skills'] = np.asarray(skills)
    return batch


WIN_IDX: list[tuple[int, int]] = []


# ----------------------------------------------------------------- 训练
def init_state(cfg):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.optimizer as _optimizer
    import openpi.training.utils as training_utils

    tx = _optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
    rng = jax.random.key(0)
    model = cfg.model.create(rng)
    graphdef, state = nnx.split(model)
    target = state.to_pure_dict()
    partial = cfg.weight_loader.load(target)
    state.replace_by_pure_dict(partial)
    model = nnx.merge(graphdef, state)
    params = nnx.state(model)
    params = nnx_utils.state_map(params, cfg.freeze_filter,
                                 lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    train_state = training_utils.TrainState(
        step=0, params=params, model_def=nnx.graphdef(model), tx=tx,
        opt_state=tx.init(params.filter(cfg.trainable_filter)),
        ema_decay=cfg.ema_decay, ema_params=None)
    return train_state


def flow_sq(model, rng, observation, actions, *, train: bool, horizon: int):
    """与 pi0.compute_loss 完全相同的 flow-matching 计算，但返回**逐维**平方误差 (B,H,32)。

    用于"有效动作维度 masked loss"（review 4.2-P0b 第三步）：官方 compute_loss 对 32 维求均值，
    这里保留逐维结果，由调用方决定对哪些维度求平均。
    """
    import jax
    import jax.numpy as jnp
    import openpi.models.model as _model
    from openpi.models.pi0 import make_attn_mask

    pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    obs = _model.preprocess_observation(pre_rng, observation, train=train)
    bs = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    tm = jax.random.beta(time_rng, 1.5, 1, bs) * 0.999 + 0.001
    te = tm[..., None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions
    pre_tok, pre_mask, pre_ar = model.embed_prefix(obs)
    suf_tok, suf_mask, suf_ar, adarms = model.embed_suffix(obs, x_t, tm)
    input_mask = jnp.concatenate([pre_mask, suf_mask], axis=1)
    ar_mask = jnp.concatenate([pre_ar, suf_ar], axis=0)
    attn = make_attn_mask(input_mask, ar_mask)
    pos = jnp.cumsum(input_mask, axis=1) - 1
    (_, suf_out), _ = model.PaliGemma.llm([pre_tok, suf_tok], mask=attn, positions=pos,
                                          adarms_cond=[None, adarms])
    v_t = model.action_out_proj(suf_out[:, -horizon:])
    return jnp.square(v_t - u_t)


def make_train_step(cfg, loss_dims: int = 0):
    """loss_dims=0 用官方 32 维 loss；loss_dims=8 只对前 8 维（7 关节 + 1 夹爪）求平均。"""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    import openpi.models.model as _model
    import openpi.shared.array_typing as at

    @at.typecheck
    def train_step(rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        @at.typecheck
        def loss_fn(model, rng, observation, actions):
            if loss_dims <= 0:
                return jnp.mean(model.compute_loss(rng, observation, actions, train=True))
            sq = flow_sq(model, rng, observation, actions, train=True,
                         horizon=model.action_horizon)
            return jnp.mean(sq[..., :loss_dims])

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, cfg.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model, train_rng, observation, actions)
        params = state.params.filter(cfg.trainable_filter)
        updates, new_opt = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_state = dataclasses.replace(state, step=state.step + 1,
                                        params=nnx.state(model), opt_state=new_opt)
        info = {'loss': loss, 'grad_norm': optax.global_norm(grads)}
        return new_state, info

    return jax.jit(train_step)


def count_params(train_state, cfg):
    import jax

    def numel(x):
        v = getattr(x, 'value', x)
        return int(np.prod(np.shape(v)))

    total = int(sum(numel(x) for x in jax.tree.leaves(train_state.params)))
    trainable = int(sum(numel(x) for x in
                        jax.tree.leaves(train_state.params.filter(cfg.trainable_filter))))
    return total, trainable


def _flat_np(payload: dict) -> dict:
    import flax.traverse_util as traverse_util
    import jax

    def to_np(x):
        return np.asarray(getattr(x, 'value', x))

    flat = traverse_util.flatten_dict(payload, sep='/')
    return {k.replace('/', '__'): to_np(v) for k, v in flat.items()}


def save_ckpt(train_state, cfg, path: Path, step: int) -> int:
    """保存**只含 LoRA 可训练参数**的 npz（小、快、便于审计）。"""
    path.mkdir(parents=True, exist_ok=True)
    lora = train_state.params.filter(cfg.trainable_filter)
    arrays = _flat_np(lora.to_pure_dict())
    f = path / f'lora_step{step:05d}.npz'
    np.savez_compressed(f, **arrays, step=np.asarray(step))
    return f.stat().st_size


def export_full_params(train_state, path: Path) -> str:
    """把（基座 + LoRA）合并后的**完整** params 存成 orbax OCDBT，供 4.0 推理服务直接加载。"""
    import jax
    import orbax.checkpoint as ocp

    if path.exists():
        import shutil

        shutil.rmtree(path)
    full = train_state.params.to_pure_dict()
    full = jax.tree.map(lambda x: np.asarray(getattr(x, 'value', x)), full)
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(str(path), args=ocp.args.PyTreeSave(full))
    return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['probe', 'train'], default='probe')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--val-every', type=int, default=0)
    ap.add_argument('--val-batches', type=int, default=8)
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--export-params', action='store_true', help='训练结束导出完整 params（orbax）')
    ap.add_argument('--init-lora', default='', help='从已有 LoRA npz 继续训练（尺度/收敛诊断用）')
    ap.add_argument('--loss-dims', type=int, default=0,
                    help='0=官方 32 维 loss；8=只在前 8 维（7 关节+1 夹爪）求平均（P0b 对照实验）')
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    import jax
    import openpi.models.model as _model

    meta = load_meta()
    cfg = build_config(batch_size=args.batch, num_steps=max(args.steps, 1))
    transform, data_cfg = make_transforms(cfg)
    global WIN_IDX
    WIN_IDX = windows(meta, args.train_split)
    print(f'train windows: {len(WIN_IDX)}  episodes: {sum(1 for m in meta if m["split"] == args.train_split)}')
    cache = EpisodeCache()
    rng_np = np.random.default_rng(args.seed)

    t0 = time.time()
    print('loading model + weights ...', flush=True)
    train_state = init_state(cfg)
    if args.init_lora:
        import flax.nnx as nnx
        import flax.traverse_util as traverse_util
        import jax.numpy as jnp
        z = np.load(args.init_lora)
        flat_lora = {tuple(k.split('__')): z[k] for k in z.files if k != 'step'}
        st = nnx.state(nnx.merge(train_state.model_def, train_state.params))
        flat = traverse_util.flatten_dict(st.to_pure_dict())
        n = 0
        for k, v in flat_lora.items():
            if k in flat:
                flat[k] = jnp.asarray(v)
                n += 1
        st.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
        train_state = dataclasses.replace(train_state, params=st)
        print(f'resumed from {args.init_lora}: {n}/{len(flat_lora)} tensors', flush=True)
    print(f'weights loaded in {time.time() - t0:.1f}s', flush=True)
    total, trainable = count_params(train_state, cfg)
    print(f'params total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M '
          f'({100 * trainable / total:.2f}%)', flush=True)

    batch = build_batch(meta, cache, transform, rng_np, args.batch)
    print('batch keys:', sorted(k for k in batch if not k.startswith('_')), flush=True)
    obs = _model.Observation.from_dict({k: v for k, v in batch.items() if not k.startswith('_')})
    actions = batch['actions']
    print(f'state {obs.state.shape} {obs.state.dtype} | actions {actions.shape} {actions.dtype}',
          flush=True)
    print(f'images {[(k, v.shape, str(v.dtype)) for k, v in obs.images.items()]}', flush=True)
    print(f'image_masks {[(k, bool(np.asarray(v).ravel()[0])) for k, v in obs.image_masks.items()]}',
          flush=True)
    print(f'tokenized_prompt {obs.tokenized_prompt.shape} sum_mask='
          f'{int(np.asarray(obs.tokenized_prompt_mask).sum())}', flush=True)
    print(f'action range [{float(actions.min()):.3f}, {float(actions.max()):.3f}] '
          f'| state range [{float(obs.state.min()):.3f}, {float(obs.state.max()):.3f}]', flush=True)
    print(f'action_mask valid per sample: {batch["_action_mask"].sum(axis=1).tolist()}', flush=True)
    print(f'batch skills: {batch["_skills"].tolist()}', flush=True)

    if args.mode == 'probe':
        import flax.nnx as nnx
        import jax.numpy as jnp

        model = nnx.merge(train_state.model_def, train_state.params)
        rng = jax.random.key(args.seed)
        loss = jnp.mean(model.compute_loss(rng, obs, actions, train=True))
        loss = float(jax.device_get(loss))
        print(f'FORWARD LOSS = {loss:.6f}', flush=True)
        (OUT / 'gate1_forward_loss.json').write_text(json.dumps({
            'loss': loss, 'batch_size': args.batch,
            'state_shape': list(obs.state.shape), 'action_shape': list(actions.shape),
            'action_dtype': str(actions.dtype), 'state_dtype': str(obs.state.dtype),
            'images': {k: list(v.shape) for k, v in obs.images.items()},
            'tokenized_prompt_shape': list(obs.tokenized_prompt.shape),
            'token_mask_sum': int(np.asarray(obs.tokenized_prompt_mask).sum()),
            'action_mask_valid': batch['_action_mask'].sum(axis=1).tolist(),
            'skills': batch['_skills'].tolist(),
            'norm_stats_source': 'openpi-assets/checkpoints/pi05_droid/assets/droid (official)',
            'params_total': total, 'params_trainable': trainable,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0

    step_fn = make_train_step(cfg, loss_dims=args.loss_dims)
    jax_rng = jax.random.key(args.seed)
    t_compile = time.time()
    batch0 = (obs, actions)
    train_state, info = step_fn(jax_rng, train_state, batch0)
    jax.block_until_ready(train_state.params)
    compile_s = time.time() - t_compile
    print(f'first step (incl. JIT) {compile_s:.1f}s loss={float(info["loss"]):.4f}', flush=True)

    log = []
    t_start = time.time()
    n_done = 1

    def eval_losses(n_batches: int) -> dict:
        """在 train/val 上各算若干 batch 的 loss（不更新参数）。"""
        import flax.nnx as nnx
        import jax.numpy as jnp

        global WIN_IDX
        model = nnx.merge(train_state.model_def, train_state.params)
        model.eval()
        out = {}
        saved = list(WIN_IDX)
        for split in ('train', 'val'):
            WIN_IDX = windows(meta, split)
            if not WIN_IDX:
                out[split] = None
                continue
            vals = []
            for b in range(n_batches):
                bt = build_batch(meta, cache, transform, rng_np, args.batch)
                ob = _model.Observation.from_dict(
                    {kk: v for kk, v in bt.items() if not kk.startswith('_')})
                # compute_loss 有 jaxtyping 检查，必须是 jax 数组而不是 numpy
                ob = jax.tree.map(lambda x: jax.numpy.asarray(x), ob)
                acts = jax.numpy.asarray(bt['actions'])
                loss = jnp.mean(model.compute_loss(jax.random.key(1000 + b), ob, acts,
                                                   train=False))
                vals.append(float(jax.device_get(loss)))
            out[split] = float(np.mean(vals))
        WIN_IDX = saved
        return out

    # 注意：上面的 pre-loop step 已经是第 1 次参数更新，因此这里从 2 开始，
    # 使 `--steps N` 恰好对应 N 次 optimizer update（独立审计缺陷 #4）。
    for k in range(2, args.steps + 1):
        batch = build_batch(meta, cache, transform, rng_np, args.batch)
        b_obs = _model.Observation.from_dict(
            {kk: v for kk, v in batch.items() if not kk.startswith('_')})
        train_state, info = step_fn(jax_rng, train_state, (b_obs, batch['actions']))
        if k % 5 == 0 or k == 1:
            jax.block_until_ready(train_state.params)
            el = time.time() - t_start
            sps = (k - 1) / max(el, 1e-6)
            entry = {'step': k, 'loss': float(info['loss']),
                     'grad_norm': float(info['grad_norm']), 'sps': sps,
                     'skills': batch['_skills'].tolist()}
            if args.val_every and (k % args.val_every == 0):
                entry['eval'] = eval_losses(args.val_batches)
            log.append(entry)
            extra = f' eval={entry["eval"]}' if 'eval' in entry else ''
            print(f'step {k}/{args.steps} loss={float(info["loss"]):.4f} '
                  f'grad={float(info["grad_norm"]):.3f} {sps:.3f} step/s{extra}', flush=True)
        n_done = k
        if k % args.save_every == 0 or k == args.steps:
            size = save_ckpt(train_state, cfg, OUT / 'ckpts', k)
            print(f'  saved ckpt step {k} ({size/1e6:.1f} MB)', flush=True)

    if args.export_params:
        t_exp = time.time()
        p = export_full_params(train_state, OUT / 'ckpts' / f'params_step{args.steps:05d}')
        print(f'exported full params -> {p} in {time.time() - t_exp:.1f}s', flush=True)

    total_s = time.time() - t_start
    sps = max(args.steps - 1, 1) / max(total_s, 1e-6)
    mem = {}
    try:
        import subprocess

        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True)
        mem['nvidia_smi'] = out.stdout.strip().splitlines()
    except Exception:
        pass
    bench = {'steps': args.steps, 'optimizer_updates': args.steps,
             'note': 'steps = 实际 optimizer update 次数（已修正旧版 +1 偏差）',
             'batch_size': args.batch, 'accum': args.accum,
             'seconds_per_optimizer_step': total_s / max(args.steps, 1),
             'seconds_per_microbatch': total_s / max(args.steps * args.accum, 1),
             'compile_seconds': compile_s, 'steps_per_second': sps,
             'checkpoint_save_seconds': None, 'gpu': mem,
             'params_total': total, 'params_trainable': trainable,
             'trainable_ratio': trainable / total,
             'windows': len(WIN_IDX),
             'epochs_equivalent': args.steps * args.batch / max(len(WIN_IDX), 1),
             'log': log}
    (OUT / 'training_speed_benchmark.json').write_text(
        json.dumps(bench, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps({k: bench[k] for k in ('steps', 'batch_size',
                                            'seconds_per_optimizer_step', 'compile_seconds',
                                            'steps_per_second', 'epochs_equivalent')},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
