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
def build_config(*, batch_size: int, num_steps: int, lr_scale: float = 1.0,
                 warmup_steps: int | None = None):
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
    # 官方 CosineDecaySchedule：warmup_steps=1000、peak_lr=2.5e-5（固定版本 openpi）。
    # 300 步实验仍处在 warmup 内（LR 只到峰值的 ~30%），因此任何 LR 相关的对照必须显式记录。
    lr_kwargs = {}
    if lr_scale != 1.0:
        lr_kwargs['peak_lr'] = cfg.lr_schedule.peak_lr * lr_scale
    if warmup_steps is not None:
        lr_kwargs['warmup_steps'] = int(warmup_steps)
    if lr_kwargs:
        cfg = dataclasses.replace(cfg, lr_schedule=dataclasses.replace(cfg.lr_schedule,
                                                                     **lr_kwargs))
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


def make_item(meta_row: dict, ep: dict, t: int, label_mode: str = 'anchor') -> dict:
    """构造一条训练样本。

    ``label_mode``：
      * ``anchor``：``a_k = A[t+k] - S[t]``，部署端用 ``anchor_reference='fixed'``；
      * ``target_increment``：``a_0 = A[t] - S[t]``、``a_k = A[t+k] - A[t+k-1]``，
        部署端用 ``anchor_reference='recursive'``。两者都能精确重建专家命令序列，
        区别只在训练目标分布（Gate 4 的单变量）。
    夹爪标签始终逐行取 ``A[t+k,7]/0.04``（未来目标，不是常数）。
    """
    S = ep['state']
    A = ep['action']
    T = S.shape[0]
    last = min(t + HORIZON, T)
    idx = np.arange(t, last)
    st = S[t]
    q = st[:7].astype(np.float64)
    g_state = float(np.clip(st[7:9].mean() / FINGER_MAX_M, 0, 1))
    chunk = A[idx].astype(np.float64)
    if label_mode == 'anchor':
        dq = chunk[:, :7] - q[None, :]
    elif label_mode == 'target_increment':
        dq = np.empty_like(chunk[:, :7])
        dq[0] = chunk[0, :7] - q
        if dq.shape[0] > 1:
            dq[1:] = chunk[1:, :7] - chunk[:-1, :7]
    else:
        raise ValueError(f'未知 label_mode: {label_mode}')
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


def build_batch(meta, cache, transform, rng, batch_size: int, label_mode: str = 'anchor'):
    import jax

    picks = []
    for _ in range(batch_size):
        ep_i, start = WIN_IDX[int(rng.integers(len(WIN_IDX)))]
        picks.append((meta[ep_i], start))
    items, masks, skills = [], [], []
    for row, t in picks:
        ep = cache.get(row)
        raw = make_item(row, ep, t, label_mode=label_mode)
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


def save_opt_state(train_state, path, *, rng_jax, rng_np):
    """保存 optimizer 状态、step 与 RNG，供真正的续训恢复。"""
    import jax
    import numpy as np

    leaves, _ = jax.tree_util.tree_flatten(train_state.opt_state)
    arrs = {f'p{i}': np.asarray(jax.device_get(l)) for i, l in enumerate(leaves)}
    arrs['step'] = np.asarray(int(train_state.step))
    arrs['rng_jax'] = np.asarray(jax.random.key_data(jax.device_get(rng_jax)))
    # 兼容 numpy.random.Generator（default_rng）与 RandomState
    if hasattr(rng_np, 'bit_generator'):
        arrs['np_rng_json'] = np.asarray(json.dumps(rng_np.bit_generator.state))
        arrs['np_rng_kind'] = np.asarray('generator')
    else:
        arrs['np_rng_json'] = np.asarray(json.dumps(rng_np.get_state(), default=str))
        arrs['np_rng_kind'] = np.asarray('randomstate')
    np.savez(path, **arrs)
    return path, len(leaves)


def load_opt_state(train_state, path):
    """恢复 optimizer 状态、step 与 RNG（结构必须与当前 tx/params 一致）。"""
    import jax
    import jax.numpy as jnp
    import numpy as np

    z = np.load(path)
    leaves, treedef = jax.tree_util.tree_flatten(train_state.opt_state)
    n = len(leaves)
    got = len([k for k in z.files if k.startswith('p')])
    if got != n:
        raise SystemExit(f'opt state 结构不一致：checkpoint {got} 叶 vs 当前 {n} 叶')
    new = [jnp.asarray(z[f'p{i}']) for i in range(n)]
    opt_state = jax.tree_util.tree_unflatten(treedef, new)
    train_state = dataclasses.replace(train_state, opt_state=opt_state,
                                      step=int(z['step']))
    rng_jax = jax.random.wrap_key_data(jnp.asarray(z['rng_jax']).astype(jnp.uint32))
    kind = str(z['np_rng_kind'])
    payload = json.loads(str(z['np_rng_json']))
    if kind == 'generator':
        rng_np = np.random.default_rng()
        rng_np.bit_generator.state = payload
    else:
        rng_np = np.random.RandomState()
        rng_np.set_state(('MT19937', np.asarray(payload[1], dtype=np.uint32),
                          int(payload[2]), int(payload[3]), float(payload[4])))
    return train_state, rng_jax, rng_np


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
    # 官方 checkpoint 顶层是 {"params": <tree>}；不嵌套会让 restore_params 报 KeyError: 'params'
    ckptr.save(str(path), args=ocp.args.PyTreeSave({'params': full}))
    return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['probe', 'train'], default='probe')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1,
                    help='梯度累积；当前实现不支持 >1（会直接报错），不要据此宣称 effective batch 扩大')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--save-at', default='',
                    help='额外保存的步数，逗号分隔（例如 1200,2400）；与 --save-every 并用')
    ap.add_argument('--save-opt-at', default='',
                    help='在这些 step 保存 optimizer 状态（逗号分隔），用于真正续训')
    ap.add_argument('--resume-opt', default='',
                    help='从 opt_stepNNNNN.npz 恢复 optimizer/RNG/step（需配合 --init-lora）')
    ap.add_argument('--max-hours', type=float, default=0.0,
                    help='>0 时训练到该小时数就保存并退出（防止超时被强杀）')
    ap.add_argument('--val-every', type=int, default=0)
    ap.add_argument('--val-batches', type=int, default=8)
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--export-params', action='store_true', help='训练结束导出完整 params（orbax）')
    ap.add_argument('--init-lora', default='',
                    help='从已有 LoRA npz 继续训练：只恢复 51 个 LoRA 张量，optimizer state 与学习率调度会重新初始化（step901=502+400 两阶段累计，非一次连续优化）')
    ap.add_argument('--label-mode', choices=['anchor', 'target_increment'],
                    default='anchor',
                    help='anchor: a_k=A[t+k]-S[t] 配 fixed；target_increment: 逐步增量 配 recursive')
    ap.add_argument('--loss-dims', type=int, default=0,
                    help='0=官方 32 维 loss；8=只在前 8 维（7 关节+1 夹爪）求平均（P0b 对照实验）')
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    import jax
    import openpi.models.model as _model

    if args.accum != 1:
        raise SystemExit(
            'gradient accumulation 未实现：--accum>1 不会扩大 effective batch；'
            '当前只支持 --accum 1（审计 P2）。')
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
    save_opt_at = {int(x) for x in args.save_opt_at.split(',') if x.strip()}
    if args.resume_opt:
        import dataclasses as _dc  # noqa: F401
        train_state, jax_rng_resume, rng_np = load_opt_state(train_state, args.resume_opt)
        print(f'resumed optimizer state <- {args.resume_opt} at step {int(train_state.step)}',
              flush=True)
    else:
        jax_rng_resume = None
    print(f'weights loaded in {time.time() - t0:.1f}s', flush=True)
    total, trainable = count_params(train_state, cfg)
    print(f'params total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M '
          f'({100 * trainable / total:.2f}%)', flush=True)

    batch = build_batch(meta, cache, transform, rng_np, args.batch,
                           label_mode=args.label_mode)
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
        # probe 路径拿到的是 numpy batch；模型要求 jax 数组（jaxtyping 会拦截 numpy）
        obs = jax.tree.map(jnp.asarray, obs)
        actions = jnp.asarray(actions)
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
    jax_rng = jax_rng_resume if jax_rng_resume is not None else jax.random.key(args.seed)
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
                bt = build_batch(meta, cache, transform, rng_np, args.batch,
                        label_mode=args.label_mode)
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
    save_at = {int(x) for x in args.save_at.split(',') if x.strip()}
    start_step = int(train_state.step) + 1 if args.resume_opt else 2
    deadline = (t_start + args.max_hours * 3600.0) if args.max_hours > 0 else None
    for k in range(start_step, args.steps + 1):
        if deadline is not None and time.time() > deadline:
            print(f'[max-hours] stopping at step {k - 1} '
                  f'({(time.time() - t_start) / 3600:.2f} h); saving opt state', flush=True)
            p_opt, n_leaf = save_opt_state(train_state, OUT / 'ckpts' / f'opt_step{k - 1:05d}.npz',
                                           rng_jax=jax_rng, rng_np=rng_np)
            print(f'  saved opt state {p_opt} ({n_leaf} leaves)', flush=True)
            break
        batch = build_batch(meta, cache, transform, rng_np, args.batch,
                           label_mode=args.label_mode)
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
        if k % args.save_every == 0 or k == args.steps or k in save_at:
            size = save_ckpt(train_state, cfg, OUT / 'ckpts', k)
            (OUT / 'ckpts' / 'manifest.json').write_text(json.dumps({
                'label_mode': args.label_mode,
                'anchor_reference': ('fixed' if args.label_mode == 'anchor'
                                     else 'recursive'),
                'horizon': HORIZON, 'loss_dims': args.loss_dims,
                'steps_done': k, 'batch': args.batch,
                'lr': getattr(args, 'lr', None),  # 训练用官方 schedule，无独立 CLI
                'base_config': 'pi05_droid_finetune',
                'lora_variants': ['gemma_2b_lora', 'gemma_300m_lora'],
                'seed': args.seed, 'init_lora': args.init_lora or None,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  saved ckpt step {k} ({size/1e6:.1f} MB)', flush=True)
            if k in save_opt_at:
                p_opt, n_leaf = save_opt_state(
                    train_state, OUT / 'ckpts' / f'opt_step{k:05d}.npz',
                    rng_jax=jax_rng, rng_np=rng_np)
                print(f'  saved opt state step {k} ({n_leaf} leaves)', flush=True)

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
             'label_mode': args.label_mode,
             'anchor_reference': ('fixed' if args.label_mode == 'anchor' else 'recursive'),
             'horizon': HORIZON, 'loss_dims': args.loss_dims,
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
