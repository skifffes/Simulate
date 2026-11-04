"""Gate 4 任务 2（第 2 步）：冻结 step901，在**专家状态**上比较模型动作与专家动作。

输入：`expert_state_picks.json`（阶段化 tick）+ 真实数据集（图像 / state / prompt）。
对每个选中状态、每个固定 seed（默认 0..3）：
  1. 用与闭环完全相同的 obs 构造（两路 256×256 图、7 维 q、1 维夹爪、同一 prompt）；
  2. `policy._rng = jax.random.key(seed)` 后 `policy.infer(obs)`（与 AuditingPolicy 一致）；
  3. 记下 chunk 首行（物理 8 维）与整段 16 行。

对照：专家首行 `A[t]-S[t]`（关节）与 `A[t,7]`（夹爪米）；方向余弦、幅值比、夹爪差。

判读：若在专家状态上首行仍明显错 → 条件动作学习不足；
若专家状态上首行正确、而自主闭环失败 → 优先怀疑早期偏差造成的状态分布漂移。

用法（服务器 openpi venv，需要 GPU）：
  python scripts/exp42/expert_state_query.py --ckpt /scratch/lijh/exp42/run2/ckpts/params_nested \
      --picks results/experiment_4_2/contract_replay_p0c/expert_state_picks.json \
      --out results/experiment_4_2/contract_replay_p0c/expert_state_diagnostic.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = Path('/scratch/lijh/exp42/data')
FINGER_MAX_M = 0.04


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--picks', required=True)
    ap.add_argument('--seeds', default='0,1,2,3')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    import jax
    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config

    from exp42 import registry

    picks = json.loads(Path(args.picks).read_text(encoding='utf-8'))['items']
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    cached = {}

    def episode_row(skill, ep):
        row = next(m for m in meta if m['skill'] == skill and m['episode_index'] == ep)
        if row['state_path'] not in cached:
            cached[row['state_path']] = np.load(DATA / row['state_path'])
        return row, cached[row['state_path']]

    registry.register()
    cfg = _config.get_config(registry.CONFIG_NAME)
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    seeds = [int(s) for s in args.seeds.split(',')]

    out_rows = []
    for pick in picks:
        row, z = episode_row(pick['skill'], pick['episode'])
        S = z['state'].astype(np.float64)
        A = z['action'].astype(np.float64)
        t = int(pick['frame'])
        top = np.load(DATA / row['top_path'], mmap_mode='r')
        wrist = np.load(DATA / row['wrist_path'], mmap_mode='r')
        obs = {
            'observation/exterior_image_1_left': np.asarray(top[t], dtype=np.uint8),
            'observation/wrist_image_left': np.asarray(wrist[t], dtype=np.uint8),
            'observation/joint_position': S[t, :7].astype(np.float64),
            'observation/gripper_position': np.asarray(
                [np.clip(S[t, 7:9].mean() / FINGER_MAX_M, 0, 1)], dtype=np.float64),
            'prompt': row['prompt'],
        }
        expert_chunk = np.zeros((16, 8))
        n = min(16, S.shape[0] - t)
        expert_chunk[:n, :7] = A[t:t + n, :7] - S[t, :7][None, :]     # anchor 标签（训练语义）
        expert_chunk[:n, 7] = np.clip(A[t:t + n, 7] / FINGER_MAX_M, 0, 1)
        for k in range(n, 16):
            expert_chunk[k] = expert_chunk[n - 1]

        for seed in seeds:
            if hasattr(policy, '_rng'):
                policy._rng = jax.random.key(seed)
            chunk = np.asarray(policy.infer(obs)['actions'], dtype=np.float64)[:16, :8]
            out_rows.append({
                'skill': pick['skill'], 'episode': pick['episode'], 'tick': int(pick['tick']),
                'phase': pick['phase'], 'seed': seed,
                'q': S[t, :7], 'expert_first': expert_chunk[0, :7],
                'model_first': chunk[0, :7],
                'expert_gripper': float(expert_chunk[0, 7]),
                'model_gripper': float(chunk[0, 7]),
                'expert_chunk': expert_chunk, 'model_chunk': chunk,
            })
        print(f"{pick['skill']} ep{pick['episode']} tick {pick['tick']:3d} {pick['phase']:14s} "
              f"|expert_first|={np.abs(expert_chunk[0, :7]).max():.4f} "
              f"|model_first|={np.abs(out_rows[-1]['model_first']).max():.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        skill=np.array([r['skill'] for r in out_rows]),
        episode=np.array([r['episode'] for r in out_rows]),
        tick=np.array([r['tick'] for r in out_rows]),
        phase=np.array([r['phase'] for r in out_rows]),
        seed=np.array([r['seed'] for r in out_rows]),
        q=np.stack([r['q'] for r in out_rows]),
        expert_first=np.stack([r['expert_first'] for r in out_rows]),
        model_first=np.stack([r['model_first'] for r in out_rows]),
        expert_gripper=np.array([r['expert_gripper'] for r in out_rows]),
        model_gripper=np.array([r['model_gripper'] for r in out_rows]),
        expert_chunk=np.stack([r['expert_chunk'] for r in out_rows]),
        model_chunk=np.stack([r['model_chunk'] for r in out_rows]),
    )
    print('wrote', out, 'rows', len(out_rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
