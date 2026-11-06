"""实验 4.7 Gate 5：离线训练路径 vs 在线 Policy 推理的端到端数值一致性。

同一帧、同一 checkpoint（exp46 step3000）、同一 RNG：
  1. 训练路径输入 → 直接调用 model.sample_actions(noise=key(seed))
  2. 在线路径 obs → policy._input_transform → policy.infer(...)（内部同样 sample_actions）
比较归一化空间 16×32 输出、反归一化 16×8、以及 Policy 的 metadata（确认 checkpoint 来源）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/scratch/lijh/exp40/repo')
for p in (REPO / 'scripts' / 'exp42', REPO / 'src', REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi.models.model as _model  # noqa: E402
import openpi.policies.policy_config as _policy_config  # noqa: E402
import train_lora as TL  # noqa: E402
from exp42 import registry  # noqa: E402

DATA = Path('/scratch/lijh/exp46/export_overfit')
OUT = REPO / 'results' / 'experiment_4_7'
CKPT = Path('/scratch/lijh/exp47/merged/step3000')
FINGER_MAX_M = 0.04


def main() -> int:
    cfg = registry.build_config()
    policy = _policy_config.create_trained_policy(cfg, str(CKPT))
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    rows = []
    for row in meta:
        z = np.load(DATA / row['state_path'])
        S = z['state'].astype(np.float64)
        ep = {'top': np.load(DATA / row['top_path'], mmap_mode='r'),
              'wrist': np.load(DATA / row['wrist_path'], mmap_mode='r'),
              'state': z['state'].astype(np.float64), 'action': z['action'].astype(np.float64)}
        for t in (max(0, len(S) // 3), max(0, len(S) // 2)):
            item = TL.make_item(row, ep, t, label_mode='anchor')
            train_out = TL.make_transforms(cfg)[0](item)
            obs_batched = _model.Observation.from_dict({
                'image': {k: jnp.asarray(v)[None, ...] for k, v in train_out['image'].items()},
                'image_mask': {k: jnp.asarray(v)[None, ...]
                               for k, v in train_out['image_mask'].items()},
                'state': jnp.asarray(train_out['state'])[None, ...],
                'tokenized_prompt': jnp.asarray(train_out['tokenized_prompt'])[None, ...],
                'tokenized_prompt_mask': jnp.asarray(train_out['tokenized_prompt_mask'])[None, ...],
            })
            model = policy._model
            # Policy.infer 会先 split 一次 rng，再用于 sample_actions；离线侧必须用同一个 key
            rng_sample = jax.random.split(jax.random.key(0))[1]
            acts_off = np.asarray(model.sample_actions(rng_sample, obs_batched))[0]

            obs = {
                'observation/exterior_image_1_left': np.asarray(ep['top'][t], dtype=np.uint8),
                'observation/wrist_image_left': np.asarray(ep['wrist'][t], dtype=np.uint8),
                'observation/joint_position': S[t, :7].astype(np.float64),
                'observation/gripper_position': np.asarray(
                    [np.clip(S[t, 7:9].mean() / FINGER_MAX_M, 0, 1)], dtype=np.float64),
                'prompt': row['prompt'],
            }
            policy._rng = jax.random.key(0)
            out_on = policy.infer(obs)
            acts_on = np.asarray(out_on['actions'], dtype=np.float64)[:16, :8]

            # 离线 16×32 → 反归一化到 8D
            if acts_off is not None:
                off8 = np.asarray(policy._output_transform({
                    'state': np.asarray(train_out['state'], np.float64),
                    'actions': np.asarray(acts_off, np.float64)})['actions'])[:16, :8]
                d = np.abs(off8 - acts_on)
                rows.append({'skill': row['skill'], 'episode': int(row['episode_index']),
                             'tick': t,
                             'offline_shape': list(off8.shape),
                             'online_shape': list(acts_on.shape),
                             'max_abs_8d': float(d.max()),
                             'mean_abs_8d': float(d.mean()),
                             'gripper_max_abs': float(d[:, 7].max()),
                             'policy_metadata': json.dumps(policy.metadata)[:200]})
    (OUT / 'policy_inference_parity.json').write_text(json.dumps(rows, ensure_ascii=False,
                                                                 indent=2), encoding='utf-8')
    import csv
    with open(OUT / 'policy_inference_parity.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r['skill'], r['episode'], r['tick'], 'max_abs', round(r['max_abs_8d'], 10),
              'mean', round(r['mean_abs_8d'], 10), 'grip', round(r['gripper_max_abs'], 10))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
