"""实验 4.7 Gate 1 + Gate 2：训练路径 vs 推理路径的输入一致性 + Anchor-16 全行重建。

对 4 条固定 episode 的多个阶段帧：
  A. 训练路径：train_lora.make_item() → training transforms（含 Normalize）→ 模型输入
  B. 推理路径：serve 客户端 dict → Policy._input_transform → 模型输入
逐字段比较（shape/dtype/maxabs/meanabs/sha256），token/mask 要求精确一致。

然后 Gate 2：用训练标签（16×32，已归一化）→ Policy._output_transform（反归一化）→
ActionAdapter(fixed anchor) → 与原始专家 A[t:t+16] 比较，按 k 分区统计误差与裁剪。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/scratch/lijh/exp40/repo')
for p in (REPO / 'scripts' / 'exp42', REPO / 'src', REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import openpi.policies.policy_config as _policy_config  # noqa: E402
import train_lora as TL  # noqa: E402
from exp42 import registry  # noqa: E402

DATA = Path('/scratch/lijh/exp46/export_overfit')
OUT = REPO / 'results' / 'experiment_4_7'
FINGER_MAX_M = 0.04
HORIZON = 16
CKPT = Path('/scratch/lijh/exp47/merged/step3000')


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def cmp_field(name, a, b, exact=False):
    a, b = np.asarray(a), np.asarray(b)
    out = {'field': name, 'shape_a': list(a.shape), 'shape_b': list(b.shape),
           'dtype_a': str(a.dtype), 'dtype_b': str(b.dtype),
           'sha_a': sha(a), 'sha_b': sha(b)}
    if a.shape != b.shape:
        out.update({'max_abs': None, 'mean_abs': None, 'pass': False, 'note': 'shape mismatch'})
        return out
    if exact:
        out.update({'max_abs': 0 if np.array_equal(a, b) else 1,
                    'mean_abs': 0.0, 'pass': bool(np.array_equal(a, b))})
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        out.update({'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
                    'pass': bool(d.max() <= 1e-5)})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = registry.build_config()
    policy = _policy_config.create_trained_policy(cfg, str(CKPT))
    transform, data_cfg = TL.make_transforms(cfg)
    meta = [json.loads(l) for l in (DATA / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]

    rows = []
    roundtrip = []
    for row in meta:
        z = np.load(DATA / row['state_path'])
        top_arr = np.load(DATA / row['top_path'], mmap_mode='r')
        wrist_arr = np.load(DATA / row['wrist_path'], mmap_mode='r')
        ep = {'top': top_arr, 'wrist': wrist_arr,
              'state': z['state'].astype(np.float64),
              'action': z['action'].astype(np.float64)}
        S = z['state'].astype(np.float64)
        A = z['action'].astype(np.float64)
        T = len(S)
        # 阶段帧：均匀 + 关键事件（夹爪闭合、首次接触/推进）
        g = S[:, 7:9].mean(axis=1)
        close_t = int(np.argmax(g < 0.03)) if (g < 0.03).any() else T // 2
        frames = sorted({0, T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T - 17,
                         max(0, close_t - 1), close_t, min(T - 17, close_t + 5)})
        frames = [t for t in frames if 0 <= t <= T - 17]
        for t in frames:
            item = TL.make_item(row, ep, t, label_mode='anchor')
            train_out = transform(item)
            obs = {
                'observation/exterior_image_1_left': np.asarray(
                    np.load(DATA / row['top_path'], mmap_mode='r')[t], dtype=np.uint8),
                'observation/wrist_image_left': np.asarray(
                    np.load(DATA / row['wrist_path'], mmap_mode='r')[t], dtype=np.uint8),
                'observation/joint_position': S[t, :7].astype(np.float64),
                'observation/gripper_position': np.asarray(
                    [np.clip(S[t, 7:9].mean() / FINGER_MAX_M, 0, 1)], dtype=np.float64),
                'prompt': row['prompt'],
            }
            inf = policy._input_transform({k: v for k, v in obs.items()})

            fields = []
            for cam in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'):
                fields.append(cmp_field(f'image.{cam}', train_out['image'][cam],
                                        inf['image'][cam]))
                fields.append(cmp_field(f'image_mask.{cam}', train_out['image_mask'][cam],
                                        inf['image_mask'][cam], exact=True))
            fields.append(cmp_field('state', train_out['state'], inf['state']))
            fields.append(cmp_field('tokenized_prompt', train_out['tokenized_prompt'],
                                    inf['tokenized_prompt'], exact=True))
            fields.append(cmp_field('tokenized_prompt_mask', train_out['tokenized_prompt_mask'],
                                    inf['tokenized_prompt_mask'], exact=True))
            for f in fields:
                f.update({'skill': row['skill'], 'episode': int(row['episode_index']), 'tick': t})
            rows.extend(fields)

            # ---- Gate 2：标签 → 反归一化 → Adapter ----
            labels_n = np.asarray(train_out['actions'], dtype=np.float64)      # (16,32) 已归一化
            out = policy._output_transform({'state': np.asarray(inf['state'], np.float64),
                                            'actions': labels_n})
            acts = np.asarray(out['actions'], dtype=np.float64)[:HORIZON, :8]  # 反归一化 8D
            from exp40.action_adapter import ActionAdapterConfig, chunk_to_targets
            adapted = chunk_to_targets(acts, S[t, :7], np.array([-2.9] * 7), np.array([2.9] * 7),
                                       ActionAdapterConfig(gripper_mode='continuous'),
                                       anchor_reference='fixed')
            expert = A[t:t + HORIZON, :8]
            rec = {'skill': row['skill'], 'episode': int(row['episode_index']), 'tick': t,
                   'n_rows': len(adapted)}
            for k in range(min(HORIZON, len(adapted))):
                a = adapted[k]
                qt = np.asarray(a.q_target, dtype=np.float64)
                e = np.abs(qt - expert[k, :7])
                rec[f'k{k}_max'] = float(e.max())
                rec[f'k{k}_grip'] = float(abs(a.gripper_target_m - expert[k, 7]))
                rec[f'k{k}_accepted'] = bool(a.accepted)
                rec[f'k{k}_reasons'] = ';'.join(a.reasons) if a.reasons else ''
            roundtrip.append(rec)

    import csv
    with open(OUT / 'observation_contract.csv', 'w', newline='', encoding='utf-8') as fh:
        keys = ['skill', 'episode', 'tick', 'field', 'shape_a', 'shape_b', 'dtype_a', 'dtype_b',
                'max_abs', 'mean_abs', 'sha_a', 'sha_b', 'pass']
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with open(OUT / 'action_roundtrip.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(roundtrip[0].keys()))
        w.writeheader()
        w.writerows(roundtrip)

    # 汇总
    import collections
    ok = sum(1 for r in rows if r['pass'])
    print(f'contract fields: {ok}/{len(rows)} pass')
    bad = [r for r in rows if not r['pass']]
    for r in bad[:8]:
        print('  FAIL', r['skill'], r['episode'], r['tick'], r['field'], r['max_abs'])
    ks = {}
    for rec in roundtrip:
        for region, kk in (('k0', [0]), ('k1_7', range(1, 8)), ('k8_15', range(8, 16))):
            for k in kk:
                if f'k{k}_max' not in rec:
                    continue
                ks.setdefault(region, []).append((rec[f'k{k}_max'], rec[f'k{k}_grip'],
                                                  rec[f'k{k}_accepted']))
    summary = {}
    for region, vals in ks.items():
        arr = np.array([[v[0], v[1], 1 if v[2] else 0] for v in vals])
        summary[region] = {'n': len(vals), 'joint_max': float(arr[:, 0].max()),
                           'joint_mean': float(arr[:, 0].mean()),
                           'gripper_max': float(arr[:, 1].max()),
                           'accepted_frac': float(arr[:, 2].mean())}
    (OUT / 'adapter_clipping_audit.json').write_text(
        json.dumps({'regions': summary, 'n_frames': len(roundtrip),
                    'contract_fail_fields': len(bad),
                    'note': 'joint_max 为 |q_target - A[t+k]| 的最大值（含 Adapter 拒绝/裁剪）'},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
