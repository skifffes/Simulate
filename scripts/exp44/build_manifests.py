"""实验 4.4：生成 data_manifest.json / training_configs/{A,B}.json / checkpoint_manifest.json。

读取：
  * /scratch/lijh/exp43/export_v3（Group B）
  * /scratch/lijh/exp44_export_A58（Group A）
  * /scratch/lijh/exp44/{A,B}/ckpts（LoRA 与 opt state）
输出：results/experiment_4_4/ 下的清单文件。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')
A58 = Path('/scratch/lijh/exp44_export_A58')
B311 = Path('/scratch/lijh/exp43/export_v3')
CKPT = {'A': Path('/scratch/lijh/exp44/A/ckpts'), 'B': Path('/scratch/lijh/exp44/B/ckpts')}


def sha256(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        n = 0
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
            n += len(chunk)
            if limit and n > limit:
                break
    return h.hexdigest()


def data_summary(root: Path) -> dict:
    meta = [json.loads(l) for l in (root / 'meta.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    train = [m for m in meta if m.get('split') == 'train']
    by = Counter((m['skill'], m.get('source', 'exp41')) for m in train)
    n_frames = sum(int(m.get('n_frames') or 0) for m in train)
    return {
        'root': str(root),
        'n_train': len(train),
        'n_by_skill': dict(Counter(m['skill'] for m in train)),
        'n_by_skill_source': {f'{k[0]}/{k[1]}': v for k, v in by.items()},
        'n_val': sum(1 for m in meta if m.get('split') == 'val'),
        'n_test': sum(1 for m in meta if m.get('split') == 'test'),
        'train_frames': n_frames,
        'meta_sha256': sha256(root / 'meta.jsonl'),
        'episode_indices': sorted({int(m['episode_index']) for m in train})[:5],
        'note': 'train windows 由训练脚本统计（anchor-16 切片）',
    }


def main() -> int:
    L.mkdir(parents=True, exist_ok=True)
    data = {'A': data_summary(A58), 'B': data_summary(B311)}
    data['B']['train_windows'] = 12967
    data['A']['train_windows'] = 2404
    (L / 'data_manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                          encoding='utf-8')

    base_cfg = {
        'base': 'pi05_droid (checkpoints/pi05_droid/params)',
        'openpi_commit': '215abfb217dbac7d5f1273282331b9b1866c0479',
        'model': 'pi05=True, action_dim=32, action_horizon=16, paligemma=gemma_2b_lora, '
                 'action_expert=gemma_300m_lora',
        'optimizer': 'openpi training.optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule)',
        'lr_schedule': {'type': 'CosineDecaySchedule', 'warmup_steps': 1000,
                        'peak_lr': 2.5e-05, 'decay_steps': 30000, 'decay_lr': 2.5e-06},
        'batch_size': 4, 'accum': 1, 'steps': 12000, 'seed': 0,
        'label_mode': 'anchor', 'anchor_reference': 'fixed', 'horizon': 16, 'loss_dims': 0,
        'normalization': 'official droid asset (unchanged)',
        'save_lora_at': [1200, 1800, 2400, 3600, 4800, 6000, 8000, 10000, 12000],
        'save_opt_at': [6000, 12000],
        'eval': 'exp4.1-strict-v1.2',
    }
    cfg = {}
    for g in ('A', 'B'):
        cfg[g] = dict(base_cfg, group=g, data_root=str(A58 if g == 'A' else B311),
                      train_episodes=data[g]['n_train'],
                      train_windows=data[g]['train_windows'],
                      out_dir=str(CKPT[g].parent))
    (L / 'training_configs').mkdir(exist_ok=True)
    for g in ('A', 'B'):
        (L / 'training_configs' / f'{g}.json').write_text(
            json.dumps(cfg[g], ensure_ascii=False, indent=2), encoding='utf-8')

    # checkpoint manifest（当前已有）
    ck = {}
    for g in ('A', 'B'):
        d = CKPT[g]
        items = []
        for p in sorted(d.glob('lora_step*.npz')):
            items.append({'step': int(p.stem.replace('lora_step', '')),
                          'path': str(p), 'bytes': p.stat().st_size,
                          'sha256': sha256(p)})
        opts = [{'step': int(p.stem.replace('opt_step', '')),
                 'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                for p in sorted(d.glob('opt_step*.npz'))]
        ck[g] = {'lora': items, 'opt_state': opts}
    (L / 'checkpoint_manifest.json').write_text(json.dumps(ck, ensure_ascii=False, indent=2),
                                                encoding='utf-8')
    print(json.dumps({'data': {g: {k: v for k, v in data[g].items()
                                    if k in ('n_train', 'n_by_skill', 'n_val', 'n_test',
                                             'train_windows')} for g in ('A', 'B')},
                      'ckpt_lora': {g: [i['step'] for i in ck[g]['lora']] for g in ('A', 'B')},
                      'ckpt_opt': {g: [i['step'] for i in ck[g]['opt_state']]
                                   for g in ('A', 'B')}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
