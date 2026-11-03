"""Read-only transform/checkpoint manifest; optional frozen model probes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_manifest(tree, prefix=''):
    import numpy as np
    if isinstance(tree, dict):
        out = {}
        for k, value in tree.items():
            out.update(tensor_manifest(value, prefix + '/' + str(k)))
        return out
    a = np.asarray(tree)
    return {prefix: {'shape': list(a.shape), 'dtype': str(a.dtype),
                     'sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                     'min': float(a.min()) if a.size and a.dtype.kind in 'fiu' else None,
                     'max': float(a.max()) if a.size and a.dtype.kind in 'fiu' else None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=Path('/scratch/lijh/exp40/repo'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    sys.path[:0] = [str(args.root), str(args.root / 'src')]
    import numpy as np
    report = {'python': sys.executable, 'checkpoints': [], 'errors': []}
    base = Path('/scratch/lijh/exp42')
    report['params_directories'] = []
    candidates = []
    for parent, dirs, files in os.walk(base):
        depth = len(Path(parent).relative_to(base).parts)
        dirs[:] = [d for d in dirs if d not in {'data', 'params', 'independent_audit_raw'} and depth < 5]
        if 'params' in os.listdir(parent):
            report['params_directories'].append(str(Path(parent) / 'params'))
        candidates.extend(Path(parent) / name for name in files if name.startswith('lora') and name.endswith('.npz'))
    for path in candidates:
        item = {'path': str(path), 'bytes': path.stat().st_size}
        report['checkpoints'].append(item)
    (args.out / 'model_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'phase': 'checkpoint_paths', 'paths': report['checkpoints']}), flush=True)
    for rel in ['models/pi0.py', 'policies/droid_policy.py', 'policies/policy.py',
                'policies/policy_config.py', 'transforms.py']:
        p = Path('/scratch/lijh/exp40/openpi/src/openpi') / rel
        report.setdefault('openpi_sources', {})[rel] = sha(p)
    git = Path('/scratch/lijh/exp40/openpi/.git')
    report['openpi_head_file'] = (git / 'HEAD').read_text().strip()
    ref = report['openpi_head_file'].removeprefix('ref: ')
    report['openpi_ref_value'] = (git / ref).read_text().strip() if (git / ref).exists() else None
    try:
        from scripts.exp42 import train_lora as tl
        import openpi.transforms as tr
        cfg = tl.build_config(batch_size=1, num_steps=1)
        transform, dc = tl.make_transforms(cfg)
        report['use_quantile_norm'] = bool(dc.use_quantile_norm)
        report['norm_stats'] = tensor_manifest({k: vars(v) for k, v in dc.norm_stats.items()})
        meta = tl.load_meta()
        cache = tl.EpisodeCache(limit=1)
        for skill in ['A90', 'B90']:
            row = next(r for r in meta if r['skill'] == skill and r['episode_index'] == 0)
            ep = cache.get(row)
            for t in [0, 4, 8, 20, 40, 80]:
                if t + 16 > len(ep['state']):
                    continue
                raw = tl.make_item(row, ep, t)
                raw.pop('_valid_mask')
                input_only = dict(raw)
                input_only.pop('actions')
                train_item = transform(dict(raw))
                infer_item = transform(input_only)
                key = f'{skill}/ep0/t{t}'
                report.setdefault('transform_samples', {})[key] = {
                    'raw': tensor_manifest(raw), 'train': tensor_manifest(train_item),
                    'inference': tensor_manifest(infer_item),
                    'common_tensors_equal': all(np.array_equal(train_item[k], infer_item[k])
                        for k in infer_item if not isinstance(infer_item[k], dict)),
                    'images_equal': all(np.array_equal(train_item['image'][k], infer_item['image'][k])
                        for k in infer_item.get('image', {})),
                    'raw_actions': np.asarray(raw['actions']).tolist()}
                before_norm = tr.compose(dc.data_transforms.inputs)(dict(raw))
                normalized = tr.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)(dict(before_norm))
                padded = dict(normalized)
                for field, stats in dc.norm_stats.items():
                    value = padded[field]
                    width = len(stats.mean)
                    if value.shape[-1] < width:
                        padded[field] = np.pad(value, [(0, 0)] * (value.ndim - 1) + [(0, width - value.shape[-1])])
                restored = tr.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)(padded)
                report['transform_samples'][key]['normalization_roundtrip_max_error'] = float(
                    np.max(np.abs(np.asarray(restored['actions'])[..., :8] - before_norm['actions'])))
                report['transform_samples'][key]['normalized_effective_action_min'] = float(np.min(normalized['actions']))
                report['transform_samples'][key]['normalized_effective_action_max'] = float(np.max(normalized['actions']))
    except Exception as exc:
        import traceback
        report['errors'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
    (args.out / 'model_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'checkpoints': len(report['checkpoints']), 'params_directories': report['params_directories'],
                      'errors': report['errors'], 'samples': len(report.get('transform_samples', {}))}))


if __name__ == '__main__':
    main()
