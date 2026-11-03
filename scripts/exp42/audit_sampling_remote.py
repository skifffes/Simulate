"""Frozen checkpoint probes with paired explicit noise; never updates parameters."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--lora', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    root = Path('/scratch/lijh/exp40/repo')
    sys.path[:0] = [str(root), str(root / 'src')]
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flax.traverse_util import flatten_dict
    from scripts.exp42 import train_lora as tl
    from src.exp42 import registry
    from openpi.policies import policy as policy_lib
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.models import model as model_lib
    from openpi.training import config as config_lib
    cfg = config_lib.get_config(registry.register())
    report = {'checkpoint': str(args.checkpoint), 'npz': str(args.lora),
              'npz_sha256': digest(args.lora), 'seeds': [17, 29, 43, 71],
              'sampling_steps': 10, 'phase_rule': 'Frame0 startup; frame8 approach; 1/2 and 3/4 trajectory are temporal strata, NOT verified physical contact.',
              'started': time.time(), 'errors': []}
    path = args.out / 'sampling_manifest.json'
    path.write_text(json.dumps(report, indent=2))
    print('loading checkpoint', flush=True)
    params = model_lib.restore_params(args.checkpoint / 'params', dtype=jnp.bfloat16)
    model = cfg.model.load(params)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_path = args.checkpoint / 'assets/droid/norm_stats.json'
    if norm_path.exists():
        norm = normalize.load(norm_path.parent)
        report['norm_source'] = str(norm_path)
        report['norm_asset_sha256'] = digest(norm_path)
        report['norm_matches_registered'] = all(
            np.array_equal(getattr(value, field), getattr(data_cfg.norm_stats[key], field))
            for key, value in norm.items() for field in ['mean', 'std', 'q01', 'q99'])
        if not report['norm_matches_registered']:
            raise RuntimeError('Checkpoint normalization differs from registered training statistics')
    else:
        raise FileNotFoundError(f'Required DROID normalization asset absent: {norm_path}')
    path.write_text(json.dumps(report, indent=2))
    policy = policy_lib.Policy(model,
        transforms=[*data_cfg.data_transforms.inputs,
                    transforms.Normalize(norm, use_quantiles=data_cfg.use_quantile_norm),
                    *data_cfg.model_transforms.inputs],
        output_transforms=[*data_cfg.model_transforms.outputs,
                           transforms.Unnormalize(norm, use_quantiles=data_cfg.use_quantile_norm),
                           *data_cfg.data_transforms.outputs],
        sample_kwargs={'num_steps': 10})
    state = nnx.state(policy._model)
    trainable = flatten_dict(state.filter(cfg.trainable_filter).to_pure_dict())
    full = flatten_dict(state.to_pure_dict())
    coverage = {'expected': len(trainable), 'matched': 0, 'missing': [], 'unexpected': [], 'mismatches': [], 'keys': []}
    with np.load(args.lora, allow_pickle=False) as z:
        keys = {tuple(k.split('__')): k for k in z.files if k != 'step'}
        coverage['step'] = int(z['step']) if 'step' in z else None
        coverage['missing'] = ['/'.join(k) for k in trainable.keys() - keys.keys()]
        coverage['unexpected'] = ['/'.join(k) for k in keys.keys() - trainable.keys()]
        for k, name in keys.items():
            a = z[name]
            b = np.asarray(full[k]) if k in full else None
            equal = b is not None and a.shape == b.shape and np.array_equal(a.astype(b.dtype), b)
            coverage['keys'].append({'key': name, 'shape': list(a.shape), 'npz_dtype': str(a.dtype),
                'restored_dtype': str(b.dtype) if b is not None else None,
                'finite': bool(np.isfinite(a).all()), 'equal_after_restored_dtype_cast': bool(equal),
                'sha256': hashlib.sha256(a.tobytes()).hexdigest()})
            coverage['matched'] += int(equal)
            if not equal:
                coverage['mismatches'].append(name)
    report['coverage'] = coverage
    path.write_text(json.dumps(report, indent=2))
    print('coverage', coverage['expected'], coverage['matched'], 'missing', len(coverage['missing']), 'mismatch', len(coverage['mismatches']), flush=True)
    if coverage['missing'] or coverage['unexpected'] or coverage['mismatches']:
        report['errors'].append('Parameter coverage failed; sampling stopped to avoid ambiguous checkpoint identity.')
        path.write_text(json.dumps(report, indent=2))
        return
    metadata = tl.load_meta()
    cache = tl.EpisodeCache(limit=2)
    records = []
    arrays = {}
    @nnx.jit
    def frozen_flow(m, rng, observation, actions):
        return tl.flow_sq(m, rng, observation, actions, train=False, horizon=16)
    train_transform, _ = tl.make_transforms(cfg)
    for skill in ['A90', 'B90']:
        row = next(r for r in metadata if r['skill'] == skill and r['split'] == 'val')
        ep = cache.get(row)
        length = len(ep['state'])
        for phase, t in [('startup', 0), ('approach', 8), ('middle_temporal', length // 2), ('late_temporal', length * 3 // 4)]:
            t = min(t, length - 16)
            raw = tl.make_item(row, ep, t)
            target = raw['actions'][:, :8]
            training_input = dict(raw)
            training_input.pop('_valid_mask')
            transformed = train_transform(training_input)
            batch = jax.tree.map(lambda x: jnp.asarray(x)[None], transformed)
            fm_observation = model_lib.Observation.from_dict(batch)
            fm_actions = batch['actions']
            fm_records = []
            for fm_seed in report['seeds']:
                sq = np.asarray(frozen_flow(model, jax.random.key(fm_seed), fm_observation, fm_actions))
                fm_records.append({'seed': fm_seed, 'joint7': float(sq[..., :7].mean()),
                    'gripper': float(sq[..., 7].mean()), 'effective8': float(sq[..., :8].mean()),
                    'padding24': float(sq[..., 8:].mean()), 'all32': float(sq.mean())})
            report.setdefault('flow_matching', {})[f'{skill}/{phase}/t{t}'] = fm_records
            obs = dict(raw)
            obs.pop('actions')
            obs.pop('_valid_mask')
            obs_hash = hashlib.sha256(b''.join(np.asarray(obs[k]).tobytes() for k in sorted(obs))).hexdigest()
            predictions = []
            for seed in report['seeds']:
                noise = np.asarray(jax.random.normal(jax.random.key(seed), (16, 32), dtype=jnp.float32))
                prediction = np.asarray(policy.infer(copy.deepcopy(obs), noise=noise)['actions'])
                if not predictions:
                    repeat = np.asarray(policy.infer(copy.deepcopy(obs), noise=noise)['actions'])
                    if not np.array_equal(prediction, repeat):
                        raise RuntimeError('Fixed input/noise repeat is not deterministic')
                predictions.append(prediction)
                ref, pred = target[0, :7], prediction[0, :7]
                norm = float(np.linalg.norm(ref))
                cosine = float(np.dot(ref, pred) / (norm * np.linalg.norm(pred))) if norm > 1e-4 and np.linalg.norm(pred) > 1e-8 else None
                records.append({'skill': skill, 'episode': row['episode_index'], 'split': row['split'],
                    'phase': phase, 'frame': t, 'seed': seed, 'observation_sha256': obs_hash,
                    'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(),
                    'first_joint_mae_rad': float(np.abs(pred-ref).mean()),
                    'chunk_joint_mae_rad': float(np.abs(prediction[:, :7]-target[:, :7]).mean()),
                    'first_cosine': cosine, 'first_norm_ratio': float(np.linalg.norm(pred)/norm) if norm > 1e-4 else None,
                    'gripper_mae_normalized': float(np.abs(prediction[:, 7]-target[:, 7]).mean())})
            key = f'{skill}_{phase}_t{t}'
            predictions = np.stack(predictions)
            arrays[key + '_prediction'] = predictions
            arrays[key + '_target'] = target
            report.setdefault('aggregates', {})[key] = {
                'mean_chunk_joint_mae_rad': float(np.abs(predictions.mean(0)[:, :7]-target[:, :7]).mean()),
                'joint_sample_std_rad': float(predictions[:, :, :7].std(0).mean()),
                'target_first_norm_rad': float(np.linalg.norm(target[0, :7])),
                'predicted_first_norm_mean_rad': float(np.linalg.norm(predictions[:, 0, :7], axis=1).mean())}
            with (args.out / 'model_behavior_diagnostics.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(records[0]))
                w.writeheader()
                w.writerows(records)
            np.savez_compressed(args.out / 'frozen_samples.npz', **arrays)
            path.write_text(json.dumps(report, indent=2))
            print(key, report['aggregates'][key], flush=True)
    report['completed'] = time.time()
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
