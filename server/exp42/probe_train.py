"""探测：openpi 训练侧依赖、缓存 checkpoint/assets、可用 config。"""
import os
import pathlib
import traceback

print('=== imports ===')
for mod in ('openpi.training.config', 'openpi.training.data_loader', 'openpi.models.pi0',
            'openpi.models.pi0_config', 'openpi.transforms', 'openpi.shared.normalize',
            'openpi.shared.download', 'openpi.training.weight_loaders', 'optax', 'orbax.checkpoint'):
    try:
        __import__(mod)
        print('OK  ', mod)
    except Exception as exc:  # noqa: BLE001
        print('FAIL', mod, '->', repr(exc)[:160])

print()
print('=== cache ===')
root = pathlib.Path(os.path.expanduser('~/.cache/openpi'))
for p in sorted(root.rglob('*')):
    if p.is_file() and ('pi05_droid' in str(p)):
        print(f'{p.stat().st_size/1e6:10.1f} MB  {p}')
    if p.is_dir() and p.name in ('pi05_droid', 'droid', 'params', 'assets'):
        print('DIR ', p)

print()
print('=== configs ===')
try:
    import openpi.training.config as cfg
    names = [c.name for c in cfg._CONFIGS]
    print('n configs', len(names))
    print([n for n in names if 'droid' in n or 'pi05' in n])
except Exception:
    traceback.print_exc()

print()
print('=== norm stats (droid) ===')
try:
    import openpi.shared.normalize as normalize
    for cand in (root / 'openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json',
                 root / 'assets/droid/norm_stats.json'):
        print(cand, cand.exists())
        if cand.exists():
            st = normalize.load(cand.parent)
            for k, v in st.items():
                if hasattr(v, 'q01'):
                    import numpy as np
                    print(' ', k, 'q01', np.asarray(v.q01).round(4)[:8], 'q99', np.asarray(v.q99).round(4)[:8])
                else:
                    print(' ', k, type(v))
except Exception:
    traceback.print_exc()
