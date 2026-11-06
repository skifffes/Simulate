"""实验 4.5：把 Action Expert-only 的 trainable-params npz 合并进基座，导出可服务 checkpoint。

与 exp42/export_params_from_lora.py 逻辑相同，只是用 exp45 的**非 LoRA** 配置，
因此参数树里没有 lora 张量，npz 的 key 直接对应完整参数路径。

用法：
  python scripts/exp45/export_ae_params.py \
      --params /scratch/lijh/exp45/run/ckpts/lora_step01200.npz \
      --out /scratch/lijh/exp45/merged/step1200/params
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import jax
    import orbax.checkpoint as ocp

    from exp45 import registry

    name = registry.register()
    cfg = registry.build_config()
    print(f'building model {name} ...', flush=True)
    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(cfg.weight_loader.load(state.to_pure_dict()))
    model = nnx.merge(graphdef, state)

    z = np.load(args.params)
    flat_in = {tuple(k.split('__')): z[k] for k in z.files if k != 'step'}
    state = nnx.state(model)
    flat_full = traverse_util.flatten_dict(state.to_pure_dict())
    n_applied = 0
    for key, val in flat_in.items():
        if key in flat_full:
            flat_full[key] = np.asarray(val, dtype=np.float32)
            n_applied += 1
        else:
            print('  WARN missing path', key)
    print(f'  tensors {len(flat_in)}, applied {n_applied}', flush=True)
    if n_applied == 0:
        raise SystemExit('ERROR: 没有任何张量匹配，拒绝保存')
    merged = traverse_util.unflatten_dict(flat_full)
    out = Path(args.out)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print('saving orbax checkpoint ->', out, flush=True)
    ocp.PyTreeCheckpointer().save(str(out), args=ocp.args.PyTreeSave(
        {'params': jax.tree.map(lambda x: np.asarray(x), merged)}))
    print('done; size', round(sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e9, 2), 'GB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
