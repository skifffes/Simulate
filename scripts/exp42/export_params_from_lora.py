"""从 LoRA-only npz 重建**完整** params（基座 + LoRA 合并），存成 orbax OCDBT 供推理服务加载。

训练脚本本身也会导出完整 params；本脚本用于：
  * 训练进程已退出但只留下 LoRA npz 时；
  * 需要在另一个 GPU 上快速生成可部署 checkpoint 时（不用重跑训练）。

用法（openpi venv）：
  python scripts/exp42/export_params_from_lora.py \
      --lora /scratch/lijh/exp42/run/ckpts/lora_step00501.npz \
      --out /scratch/lijh/exp42/run/ckpts/step00501/params
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
    ap.add_argument('--lora', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import jax
    import orbax.checkpoint as ocp

    import openpi.training.config as _config

    from exp42 import registry

    name = registry.register()
    cfg = _config.get_config(name)

    print('building model ...', flush=True)
    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    print('loading base params ...', flush=True)
    state.replace_by_pure_dict(cfg.weight_loader.load(state.to_pure_dict()))
    model = nnx.merge(graphdef, state)

    print('merging LoRA ...', flush=True)
    z = np.load(args.lora)
    flat_lora = {tuple(k.split('__')): z[k] for k in z.files if k != 'step'}
    state = nnx.state(model)
    # 注意：flatten_dict 默认用 **tuple** 作为 key；带 sep 时 key 是字符串，不能混用
    flat_full = traverse_util.flatten_dict(state.to_pure_dict())
    n_applied = 0
    for key, val in flat_lora.items():
        if key in flat_full:
            flat_full[key] = np.asarray(val, dtype=np.float32)
            n_applied += 1
        else:
            print('  WARN missing path', key)
    print(f'  lora tensors {len(flat_lora)}, applied {n_applied}', flush=True)
    if n_applied == 0:
        raise SystemExit('ERROR: 没有匹配到任何 LoRA 张量，拒绝保存（否则会得到未微调的 checkpoint）')
    merged = traverse_util.unflatten_dict(flat_full)
    out = Path(args.out)
    if out.exists():
        import shutil

        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print('saving orbax checkpoint ->', out, flush=True)
    # 官方 checkpoint 的顶层是 {"params": <tree>}，restore_params 会读 metadata['params']，
    # 因此这里必须嵌套一层，否则推理服务加载时报 KeyError: 'params'。
    ocp.PyTreeCheckpointer().save(str(out), args=ocp.args.PyTreeSave(
        {'params': jax.tree.map(lambda x: np.asarray(x), merged)}))
    print('done; size', sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e9, 'GB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
