"""Gate 1 证据：专家权重确实来自 pi05_droid checkpoint（不是随机初始化）。

做法：先用固定 RNG 创建**未加载**的模型，记录若干专家/视觉张量的统计量；
再加载 pi05_droid 权重，比较同一批张量。若加载生效，专家张量应显著不同于随机初始化。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/scratch/lijh/exp40/repo')
sys.path.insert(0, str(REPO / 'scripts' / 'exp45'))
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

import flax.nnx as nnx  # noqa: E402
import flax.traverse_util as traverse_util  # noqa: E402
import jax  # noqa: E402
from exp45_common import build_config_ae  # noqa: E402

CHECK = [
    'PaliGemma/llm/layers/mlp_1/gating_einsum',
    'PaliGemma/llm/layers/attn/q_einsum_1/w',
    'PaliGemma/llm/final_norm_1/Dense_0/kernel',
    'action_out_proj/kernel',
    'PaliGemma/llm/layers/mlp/gating_einsum',          # VLM（也应来自 checkpoint）
    'PaliGemma/llm/embedder/input_embedding',
]


def stats(params, path):
    flat = traverse_util.flatten_dict(params.to_pure_dict())
    key = tuple(path.split('/'))
    if key not in flat:
        return None
    a = np.asarray(jax.device_get(flat[key]), dtype=np.float32)
    return {'shape': list(a.shape), 'mean': float(a.mean()), 'std': float(a.std()),
            'absmax': float(np.abs(a).max())}


def main() -> int:
    cfg = build_config_ae()
    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    fresh = nnx.state(model)
    fresh_stats = {p: stats(fresh, p) for p in CHECK}
    loaded_dict = cfg.weight_loader.load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded_dict)
    loaded = nnx.state(nnx.merge(graphdef, state))
    loaded_stats = {p: stats(loaded, p) for p in CHECK}

    rows = []
    for p in CHECK:
        f, l = fresh_stats[p], loaded_stats[p]
        differ = f is not None and l is not None and f['std'] != l['std']
        rows.append({'path': p, 'fresh': f, 'loaded': l, 'changed_by_loading': differ})
    out = {'rows': rows, 'all_loaded_differ_from_fresh': all(r['changed_by_loading'] for r in rows)}
    Path('/scratch/lijh/exp45/gate1_weight_loading.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in rows:
        print(f"{r['path']:58s} fresh std={r['fresh']['std']:.5f} -> "
              f"loaded std={r['loaded']['std']:.5f} changed={r['changed_by_loading']}")
    print('all differ:', out['all_loaded_differ_from_fresh'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
