#!/usr/bin/env bash
# 1) 用 simulate 环境把一条 A90 episode 导出成 npz；2) 在 openpi 环境跑 transforms 检查
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
PY=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$REPO/src:$REPO"
"$PY" - <<'EOF'
import numpy as np, pandas as pd, sys
sys.path.insert(0, 'src')
from exp41.training_export import build_training_arrays
df = pd.read_parquet('data/exp41/A90/data/chunk-000/episode_000000.parquet',
                     columns=['observation.state', 'action'])
S = np.stack(df['observation.state'].to_numpy()).astype(np.float64)
A = np.stack(df['action'].to_numpy()).astype(np.float64)
a = build_training_arrays(S, A)
np.savez('/tmp/a90_export.npz', state=a['state'], action=a['action'])
print('exported', a['state'].shape, a['action'].shape)
EOF
export PYTHONPATH="$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
"$OAPI/.venv/bin/python" scripts/exp41/check_openpi_transforms.py --npz /tmp/a90_export.npz \
  --out results/experiment_4_1/preflight 2>&1 | tail -6
