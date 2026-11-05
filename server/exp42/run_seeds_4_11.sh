#!/usr/bin/env bash
# 用 seed 4..11 复测 1800 与 2400：判断 2400 的退化是系统性还是"4 个种子不巧"
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c
SEEDS=${SEEDS:-4,5,6,7,8,9,10,11}

for pair in "1800:/scratch/lijh/exp42/curve1800/ckpt_1800" "2400:/scratch/lijh/exp42/curve2400/ckpt_2400"; do
  STEP=${pair%%:*}; CKPT=${pair#*:}
  echo "=== $STEP seeds $SEEDS ==="
  CUDA_VISIBLE_DEVICES=0 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
    --ckpt "$CKPT" --picks "$OUT/expert_state_picks_v2.json" --seeds "$SEEDS" \
    --out "$OUT/expert_state_curve${STEP}_s4.json.npz" 2>&1 | tail -1
  "$SIM" scripts/exp42/expert_state_analysis.py --npz "$OUT/expert_state_curve${STEP}_s4.json.npz" \
    --out "$OUT/expert_state_analysis_curve${STEP}_seeds4_11.json" 2>&1 | tail -8
done
