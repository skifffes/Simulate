#!/usr/bin/env bash
# A90 原始分布评测（1800 步模型），与 B90 同口径
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
CKPT=/scratch/lijh/exp42/curve1800/ckpt_1800
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c
WORK=/scratch/lijh/exp42/dist_eval

for SPLIT in train test; do
  if [ "$SPLIT" = "train" ]; then STRIDE=20; else STRIDE=10; fi
  echo "=== A90/$SPLIT ==="
  "$SIM" scripts/exp42/make_dist_picks.py --skill A90 --split "$SPLIT" --stride "$STRIDE" \
    --max-frames 500 --out "$WORK/picks_A90_${SPLIT}.json" 2>&1 | tail -1
  CUDA_VISIBLE_DEVICES=0 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
    --ckpt "$CKPT" --picks "$WORK/picks_A90_${SPLIT}.json" --seeds 0 \
    --out "$OUT/dist_A90_${SPLIT}_step1800.npz" 2>&1 | tail -1
  "$SIM" scripts/exp42/dist_eval.py --npz "$OUT/dist_A90_${SPLIT}_step1800.npz" \
    --tag "A90/${SPLIT}/step1800" --out "$OUT/dist_eval_A90_${SPLIT}_step1800.json" 2>&1 | tail -9
done
echo '=== done ==='
