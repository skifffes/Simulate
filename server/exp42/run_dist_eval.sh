#!/usr/bin/env bash
# 用"原始分布"快速评测当前最优模型（1800 步）
#   B90 train: stride 20；B90 test: stride 10（未见分布）；各 1 个 seed
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
mkdir -p "$WORK"

for SPLIT in train test; do
  if [ "$SPLIT" = "train" ]; then STRIDE=20; else STRIDE=10; fi
  echo "=== B90/$SPLIT picks (stride $STRIDE) ==="
  "$SIM" scripts/exp42/make_dist_picks.py --skill B90 --split "$SPLIT" --stride "$STRIDE" \
    --max-frames 500 --out "$WORK/picks_B90_${SPLIT}.json" 2>&1 | tail -2
  echo "=== B90/$SPLIT infer ==="
  CUDA_VISIBLE_DEVICES=0 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
    --ckpt "$CKPT" --picks "$WORK/picks_B90_${SPLIT}.json" --seeds 0 \
    --out "$OUT/dist_B90_${SPLIT}_step1800.npz" 2>&1 | tail -1
  echo "=== B90/$SPLIT eval ==="
  "$SIM" scripts/exp42/dist_eval.py --npz "$OUT/dist_B90_${SPLIT}_step1800.npz" \
    --tag "B90/${SPLIT}/step1800" --out "$OUT/dist_eval_B90_${SPLIT}_step1800.json" 2>&1 | tail -10
done
echo '=== done ==='
