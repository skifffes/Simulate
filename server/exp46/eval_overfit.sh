#!/usr/bin/env bash
# 实验 4.6：过拟合模型的离线逐帧评测（合并 → 查询 → 分析 → 清理）
# 用法：bash eval_overfit.sh <STEP>
set -u
STEP=${1:?step}
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
LORA=/scratch/lijh/exp46/run/ckpts/lora_step$(printf '%05d' "$STEP").npz
MERGED=/scratch/lijh/exp46/merged/step${STEP}
OUT=results/experiment_4_6/offline
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=2
mkdir -p "$OUT" /scratch/lijh/exp46/merged
[ -f "$LORA" ] || { echo "missing $LORA"; exit 2; }
rm -rf "$MERGED"
"$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py --lora "$LORA" \
  --out "$MERGED/params" 2>&1 | tail -2
ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$MERGED/assets"
"$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py --ckpt "$MERGED" \
  --picks results/experiment_4_6/overfit_picks.json --seeds 0,1 \
  --out "$OUT/step${STEP}_val.npz" 2>&1 | tail -2
"$SIM" scripts/exp44/analyze_val.py --npz "$OUT/step${STEP}_val.npz" \
  --out "$OUT/step${STEP}_stats.json" 2>&1 | tail -3
"$SIM" scripts/exp42/full_chunk_eval.py --npz "$OUT/step${STEP}_val.npz" \
  --out "$OUT/step${STEP}_chunk.json" 2>&1 | tail -3
rm -rf "$MERGED"
echo "done $STEP"
