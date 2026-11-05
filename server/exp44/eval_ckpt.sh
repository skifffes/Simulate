#!/usr/bin/env bash
# 实验 4.4：单个 LoRA checkpoint 的离线评测流水线（合并 → 查询 → 分析 → 清理）
# 用法：bash eval_ckpt.sh <GROUP> <STEP> [SEEDS]
#   环境变量 LORA_OVERRIDE 指定 LoRA npz（冒烟测试用）；EVAL_GPU 指定评测 GPU（默认 2）
set -u
GROUP=${1:?group A|B}
STEP=${2:?step}
SEEDS=${3:-0,1,2,3}
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
LORA=${LORA_OVERRIDE:-/scratch/lijh/exp44/$GROUP/ckpts/lora_step$(printf '%05d' "$STEP").npz}
MERGED=/scratch/lijh/exp44/merged/${GROUP}_step${STEP}
OUT=results/experiment_4_4/offline
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-2}
mkdir -p "$OUT"

[ -f "$LORA" ] || { echo "missing lora: $LORA"; exit 2; }
echo "=== merge $GROUP step $STEP ($LORA) on GPU $CUDA_VISIBLE_DEVICES ==="
rm -rf "$MERGED"
"$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py \
  --lora "$LORA" --out "$MERGED/params" 2>&1 | tail -3
ln -sfn /scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$MERGED/assets"

echo "=== query $GROUP step $STEP seeds=$SEEDS ==="
"$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
  --ckpt "$MERGED" --picks results/experiment_4_4/val_picks.json --seeds "$SEEDS" \
  --out "$OUT/${GROUP}_step${STEP}_val.npz" 2>&1 | tail -2

echo "=== analysis ==="
"$SIM" scripts/exp42/expert_state_analysis.py --npz "$OUT/${GROUP}_step${STEP}_val.npz" \
  --out "$OUT/${GROUP}_step${STEP}_val_analysis.json" 2>&1 | tail -3
"$SIM" scripts/exp42/full_chunk_eval.py --npz "$OUT/${GROUP}_step${STEP}_val.npz" \
  --out "$OUT/${GROUP}_step${STEP}_chunk.json" 2>&1 | tail -3

echo "=== cleanup merged (keep npz) ==="
rm -rf "$MERGED"
echo "done $GROUP $STEP"
