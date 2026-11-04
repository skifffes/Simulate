#!/usr/bin/env bash
# curve600 的 300/600 步 checkpoint：合并 + 在 picks_v2 上推理 + 分析（GPU1）
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c
CURVE=/scratch/lijh/exp42/curve600

for STEP in 300 600; do
  echo "=== step $STEP: merge ==="
  CUDA_VISIBLE_DEVICES=1 "$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py \
    --lora "$CURVE/ckpts/lora_step00$STEP.npz" --out "$CURVE/ckpt_$STEP/params" 2>&1 | tail -2
  ln -sfn /scratch/lijh/exp42/run2/ckpts/params_nested/assets "$CURVE/ckpt_$STEP/assets"
  echo "=== step $STEP: infer ==="
  CUDA_VISIBLE_DEVICES=1 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
    --ckpt "$CURVE/ckpt_$STEP" --picks "$OUT/expert_state_picks_v2.json" --seeds 0,1,2,3 \
    --out "$OUT/expert_state_curve${STEP}.npz" 2>&1 | tail -2
  "$SIM" scripts/exp42/expert_state_analysis.py --npz "$OUT/expert_state_curve${STEP}.npz" \
    --out "$OUT/expert_state_analysis_curve${STEP}.json" 2>&1 | tail -8
done
echo '=== done ==='
