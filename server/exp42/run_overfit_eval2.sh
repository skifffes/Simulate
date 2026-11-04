#!/usr/bin/env bash
# 过拟合模型的采样评估 + B90 主动推动帧检查
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c
OF=/scratch/lijh/exp42/overfit/lr4

echo '=== push 阶段检查 ==='
"$SIM" /scratch/lijh/exp42/check_push_phase.py

echo '=== overfit: 采样评估 ==='
ln -sfn /scratch/lijh/exp42/run2/ckpts/params_nested/assets "$OF/assets" 2>/dev/null || true
mkdir -p "$OF/ckpt_eval"
ln -sfn "$OF/params_step00150" "$OF/ckpt_eval/params"
ln -sfn "$OF/assets" "$OF/ckpt_eval/assets"
CUDA_VISIBLE_DEVICES=0 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
  --ckpt "$OF/ckpt_eval" --picks "$OF/overfit_picks.json" --seeds 0 \
  --out "$OUT/overfit_eval.npz" 2>&1 | tail -2

"$SIM" scripts/exp42/full_chunk_eval.py --npz "$OUT/overfit_eval.npz" --labels anchor \
  --out "$OF/overfit_eval_chunk.json" 2>&1 | tail -3
