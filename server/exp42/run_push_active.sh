#!/usr/bin/env bash
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c

"$SIM" /scratch/lijh/exp42/make_push_active_picks.py
CUDA_VISIBLE_DEVICES=1 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
  --ckpt /scratch/lijh/exp42/run2/ckpts/params_nested \
  --picks "$OUT/expert_state_picks_push_active.json" --seeds 0,1,2,3 \
  --out "$OUT/expert_state_push_active.npz" 2>&1 | tail -3
"$SIM" scripts/exp42/expert_state_analysis.py --npz "$OUT/expert_state_push_active.npz" \
  --out "$OUT/expert_state_analysis_push_active.json" 2>&1 | tail -4
