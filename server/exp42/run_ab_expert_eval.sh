#!/usr/bin/env bash
# A/B 在专家状态上的离线诊断：查询 + 分析 + 对照
set -eu
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO"
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c

for G in A B; do
  ln -sfn /scratch/lijh/exp42/run2/ckpts/params_nested/assets "/scratch/lijh/exp42/ab_repr/$G/ckpt_merged/assets"
done

for G in A B; do
  GPU=$([ "$G" = "A" ] && echo 0 || echo 1)
  echo "=== query $G (gpu$GPU) ==="
  CUDA_VISIBLE_DEVICES=$GPU "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
    --ckpt "/scratch/lijh/exp42/ab_repr/$G/ckpt_merged" \
    --picks "$OUT/expert_state_picks.json" --seeds 0,1,2,3 \
    --out "$OUT/expert_state_$G.npz" 2>&1 | tail -2
done

for G in A B; do
  echo "=== analyse $G ==="
  "$SIM" scripts/exp42/expert_state_analysis.py --npz "$OUT/expert_state_$G.npz" \
    --out "$OUT/expert_state_analysis_$G.json" 2>&1 | tail -10
done

echo "=== compare ==="
"$SIM" scripts/exp42/ab_repr_compare.py --a "$OUT/expert_state_analysis_A.json" \
  --b "$OUT/expert_state_analysis_B.json" --out "$OUT/ab_repr_comparison.json" 2>&1 | tail -14
