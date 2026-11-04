#!/usr/bin/env bash
# 并行两件事：
#   GPU0: 合并 overfit 参数 -> 在 8 个过拟合窗口上采样 -> 整段误差
#   GPU1: 用 picks_v2（B90 四阶段拆分 + 去重）重新评测 step901
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="/scratch/lijh/exp40/lerobot_stub:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache
OUT=results/experiment_4_2/contract_replay_p0c
OF=/scratch/lijh/exp42/overfit/lr4

# --- GPU0: overfit 参数合并 + 采样 ---
(
  CUDA_VISIBLE_DEVICES=0 "$OAPI/.venv/bin/python" scripts/exp42/export_params_from_lora.py \
    --lora "$OF/ckpts/lora_step00150.npz" --out "$OF/ckpt_merged/params" 2>&1 | tail -2
  ln -sfn /scratch/lijh/exp42/run2/ckpts/params_nested/assets "$OF/ckpt_merged/assets"
  CUDA_VISIBLE_DEVICES=0 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
    --ckpt "$OF/ckpt_merged" --picks "$OF/overfit_picks.json" --seeds 0 \
    --out "$OUT/overfit_eval.npz" 2>&1 | tail -2
) > /tmp/gpu0_overfit.log 2>&1 &

# --- GPU1: step901 on picks_v2 ---
(
  CUDA_VISIBLE_DEVICES=1 "$OAPI/.venv/bin/python" scripts/exp42/expert_state_query.py \
    --ckpt /scratch/lijh/exp42/run2/ckpts/params_nested \
    --picks "$OUT/expert_state_picks_v2.json" --seeds 0,1,2,3 \
    --out "$OUT/expert_state_step901_v2.npz" 2>&1 | tail -2
) > /tmp/gpu1_step901.log 2>&1 &

wait
echo '=== gpu0 ==='; tail -4 /tmp/gpu0_overfit.log
echo '=== gpu1 ==='; tail -2 /tmp/gpu1_step901.log

# --- 分析 ---
CUDA_VISIBLE_DEVICES="" "$SIM" scripts/exp42/full_chunk_eval.py \
  --npz "$OUT/overfit_eval.npz" --labels anchor --out "$OF/overfit_eval_chunk.json" 2>&1 | tail -3
CUDA_VISIBLE_DEVICES="" "$SIM" scripts/exp42/expert_state_analysis.py \
  --npz "$OUT/expert_state_step901_v2.npz" --out "$OUT/expert_state_analysis_step901_v2.json" 2>&1 | tail -12
