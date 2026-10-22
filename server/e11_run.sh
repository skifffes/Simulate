#!/usr/bin/env bash
# 实验 1.1 通用评测启动器（由 batch 脚本调用，也可单独用）。
# 环境变量：MODE SS NA SEEDS BASE N OUT EXPERT VIDEOS SHIFT_X SHIFT_Y YAW FRICTION GPU
set -euo pipefail
cd "$HOME/simulate_vla"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HUB_OFFLINE=1
mkdir -p results/experiment_1_1 logs

out="${OUT:?OUT 必须指定}"
args=(
  --policy outputs/smolvla_panda/checkpoints/015000/pretrained_model
  --mode "${MODE:-strict}"
  --supersample "${SS:-1}"
  --out "results/experiment_1_1/${out}"
)
if [ "${VIDEOS:-0}" = "1" ]; then
  args+=(--record-video)
fi
if [ "${EXPERT:-0}" = "1" ]; then
  args+=(--expert)
fi
if [ -n "${NA:-}" ]; then
  args+=(--n-action-steps "${NA}")
fi
if [ -n "${SEEDS:-}" ]; then
  args+=(--seeds "${SEEDS}")
else
  args+=(--episodes "${N:-20}" --seed-base "${BASE:-30000000}")
fi
[ -n "${SHIFT_X:-}" ] && args+=(--ood-shift-x "${SHIFT_X}")
[ -n "${SHIFT_Y:-}" ] && args+=(--ood-shift-y "${SHIFT_Y}")
[ -n "${YAW:-}" ] && args+=(--ood-yaw "${YAW}")
[ -n "${FRICTION:-}" ] && args+=(--ood-friction "${FRICTION}")

echo "[e11] out=${out} mode=${MODE:-strict} ss=${SS:-1} na=${NA:-default} expert=${EXPERT:-0} gpu=${GPU:-0} $(date +%T)"
exec venv/bin/python -u scripts/eval_vla_policy.py "${args[@]}" >> "logs/e11_${out}.log" 2>&1
