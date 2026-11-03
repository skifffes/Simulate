#!/usr/bin/env bash
# P0b 闭环诊断：动作增益（0.5 / 0.2）+ 原始模型基线，同一场景/seed/评价器
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
SIM=/scratch/lijh/simulate_vla/venv/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

start_server() {  # $1=config $2=ckpt $3=port $4=gpu $5=log
  CUDA_VISIBLE_DEVICES=$4 setsid nohup "$OAPI/.venv/bin/python" scripts/exp42/serve_finetuned.py \
    --mode checkpoint --config "$1" --checkpoint "$2" --port "$3" > "$5" 2>&1 < /dev/null &
  echo $!
}

wait_up() {  # $1=pid $2=log $3=seconds
  for i in $(seq 1 "$3"); do
    sleep 3
    if ! kill -0 "$1" 2>/dev/null; then echo "server died"; tail -15 "$2"; return 1; fi
    if grep -qi 'listening' "$2" 2>/dev/null; then echo "up after $((i*3))s"; return 0; fi
  done
  echo timeout; tail -15 "$2"; return 1
}

echo "=== 微调 step901 + gain 0.5 / 0.2 ==="
P1=$(start_server exp42_pi05_droid_lora /scratch/lijh/exp42/run2/ckpts/params_nested 8131 1 /scratch/lijh/exp42/serve_gain.log)
wait_up "$P1" /scratch/lijh/exp42/serve_gain.log 60 || exit 1
for G in 0.5 0.2; do
  for S in A90 B90; do
    "$SIM" scripts/exp42/eval_rollouts.py --skill $S --episodes 1 --ticks 45 --port 8131 \
      --tag "gain$G" --action-gain $G --out results/experiment_4_2/gain_rollouts 2>&1 | tail -3
  done
done
kill "$P1" 2>/dev/null || true

echo "=== 原始 pi05_droid 基线 ==="
CUDA_VISIBLE_DEVICES=0 setsid nohup "$OAPI/.venv/bin/python" scripts/exp40/serve_policy.py \
  --mode checkpoint --config pi05_droid --port 8132 > /scratch/lijh/exp42/serve_base.log 2>&1 < /dev/null &
P2=$!
wait_up "$P2" /scratch/lijh/exp42/serve_base.log 60 || { kill $P2; exit 1; }
for S in A90 B90; do
  "$SIM" scripts/exp42/eval_rollouts.py --skill $S --episodes 2 --ticks 45 --port 8132 \
    --tag baseline --out results/experiment_4_2/baseline_rollouts 2>&1 | tail -3
done
kill "$P2" 2>/dev/null || true
echo "done"
