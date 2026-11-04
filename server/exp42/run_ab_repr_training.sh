#!/usr/bin/env bash
# Gate 4 任务 3：A/B 单变量动作表示对照（各 300 optimizer updates）
#   A: label_mode=anchor            + anchor_reference=fixed
#   B: label_mode=target_increment  + anchor_reference=recursive
# 其余完全相同：base=pi05_droid_finetune、同 LoRA、batch 4、lr 默认、seed 0、同数据顺序、horizon 16、
# DROID stats、32D loss、无 accum。
set -u
REPO=/scratch/lijh/exp40/repo
OAPI=/scratch/lijh/exp40/openpi
STUB=/scratch/lijh/exp40/lerobot_stub
cd "$REPO" || exit 1
export PYTHONPATH="$STUB:$REPO/src:$REPO:$OAPI/src:$OAPI/packages/openpi-client/src"
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/lijh/exp42/jax_cache

STEPS=${1:-300}
run_group () {
  local mode=$1 gpu=$2 out=$3
  CUDA_VISIBLE_DEVICES=$gpu EXP42_OUT="$out" setsid nohup "$OAPI/.venv/bin/python" \
    scripts/exp42/train_lora.py --mode train --steps "$STEPS" --batch 4 --seed 0 \
    --label-mode "$mode" --save-every 100 --export-params \
    > "/scratch/lijh/exp42/train_${mode}.log" 2>&1 < /dev/null &
  echo "$mode pid $! on gpu$gpu -> $out"
}

run_group anchor 0 /scratch/lijh/exp42/ab_repr/A
run_group target_increment 1 /scratch/lijh/exp42/ab_repr/B
echo 'launched; follow /scratch/lijh/exp42/train_{anchor,target_increment}.log'
