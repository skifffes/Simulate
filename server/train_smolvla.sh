#!/usr/bin/env bash
# 在服务器上微调 SmolVLA（150 条 pick-and-place 原子数据集）
# 用法:
#   GPU=0 STEPS=20000 BATCH=8 bash train_smolvla.sh
# 后台运行:
#   GPU=0 nohup bash train_smolvla.sh > ~/simulate_vla/train.log 2>&1 &
set -euo pipefail

VENV=${VENV:-$HOME/simulate_vla/venv}
DATA=${DATA:-$HOME/simulate_vla/data/lerobot_panda_pick_place}
OUT=${OUT:-$HOME/simulate_vla/outputs/smolvla_panda}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-8}
SAVE_FREQ=${SAVE_FREQ:-2000}
LOG_FREQ=${LOG_FREQ:-50}
WORKERS=${WORKERS:-6}
GPU=${GPU:-0}
JOB=${JOB:-smolvla_panda}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# 无系统 ffmpeg，视频解码走 pyav
export VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}

# 注意：不要预先 mkdir "$OUT"，lerobot-train 要求输出目录不存在，否则直接报 FileExistsError。
echo "[train] gpu=$GPU steps=$STEPS batch=$BATCH data=$DATA out=$OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed -n "$((GPU + 1))p"

exec "$VENV/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/panda_pick_place \
  --dataset.root="$DATA" \
  --dataset.video_backend="$VIDEO_BACKEND" \
  --output_dir="$OUT" \
  --job_name="$JOB" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --num_workers="$WORKERS" \
  --seed=1000 \
  --wandb.enable=false
