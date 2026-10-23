#!/usr/bin/env bash
# 实验 2 阶段 3：在 480 集多任务数据集上微调共享 SmolVLA。
# 用法：
#   GPU=0 STEPS=20000 BATCH=8 bash server/train_smolvla_multi.sh
# 后台：
#   GPU=0 STEPS=20000 setsid nohup bash server/train_smolvla_multi.sh > logs/e2_train.log 2>&1 &
set -euo pipefail

VENV=${VENV:-$HOME/simulate_vla/venv}
DATA=${DATA:-$HOME/simulate_vla/data/lerobot_panda_multi}
OUT=${OUT:-$HOME/simulate_vla/outputs/smolvla_multi}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-8}
SAVE_FREQ=${SAVE_FREQ:-2000}
LOG_FREQ=${LOG_FREQ:-50}
WORKERS=${WORKERS:-6}
GPU=${GPU:-0}
JOB=${JOB:-smolvla_multi}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# 服务器没有系统 ffmpeg 的 libavutil，torchcodec 不可用；用 pyav 解码视频。
export VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}

echo "[train] gpu=$GPU steps=$STEPS batch=$BATCH data=$DATA out=$OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed -n "$((GPU + 1))p"

exec "$VENV/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/panda_multi \
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
