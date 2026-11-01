#!/usr/bin/env bash
# 实验 4.0：建立 OpenPI 推理环境（第三轮：放弃 uv sync，改用显式依赖集）
#
# 原因（都有日志证据，见 install_openpi.log 第一/二轮）：
#   rerun-sdk==0.23.1 只有 manylinux_2_31 wheel  -> / 不适用于 glibc 2.17
#   torch==2.7.1      只有 manylinux_2_28 wheel  -> 同上
#   torchcodec==0.4.0 只有 manylinux_2_28 wheel  -> 同上（lerobot 依赖）
# 这些包属于 training / 数据加载 / 可视化路径。本次只做推理，因此只安装推理所需的
# 显式依赖集，torch 用本机已验证可跑的 2.6.0（manylinux1 wheel）。
set -u
WS=/scratch/lijh/exp40
export PATH="$WS/bin:$PATH"
UV="$WS/bin/uv"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_HTTP_TIMEOUT=180
LOG="$WS/install_openpi.log"

cd "$WS/openpi" || exit 1
PY=.venv/bin/python

{
  echo "=== $(date -Is) 第三轮安装开始（显式依赖集） ==="

  echo "=== 基础: numpy<2 + torch 2.6.0 ==="
  "$UV" pip install --python "$PY" "numpy==1.26.4" "torch==2.6.0" 2>&1 | tail -6

  echo "=== JAX CUDA12 0.5.3 ==="
  "$UV" pip install --python "$PY" "jax[cuda12]==0.5.3" 2>&1 | tail -12

  echo "=== openpi 运行时依赖（去掉 lerobot/wandb/gym-aloha/torchcodec/rerun） ==="
  "$UV" pip install --python "$PY" \
    "flax==0.10.2" \
    "orbax-checkpoint==0.11.13" \
    "tensorstore==0.1.74" \
    "ml-dtypes==0.4.1" \
    "chex==0.1.90" \
    "einops>=0.8.0" \
    "equinox>=0.11.8" \
    "augmax>=0.3.4" \
    "dm-tree>=0.1.8" \
    "jaxtyping==0.2.36" \
    "ml_collections==1.0.0" \
    "safetensors" \
    "transformers==4.53.2" \
    "sentencepiece>=0.2.0" \
    "pillow>=11.0.0" \
    "opencv-python-headless>=4.10.0.84" \
    "imageio>=2.36.1" \
    "numpydantic>=1.6.6" \
    "beartype==0.19.0" \
    "treescope>=0.1.7" \
    "flatbuffers>=24.3.25" \
    "tqdm" \
    "tqdm-loggable>=0.2" \
    "tyro>=0.9.5" \
    "rich>=14.0.0" \
    "polars>=1.30.0" \
    "fsspec[gcs]>=2024.6.0" \
    "filelock>=3.16.1" \
    "typing-extensions>=4.12.2" \
    "dm-tree" \
    "msgpack" \
    "websockets>=13.0" \
    2>&1 | tail -12

  echo "=== openpi-client（本地包） ==="
  "$UV" pip install --python "$PY" -e packages/openpi-client 2>&1 | tail -4

  echo "=== openpi 本体（--no-deps，依赖已显式安装） ==="
  "$UV" pip install --python "$PY" -e . --no-deps 2>&1 | tail -4

  echo "=== 版本自检 ==="
  "$PY" - <<'PY'
import jax, jaxlib, flax, numpy
print("jax    ", jax.__version__)
print("jaxlib ", jaxlib.__version__)
print("flax   ", flax.__version__)
print("numpy  ", numpy.__version__)
try:
    print("devices", jax.devices())
except Exception as e:
    print("devices ERROR", e)
import torch
print("torch  ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
PY

  echo "=== import openpi 推理链路 ==="
  "$PY" - <<'PY'
import openpi.training.config as cfg
print("pi05 configs:", [c.name for c in cfg._CONFIGS if "pi05" in c.name])
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
print("policy_config + serving ok")
PY
  echo "=== $(date -Is) 第三轮安装结束 ==="
} > "$LOG" 2>&1
echo "log=$LOG"
tail -40 "$LOG"
