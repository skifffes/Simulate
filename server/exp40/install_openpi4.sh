#!/usr/bin/env bash
# 环境修复：pip 解析到的不兼容版本（缺 glibc2.17 wheel）逐个钉住
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
  echo "=== $(date -Is) 第四轮安装（钉住 glibc2.17 可用版本） ==="
  echo "--- cmake? ---"; which cmake || echo "no cmake"

  echo "=== 步骤1：scipy / dm-tree 用有 manylinux2014 wheel 的版本 ==="
  "$UV" pip install --python "$PY" "scipy==1.13.1" "dm-tree==0.1.8" 2>&1 | tail -8

  echo "=== 步骤2：JAX CUDA12 0.5.3 ==="
  "$UV" pip install --python "$PY" "jax[cuda12]==0.5.3" 2>&1 | tail -10

  echo "=== 步骤3：openpi 运行时依赖 ==="
  "$UV" pip install --python "$PY" \
    "flax==0.10.2" "orbax-checkpoint==0.11.13" "tensorstore==0.1.74" "ml-dtypes==0.4.1" \
    "chex==0.1.90" "einops>=0.8.0" "equinox>=0.11.8" "augmax>=0.3.4" "jaxtyping==0.2.36" \
    "ml_collections==1.0.0" "safetensors" "transformers==4.53.2" "sentencepiece>=0.2.0" \
    "pillow>=11.0.0" "opencv-python-headless>=4.10.0.84" "imageio>=2.36.1" "numpydantic>=1.6.6" \
    "beartype==0.19.0" "treescope>=0.1.7" "flatbuffers>=24.3.25" "tqdm" "tqdm-loggable>=0.2" \
    "tyro>=0.9.5" "rich>=14.0.0" "polars>=1.30.0" "fsspec[gcs]>=2024.6.0" "filelock>=3.16.1" \
    "typing-extensions>=4.12.2" "etils[epath]" "msgpack" "websockets>=13.0" 2>&1 | tail -12

  echo "=== 步骤4：openpi-client（--no-deps，依赖已就位） ==="
  "$UV" pip install --python "$PY" -e packages/openpi-client --no-deps 2>&1 | tail -4

  echo "=== 版本自检 ==="
  "$PY" - <<'PY'
import jax, jaxlib, flax, numpy, scipy
print("jax    ", jax.__version__)
print("jaxlib ", jaxlib.__version__)
print("flax   ", flax.__version__)
print("numpy  ", numpy.__version__)
print("scipy  ", scipy.__version__)
print("devices", jax.devices())
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
  echo "=== $(date -Is) 第四轮安装结束 ==="
} > "$LOG" 2>&1
echo "log=$LOG"
tail -45 "$LOG"
