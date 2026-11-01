#!/usr/bin/env bash
# 实验 4.0：建立 OpenPI 环境（第二轮，处理 glibc 2.17 不兼容的两个包）
#
# 第一轮报错：rerun-sdk==0.23.1 只有 manylinux_2_31 wheel（lerobot 的依赖）。
# 第二轮同时排除 torch==2.7.1（manylinux_2_28），改成本机已验证的 torch 2.6.0。
# torch 在 openpi 推理路径里只用于图像工具与类型声明，模型本身是 JAX。
set -u
WS=/scratch/lijh/exp40
export PATH="$WS/bin:$PATH"
UV="$WS/bin/uv"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_HTTP_TIMEOUT=180
export GIT_LFS_SKIP_SMUDGE=1
LOG="$WS/install_openpi.log"

cd "$WS/openpi" || exit 1
{
  echo "=== $(date -Is) 第二轮安装开始 ==="
  echo "=== uv sync: 排除 rerun-sdk / torch ==="
  "$UV" sync --no-install-package rerun-sdk --no-install-package torch 2>&1 | tail -25

  echo "=== 安装 torch 2.6.0 ==="
  "$UV" pip install --python .venv/bin/python "torch==2.6.0" "torchvision==0.21.0" 2>&1 | tail -15

  echo "=== 版本自检 ==="
  ./.venv/bin/python - <<'PY'
import jax, jaxlib, flax, numpy
print("jax    ", jax.__version__)
print("jaxlib ", jaxlib.__version__)
print("flax   ", flax.__version__)
print("numpy  ", numpy.__version__)
print("devices", jax.devices())
import torch
print("torch  ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
PY

  echo "=== import openpi 推理链路 ==="
  ./.venv/bin/python - <<'PY'
import openpi.training.config as cfg
print("config ok; pi05 configs:", [c.name for c in cfg._CONFIGS if "pi05" in c.name])
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
print("policy_config + serving ok")
PY
  echo "=== $(date -Is) 第二轮安装结束 ==="
} > "$LOG" 2>&1
echo "log=$LOG"
tail -40 "$LOG"
