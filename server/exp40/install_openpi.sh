#!/usr/bin/env bash
# 实验 4.0：在 A40 服务器上建立独立 OpenPI 环境（JAX 路线）。
#
# 关键约束：CentOS 7 / glibc 2.17。
#  - jaxlib 0.5.3 提供 manylinux2014 wheel  -> 可用
#  - torch 2.7.1 只提供 manylinux_2_28 wheel -> 不可用，改用本机已验证的 torch 2.6.0
#    （torch 只用于 openpi 的图像工具与 lerobot 数据加载，推理路径是 JAX）
set -u
set -o pipefail

WS=/scratch/lijh/exp40
UV="$WS/bin/uv"
export PATH="$WS/bin:$PATH"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_HTTP_TIMEOUT=120
export GIT_LFS_SKIP_SMUDGE=1

LOG="$WS/install_openpi.log"
cd "$WS/openpi" || exit 1

{
  echo "=== $(date -Is) 开始安装 ==="
  "$UV" --version

  echo "=== 创建 python 3.11 venv ==="
  "$UV" venv --python 3.11 --seed .venv 2>&1 | tail -5
  ./.venv/bin/python -V

  echo "=== uv sync（跳过 torch，随后单独装 2.6.0） ==="
  "$UV" sync --no-install-package torch 2>&1 | tail -40

  echo "=== 安装 torch 2.6.0（glibc 2.17 可用） ==="
  "$UV" pip install "torch==2.6.0" "torchvision==0.21.0" 2>&1 | tail -10

  echo "=== 版本自检 ==="
  ./.venv/bin/python - <<'PY'
import jax, jaxlib, flax, numpy, optax
print("jax      ", jax.__version__)
print("jaxlib   ", jaxlib.__version__)
print("flax     ", flax.__version__)
print("numpy    ", numpy.__version__)
print("devices  ", jax.devices())
import torch
print("torch    ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
PY

  echo "=== import openpi 关键模块 ==="
  ./.venv/bin/python - <<'PY'
import openpi
import openpi.training.config as cfg
print("openpi import ok")
names = [c.name for c in cfg._CONFIGS]
print("pi05 configs:", [n for n in names if "pi05" in n])
PY

  echo "=== $(date -Is) 安装结束 ==="
} > "$LOG" 2>&1
echo "done, log=$LOG"
tail -60 "$LOG"
