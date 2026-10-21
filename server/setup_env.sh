#!/usr/bin/env bash
# 服务器环境准备（可重复执行，已安装的步骤会跳过）。
# 目标：在 CentOS 7 / glibc 2.17 / 驱动 550.120 的 A40 机器上跑起 LeRobot 0.3.3 + SmolVLA。
#
# 用法：bash server/setup_env.sh
set -euo pipefail

ROOT="${ROOT:-$HOME/simulate_vla}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
BASE_PY="${BASE_PY:-/scratch/lijh/.conda/envs/gxc/bin/python3.10}"
TORCH_VER="${TORCH_VER:-2.6.0}"
TV_VER="${TV_VER:-0.21.0}"
LEROBOT_VER="${LEROBOT_VER:-0.3.3}"

mkdir -p "$ROOT"

if [ ! -x "$ROOT/venv/bin/python" ]; then
    echo "== 创建 venv（基于 $BASE_PY）"
    "$BASE_PY" -m venv "$ROOT/venv"
fi

PIP="$ROOT/venv/bin/pip"
PY="$ROOT/venv/bin/python"
"$PIP" install -q -U pip -i "$PIP_MIRROR"

echo "== PyTorch $TORCH_VER + torchvision $TV_VER（PyPI 默认即 cu124 构建）"
"$PIP" install "torch==$TORCH_VER" "torchvision==$TV_VER" -i "$PIP_MIRROR"

echo "== LeRobot $LEROBOT_VER"
"$PIP" install "lerobot==$LEROBOT_VER" -i "$PIP_MIRROR"

echo "== 评测侧依赖（PyBullet 无窗口渲染 + 视频导出）"
"$PIP" install pybullet imageio-ffmpeg -i "$PIP_MIRROR"

echo "== 版本自检"
"$PY" - <<'EOF'
import sys, torch, torchvision, lerobot, pybullet
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("lerobot    ", lerobot.__version__)
print("pybullet   ", pybullet.getAPIVersion())
EOF
echo
echo "提示：下载 Hugging Face 权重时请设置 HF_ENDPOINT=https://hf-mirror.com"
