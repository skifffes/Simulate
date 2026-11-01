#!/usr/bin/env bash
# 准备 uv（官方推荐工具），并确认 torch 2.7.1 的 wheel 标签是否与 glibc 2.17 兼容
set -u
WS=/scratch/lijh/exp40
mkdir -p "$WS/bin"

echo "=== conda 定位 ==="
head -5 /scratch/lijh/.conda/environments.txt 2>/dev/null
ls -l /scratch/lijh/anaconda3/bin/conda /scratch/lijh/miniconda3/bin/conda 2>/dev/null

echo
echo "=== torch 2.7.1 默认 linux wheel 文件名 ==="
curl -s -m 40 https://pypi.org/pypi/torch/2.7.1/json \
 | python -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(sorted({f['filename'] for f in d['urls'] if 'x86_64' in f['filename'] and 'cp311' in f['filename']})))" 2>/dev/null \
 || curl -s -m 40 https://pypi.org/pypi/torch/2.7.1/json | tr ',' '\n' | grep -o '[a-zA-Z0-9_.+-]*cp311[a-zA-Z0-9_.+-]*x86_64\.whl' | sort -u

echo
echo "=== torch 2.6.0 对照 ==="
curl -s -m 40 https://pypi.org/pypi/torch/2.6.0/json | tr ',' '\n' | grep -o '[a-zA-Z0-9_.+-]*cp311[a-zA-Z0-9_.+-]*x86_64\.whl' | sort -u

echo
echo "=== 安装 uv（静态二进制） ==="
if [ -x "$WS/bin/uv" ]; then
  "$WS/bin/uv" --version
else
  curl -LsSf -m 120 https://astral.sh/uv/install.sh -o "$WS/uv_install.sh" && \
    INSTALLER_NO_MODIFY_PATH=1 XDG_BIN_HOME="$WS/bin" sh "$WS/uv_install.sh" 2>&1 | tail -5
  ls -l "$WS/bin" 2>/dev/null
  [ -x "$WS/bin/uv" ] && "$WS/bin/uv" --version
fi
