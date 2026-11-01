#!/usr/bin/env bash
# 在“仿真运行时”venv 里安装 openpi 官方客户端（只新增 msgpack/websockets/openpi_client，不改动已有依赖）
set -u
V=/scratch/lijh/simulate_vla/venv/bin/python
UV=/scratch/lijh/exp40/bin/uv
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 安装前 ==="
$V -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"

"$UV" pip install --python "$V" msgpack "websockets>=13.0" 2>&1 | tail -5
"$UV" pip install --python "$V" --no-deps -e /scratch/lijh/exp40/openpi/packages/openpi-client 2>&1 | tail -3

echo "=== 安装后自检 ==="
$V - <<'PY'
import torch, numpy, pybullet
print("torch", torch.__version__, "numpy", numpy.__version__, "pybullet", pybullet.getAPIVersion())
import openpi_client, msgpack, websockets
print("openpi_client ok, websockets", websockets.__version__, "msgpack", msgpack.version)
PY
