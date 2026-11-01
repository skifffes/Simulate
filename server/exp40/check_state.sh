#!/usr/bin/env bash
# 服务器现状与仿真侧 venv 依赖检查
set -u
WS=/scratch/lijh/exp40
echo "=== torch 安装 / uv 进程 ==="
pgrep -af "uv pip|uv sync|install_openpi" | head -5
tail -4 "$WS/install_openpi.log"
du -sh "$WS/openpi/.venv" 2>/dev/null

echo
echo "=== checkpoint 下载进度 ==="
tail -6 "$WS/download_ckpt.log" 2>/dev/null
du -sh "$HOME/.cache/openpi" 2>/dev/null

echo
echo "=== 仿真工程目录结构 ==="
ls /scratch/lijh/simulate_vla | head -20
echo "--- 是否是 git 仓库 ---"
git -C /scratch/lijh/simulate_vla rev-parse HEAD 2>&1 | head -2
echo "--- src 是否存在 ---"
ls /scratch/lijh/simulate_vla/src 2>&1 | head -10

echo
echo "=== 仿真 venv 依赖 ==="
PY=/scratch/lijh/simulate_vla/venv/bin/python
$PY -V
$PY - <<'PY'
import importlib
for m in ("pybullet","numpy","torch","lerobot","msgpack","websockets","PIL","yaml","matplotlib","pandas"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:12s} MISSING ({type(e).__name__})")
PY

echo
echo "=== GPU 占用 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== 端口 8000/8001 ==="
ss -ltn 2>/dev/null | grep -E ':(8000|8001|9000)' || echo "no listener"
