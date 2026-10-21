#!/usr/bin/env bash
# 服务器环境体检：系统 / GPU / 磁盘 / 已有 Python 环境 / 网络
set -u

echo "=== 系统 ==="
hostname
grep PRETTY_NAME /etc/os-release 2>/dev/null
uname -r
echo "glibc: $(ldd --version | head -1)"
echo "CPU: $(nproc) cores"
free -g | awk 'NR<=2'

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
  --format=csv,noheader

echo
echo "=== 磁盘 ==="
df -h / /scratch 2>/dev/null

echo
echo "=== 已有 Python 环境 ==="
for e in $(ls /scratch/*/.conda/envs 2>/dev/null); do
  p="/scratch/lijh/.conda/envs/$e/bin/python"
  [ -x "$p" ] || continue
  printf '%-14s ' "$e"
  "$p" -c 'import sys;print(sys.version.split()[0], end=" ")' 2>/dev/null
  "$p" -c 'import torch;print("torch", torch.__version__, "cu"+str(torch.version.cuda), end=" ")' 2>/dev/null || printf 'no-torch '
  "$p" -c 'import lerobot;print("lerobot", lerobot.__version__, end=" ")' 2>/dev/null || printf 'no-lerobot'
  echo
done

echo
echo "=== 网络 ==="
for url in https://pypi.tuna.tsinghua.edu.cn/simple/ https://hf-mirror.com https://huggingface.co https://github.com; do
  code=$(curl -s -o /dev/null -m 12 -w '%{http_code}' "$url")
  echo "$code  $url"
done

if [ -x /scratch/lijh/simulate_vla/venv/bin/python ]; then
  echo
  echo "=== 本项目环境 ==="
  /scratch/lijh/simulate_vla/venv/bin/python -c 'import sys, torch;print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())' 2>&1 | tail -2
  /scratch/lijh/simulate_vla/venv/bin/python -c 'import lerobot;print("lerobot", lerobot.__version__)' 2>&1 | tail -1
fi
