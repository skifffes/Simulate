#!/usr/bin/env bash
# 实验 4.0：服务器网络与资源侦察（只读，不修改任何环境）
set -u

echo "=== 网络可达性 ==="
for url in \
  https://pypi.tuna.tsinghua.edu.cn/simple/ \
  https://mirrors.aliyun.com/pypi/simple/ \
  https://hf-mirror.com \
  https://huggingface.co \
  https://github.com \
  https://storage.googleapis.com \
  https://pypi.org/simple/ ; do
  code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$url")
  echo "$code  $url"
done

echo
echo "=== 代理环境变量 ==="
env | grep -i proxy || echo "(none)"

echo
echo "=== 关键 URL 深层探测 ==="
# openpi checkpoint 默认托管在 GCS 上
for url in \
  "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/pi05_droid/norm_stats.json" \
  "https://github.com/Physical-Intelligence/openpi/info/refs?service=git-upload-pack" ; do
  code=$(curl -s -o /dev/null -m 20 -w '%{http_code} size=%{size_download}' "$url")
  echo "$code  $url"
done

echo
echo "=== curl 版本（是否支持断点续传） ==="
curl --version | head -2

echo
echo "=== 已有 conda env 的 python 版本 ==="
for e in /scratch/lijh/.conda/envs/*/; do
  name=$(basename "$e")
  p="$e/bin/python"
  if [ -x "$p" ]; then
    v=$("$p" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null || echo "?")
    echo "$name  $v"
  fi
done

echo
echo "=== 已有 openpi / jax / torch ==="
for e in /scratch/lijh/.conda/envs/*/; do
  name=$(basename "$e")
  p="$e/bin/python"
  [ -x "$p" ] || continue
  out=$("$p" - <<'PY' 2>/dev/null
mods = []
for m in ("jax", "torch", "openpi", "flax", "pybullet"):
    try:
        mod = __import__(m)
        mods.append(f"{m}={getattr(mod,'__version__','?')}")
    except Exception:
        pass
print(" ".join(mods))
PY
)
  [ -n "$out" ] && echo "$name: $out"
done

echo
echo "=== 磁盘 ==="
df -h /scratch / 2>/dev/null
echo
echo "=== /scratch 大目录 ==="
du -sh /scratch/lijh/.cache /scratch/lijh/simulate_vla 2>/dev/null
