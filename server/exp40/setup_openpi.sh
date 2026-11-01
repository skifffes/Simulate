#!/usr/bin/env bash
# 实验 4.0：拉取官方 openpi 源码（不含 LFS 权重），并侦察 checkpoint 资源。
set -u
WS="${WS:-/scratch/lijh/exp40}"
mkdir -p "$WS"
cd "$WS" || exit 1

echo "=== git / disk ==="
git --version
df -h /scratch | tail -1

echo
echo "=== clone openpi ==="
if [ -d "$WS/openpi/.git" ]; then
  echo "openpi 已存在，跳过 clone"
else
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git "$WS/openpi" 2>&1 | tail -5
fi
cd "$WS/openpi" || exit 1
echo "commit: $(git rev-parse HEAD)"
echo "date  : $(git log -1 --format=%cI)"
echo "files : $(git ls-files | wc -l)"

echo
echo "=== pyproject (python/jax 版本要求) ==="
sed -n '1,80p' pyproject.toml

echo
echo "=== README 中的安装说明片段 ==="
grep -n -A6 -i "uv sync" README.md | head -40

echo
echo "=== 是否存在 docs/remote_inference.md ==="
ls -la docs/ 2>&1 | head -20

echo
echo "=== models_pytorch 是否存在 ==="
ls src/openpi/models_pytorch/ 2>&1 | head -20

echo
echo "=== GCS checkpoint 列表（pi05_droid） ==="
curl -s -m 40 "https://storage.googleapis.com/openpi-assets?prefix=checkpoints/pi05_droid&max-keys=1000" | tr '<' '\n' | grep -E "^(Key|Size)>" | paste - - | head -60

echo
echo "=== GCS checkpoint 列表（pi05_base 摘要） ==="
curl -s -m 40 "https://storage.googleapis.com/openpi-assets?prefix=checkpoints/pi05_base&max-keys=1000" | tr '<' '\n' | grep -E "^(Key|Size)>" | paste - - | head -40
