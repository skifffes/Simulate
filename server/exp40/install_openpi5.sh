#!/usr/bin/env bash
# 实验 4.0：逐个安装 openpi 推理依赖，单个包失败不影响其它包，并强制优先二进制 wheel。
set -u
WS=/scratch/lijh/exp40
export PATH="$WS/bin:$PATH"
UV="$WS/bin/uv"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_HTTP_TIMEOUT=180
LOG="$WS/install_openpi5.log"
cd "$WS/openpi" || exit 1
PY=.venv/bin/python

PKGS="flax==0.10.2 orbax-checkpoint==0.11.13 tensorstore==0.1.74 ml-dtypes==0.4.1 chex==0.1.90 \
einops>=0.8.0 equinox>=0.11.8 augmax>=0.3.4 jaxtyping==0.2.36 ml_collections==1.0.0 safetensors \
transformers==4.53.2 sentencepiece>=0.2.0 pillow>=11.0.0 opencv-python-headless>=4.10.0.84 imageio>=2.36.1 \
numpydantic>=1.6.6 beartype==0.19.0 treescope>=0.1.7 flatbuffers>=24.3.25 tqdm tqdm-loggable>=0.2 \
tyro>=0.9.5 rich>=14.0.0 polars>=1.30.0 fsspec[gcs]>=2024.6.0 filelock>=3.16.1 typing-extensions>=4.12.2 \
etils[epath] msgpack websockets>=13.0"

{
echo "=== $(date -Is) 第五轮：逐包安装 ==="
for p in $PKGS; do
  echo "--- $p"
  if "$UV" pip install --python "$PY" --only-binary=:all: "$p" 2>&1 | tail -3; then
    echo "    rc=0"
  else
    echo "    rc!=0 (wheel 不可用，尝试允许源码)"
    "$UV" pip install --python "$PY" "$p" 2>&1 | tail -3 || echo "    FAILED $p"
  fi
done

echo "=== 版本自检 ==="
"$PY" - <<'PY'
import importlib
mods = ["jax","jaxlib","flax","numpy","scipy","orbax.checkpoint","tensorstore","etils.epath",
        "transformers","safetensors","einops","augmax","chex","ml_collections","tyro","openpi_client"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m:22s} {getattr(mod,'__version__','ok')}")
    except Exception as e:
        print(f"{m:22s} MISSING {type(e).__name__}: {e}")
print("devices", jax.devices())
PY

echo "=== import openpi 推理链路 ==="
"$PY" - <<'PY'
import openpi.training.config as cfg
print("pi05 configs:", [c.name for c in cfg._CONFIGS if "pi05" in c.name])
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
print("policy_config + serving ok")
PY
echo "=== $(date -Is) 第五轮结束 ==="
} > "$LOG" 2>&1
echo "log=$LOG"
grep -E "MISSING|FAILED|ok$|devices|pi05 configs|Traceback" "$LOG" | head -40
