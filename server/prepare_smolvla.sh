#!/usr/bin/env bash
# 在服务器 venv 里准备 SmolVLA 训练所需的一切（依赖已在 venv 中，这里只做模型下载 + 自检）
# 用法: bash prepare_smolvla.sh
set -euo pipefail

VENV=${VENV:-$HOME/simulate_vla/venv}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

echo "== 1. 校验 torch =="
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
assert torch.cuda.is_available(), "GPU 不可用"
PY

echo "== 2. 下载 smolvla_base 与 VLM 主干 =="
"$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

for repo in ("lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
    print("ok", repo, "->", path)
PY

echo "== 3. 检查 lerobot 可导入与数据集可加载 =="
"$VENV/bin/python" - <<'PY'
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = os.path.expanduser("~/simulate_vla/data/lerobot_panda_pick_place")
ds = LeRobotDataset("local/panda_pick_place", root=root, video_backend="pyav",
                    delta_timestamps={"action": [i / 20 for i in range(50)]})
print("episodes", ds.num_episodes, "frames", ds.num_frames, "fps", ds.fps)
item = ds[0]
for k, v in item.items():
    print(" ", k, getattr(v, "shape", v))
PY

echo "== 4. 检查 smolvla 策略可构建 =="
"$VENV/bin/python" - <<'PY'
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
n = sum(p.numel() for p in policy.parameters())
print("params", f"{n/1e6:.1f}M", "image_features", list(policy.config.image_features))
PY

echo "全部就绪"
