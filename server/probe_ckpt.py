"""检查训练 checkpoint 的输入特征、归一化配置与统计量是否完整。"""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "outputs/smolvla_panda/checkpoints/003000/pretrained_model")
cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
print("chunk_size", cfg.get("chunk_size"), "n_action_steps", cfg.get("n_action_steps"))
print("input_features", list(cfg.get("input_features", {})))
print("output_features", list(cfg.get("output_features", {})))
print("normalization_mapping", cfg.get("normalization_mapping"))
print("use_imagenet_stats", cfg.get("use_imagenet_stats"))
