"""校验 SmolVLA 预训练权重可加载，并打印关键训练配置。"""

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
total = sum(p.numel() for p in policy.parameters()) / 1e6
print(f"params {total:.1f}M")
print("chunk_size", policy.config.chunk_size, "n_action_steps", policy.config.n_action_steps)
print("image_features", list(policy.config.image_features))
print("use_imagenet_stats", policy.config.use_imagenet_stats)
print("normalization_mapping", policy.config.normalization_mapping)
