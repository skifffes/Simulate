"""实验 4.5 公共配置：Action Expert-only 的模型/冻结过滤/数据路径。"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import flax.nnx as nnx
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.droid_policy as droid_policy
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ASSETS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets')
PARAMS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params')
DATA = Path('/scratch/lijh/exp43/export_v3')
HORIZON = 16

# 可训练 = Action Expert（llm 下所有 *_1 参数，含 adaRMSNorm Dense）+ 动作投影
TRAINABLE_PATTERNS = [
    r'PaliGemma/llm/.*_1(/.*)?',
    r'action_in_proj(:?/.*)?',
    r'action_out_proj(:?/.*)?',
    r'time_mlp_in(:?/.*)?',
    r'time_mlp_out(:?/.*)?',
]


def build_config_ae(batch_size: int = 4, num_steps: int = 12000, lr_scale: float = 1.0):
    base = _config.get_config('pi05_droid_finetune')
    model_cfg = pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=HORIZON,
        paligemma_variant='gemma_2b', action_expert_variant='gemma_300m')
    data_factory = _config.SimpleDataConfig(
        assets=_config.AssetsConfig(assets_dir=str(ASSETS), asset_id='droid'),
        data_transforms=lambda m: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=_model.ModelType.PI05)],
            outputs=[droid_policy.DroidOutputs()]),
        base_config=_config.DataConfig(prompt_from_task=True))
    trainable = nnx.Any(*[nnx_utils.PathRegex(p) for p in TRAINABLE_PATTERNS])
    cfg = dataclasses.replace(
        base, model=model_cfg, data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARAMS)),
        freeze_filter=nnx.Not(trainable), ema_decay=None,
        batch_size=int(batch_size), num_train_steps=int(num_steps))
    if lr_scale != 1.0:
        cfg = dataclasses.replace(cfg, lr_schedule=dataclasses.replace(
            cfg.lr_schedule, peak_lr=cfg.lr_schedule.peak_lr * lr_scale))
    return cfg
