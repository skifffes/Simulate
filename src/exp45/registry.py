"""实验 4.5 的 Action Expert-only 配置注册（非 LoRA 变体）。

与 exp42 的 LoRA registry 的唯一差别：paligemma_variant='gemma_2b'、
action_expert_variant='gemma_300m'，参数树无 lora 张量。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import flax.nnx as nnx
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.droid_policy as droid_policy
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

CONFIG_NAME = 'exp45_pi05_ae'
ASSETS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets')
PARAMS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params')
HORIZON = 16


def build_config():
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
    return dataclasses.replace(
        base, model=model_cfg, data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARAMS)),
        freeze_filter=nnx.Nothing, ema_decay=None, batch_size=4, num_train_steps=12000)


def register() -> str:
    cfg = build_config()
    names = {c.name for c in _config._CONFIGS}
    if CONFIG_NAME not in names:
        _config._CONFIGS.append(cfg)
    _config._CONFIGS_DICT[CONFIG_NAME] = cfg
    return CONFIG_NAME


if __name__ == '__main__':
    print(register())
