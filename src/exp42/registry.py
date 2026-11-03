"""把实验 4.2 的 LoRA 训练配置注册进 openpi 的 config 表（供推理服务复用）。

必要性：训练用的是 `gemma_2b_lora` + `gemma_300m_lora` 变体，参数树里含 LoRA 参数；
官方 `pi05_droid` 配置是全参数变体，参数树不匹配，**不能**直接加载微调 checkpoint。
因此推理端必须用与训练**完全相同**的模型配置（pi05=True, action_dim=32,
action_horizon=16, 两个 lora variant）来构建策略。

用法：被 `scripts/exp42/serve_finetuned.py` 在启动前 import，自动完成注册。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.droid_policy as droid_policy
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

CONFIG_NAME = 'exp42_pi05_droid_lora'
ASSETS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets')
PARAMS = Path('/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params')
HORIZON = 16


def build_config():
    base = _config.get_config('pi05_droid_finetune')
    model_cfg = pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=HORIZON,
        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    data_factory = _config.SimpleDataConfig(
        assets=_config.AssetsConfig(assets_dir=str(ASSETS), asset_id='droid'),
        data_transforms=lambda m: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=_model.ModelType.PI05)],
            outputs=[droid_policy.DroidOutputs()]),
        base_config=_config.DataConfig(prompt_from_task=True))
    freeze = pi0_config.Pi0Config(
        paligemma_variant='gemma_2b_lora',
        action_expert_variant='gemma_300m_lora').get_freeze_filter()
    return dataclasses.replace(
        base, name=CONFIG_NAME, model=model_cfg, data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARAMS)),
        freeze_filter=freeze, ema_decay=None)


def register() -> str:
    """幂等注册；返回 config 名。

    openpi 用模块级 `_CONFIGS_DICT` 做查找（不是每次扫 `_CONFIGS`），
    因此必须同时写入列表与字典，否则 `get_config` 仍会报 not found。
    """
    cfg = build_config()
    names = {c.name for c in _config._CONFIGS}
    if CONFIG_NAME not in names:
        _config._CONFIGS.append(cfg)
    _config._CONFIGS_DICT[CONFIG_NAME] = cfg
    return CONFIG_NAME


if __name__ == '__main__':
    print(register())
