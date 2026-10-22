"""场景随机化与构建。

采集脚本、离线校验和闭环评测都从这里取场景，保证：
- 同一个随机种子在不同脚本里生成完全相同的初始工件位置与朝向；
- 专家评测与 VLA 评测面对的是同一批初始状态；
- 场景变化只来自种子，而不是隐式的全局随机数状态。
"""
from dataclasses import dataclass, field

import numpy as np

from grasp_task import DEFAULT_PARAMS
from manipulation import create_block, enable_contact_solver
from simulation import Simulation


@dataclass
class Scenario:
    """一个可复现的抓放场景。"""

    seed: int
    block_start: tuple
    block_yaw: float
    block_friction: float
    block_half: tuple
    block_mass: float
    params: dict = field(repr=False)

    @property
    def offset_m(self):
        base = DEFAULT_PARAMS['block_start']
        return float(np.hypot(self.block_start[0] - base[0], self.block_start[1] - base[1]))


def make_params(rng, jitter, friction_jitter):
    params = {key: (tuple(value) if isinstance(value, tuple) else value)
              for key, value in DEFAULT_PARAMS.items()}
    params['block_yaw'] = 0.0
    dx = dy = 0.0
    if jitter > 0:
        dx, dy = rng.uniform(-jitter, jitter, 2)
        params['block_yaw'] = float(rng.uniform(-0.10, 0.10))
        base = DEFAULT_PARAMS['block_start']
        params['block_start'] = (base[0] + dx, base[1] + dy, base[2])
        for key in ('pre_grasp', 'grasp', 'lift'):
            params[key] = (params['block_start'][0], params['block_start'][1],
                           DEFAULT_PARAMS[key][2])
    if friction_jitter > 0:
        params['block_friction'] = float(np.clip(
            DEFAULT_PARAMS['block_friction'] + rng.uniform(-friction_jitter, friction_jitter),
            0.8, 2.0))
    return params


def scenario_for_seed(seed, jitter=0.008, friction_jitter=0.2,
                      shift=None, yaw=None, friction=None):
    """由种子确定性地生成场景参数。

    shift/yaw/friction 用于受控 OOD：显式覆盖采样结果，一次只改一个因素。
    目标点、工件尺寸与质量、成功判据都保持不变。
    """
    rng = np.random.default_rng(int(seed))
    params = make_params(rng, jitter, friction_jitter)
    if shift is not None:
        base = DEFAULT_PARAMS['block_start']
        dx, dy = float(shift[0]), float(shift[1])
        params['block_start'] = (base[0] + dx, base[1] + dy, base[2])
        for key in ('pre_grasp', 'grasp', 'lift'):
            params[key] = (params['block_start'][0], params['block_start'][1],
                           DEFAULT_PARAMS[key][2])
    if yaw is not None:
        params['block_yaw'] = float(yaw)
    if friction is not None:
        params['block_friction'] = float(friction)
    return Scenario(seed=int(seed), block_start=tuple(params['block_start']),
                    block_yaw=float(params['block_yaw']),
                    block_friction=float(params['block_friction']),
                    block_half=tuple(params['block_half']),
                    block_mass=float(params['block_mass']), params=params)


def build_scene(scenario):
    """按场景参数搭建 PyBullet 世界，返回 (sim, block)。"""
    import pybullet as p

    params = scenario.params
    sim = Simulation(gui=False, reference_cube=False)
    enable_contact_solver(sim)
    block = create_block(sim, params['block_half'], params['block_mass'], params['block_start'],
                         orientation=p.getQuaternionFromEuler([0, 0, params.get('block_yaw', 0.0)]),
                         lateral_friction=params['block_friction'])
    sim.allowed_contacts = {(block, 9), (block, 10)}
    return sim, block


__all__ = ['Scenario', 'make_params', 'scenario_for_seed', 'build_scene']
