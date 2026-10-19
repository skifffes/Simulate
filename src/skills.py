"""简化的技能 A/B/C：用于研究交接状态，不代表真实魔方操作。"""
from dataclasses import dataclass
import numpy as np
import pybullet as p

@dataclass(frozen=True)
class SkillTarget:
    name: str
    position: tuple
    orientation: tuple

DOWN = p.getQuaternionFromEuler([np.pi, 0, np.pi / 2])

# A 的成功条件相同：末端到达立方体左侧的准备区域；z1/z2 只改变最终姿态。
A_TARGET = (0.43, 0.0, 0.34)
Z1 = SkillTarget('z1', A_TARGET, p.getQuaternionFromEuler([np.pi, 0, np.pi / 2]))
Z2 = SkillTarget('z2', A_TARGET, p.getQuaternionFromEuler([np.pi, 0, -np.pi / 2]))
B = SkillTarget('B', (0.55, 0.0, 0.36), DOWN)       # 立方体上方
C = SkillTarget('C', (0.55, -0.22, 0.12), DOWN)     # 立方体前方
CANONICAL = SkillTarget('canonical', (0.43, 0.0, 0.34), DOWN)


def pose_error(sim, target):
    position, _ = sim.pose()
    return float(np.linalg.norm(position - np.asarray(target.position)))


def skill_success(sim, target, tolerance=0.08):
    return pose_error(sim, target) < tolerance and not sim.collisions()
