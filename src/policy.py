"""未来策略替换接口：当前仅提供脚本策略，不加载任何 VLA 权重。"""
from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def act(self, observation):
        """根据观测返回下一步关节目标或末端动作。"""
        raise NotImplementedError

class ScriptPolicy(Policy):
    def act(self, observation):
        return observation.get('target')
