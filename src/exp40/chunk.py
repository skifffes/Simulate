"""Action chunk 执行机制。

prompt 7.3 要求明确：一次预测多少步、实际执行多少步、何时重新推理、chunk 缓冲、
重置时如何清空缓存、不同任务指令如何切换。

设计：
- 模型一次返回 (action_horizon=15, 8)；
- 默认只执行前 `execute_k` 步就重新推理（DROID 官方 open_loop_horizon=8）；
- `reset()` 清空缓冲（回合开始/场景重置/切换技能时调用）；
- `set_prompt()` 切换指令并可选清空缓冲；4.0 不实现 A→B 技能连接器，
  但保留 `clear_on_prompt_change` 这个开关，供 4.2 的 A→B 边界实验使用。
"""
from __future__ import annotations

import numpy as np


class ActionChunker:
    def __init__(self, execute_k: int = 8, clear_on_prompt_change: bool = True):
        if execute_k < 1:
            raise ValueError('execute_k 必须 >= 1')
        self.execute_k = int(execute_k)
        self.clear_on_prompt_change = bool(clear_on_prompt_change)
        self._buffer: np.ndarray | None = None
        self._cursor = 0
        self._prompt: str | None = None
        self.stats = {
            'chunks_received': 0,
            'actions_consumed': 0,
            'refills': 0,
            'prompt_changes': 0,
        }

    # ---- 状态 -------------------------------------------------------------
    @property
    def prompt(self) -> str | None:
        return self._prompt

    def needs_inference(self) -> bool:
        if self._buffer is None:
            return True
        return self._cursor >= min(self.execute_k, len(self._buffer))

    def remaining(self) -> int:
        if self._buffer is None:
            return 0
        return max(0, min(self.execute_k, len(self._buffer)) - self._cursor)

    # ---- 缓冲 -------------------------------------------------------------
    def push(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f'chunk 必须是 (T, A)，收到 {chunk.shape}')
        self._buffer = chunk
        self._cursor = 0
        self.stats['chunks_received'] += 1
        self.stats['refills'] += 1

    def pop(self) -> np.ndarray | None:
        """取当前 chunk 里下一条待执行动作；耗尽返回 None（调用方应重新推理）。"""
        if self.needs_inference():
            return None
        action = np.array(self._buffer[self._cursor], copy=True)
        self._cursor += 1
        self.stats['actions_consumed'] += 1
        return action

    def reset(self) -> None:
        self._buffer = None
        self._cursor = 0

    def set_prompt(self, prompt: str) -> None:
        if self._prompt != prompt:
            self.stats['prompt_changes'] += 1
            self._prompt = str(prompt)
            if self.clear_on_prompt_change:
                self.reset()

    def summary(self) -> dict:
        return {
            'execute_k': self.execute_k,
            'clear_on_prompt_change': self.clear_on_prompt_change,
            'buffer_shape': None if self._buffer is None else list(self._buffer.shape),
            'cursor': self._cursor,
            'prompt': self._prompt,
            **self.stats,
        }
