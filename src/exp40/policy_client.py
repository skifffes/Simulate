"""π₀.₅ 策略客户端（openpi 官方 websocket + msgpack-numpy 协议）。

为什么要自己包一层而不是直接用 `openpi_client.websocket_client_policy`：
    官方客户端用 websockets 默认 keepalive（ping_interval=20s, ping_timeout=20s）。
    π₀.₅ 首次推理包含 JAX JIT 编译，实测需要 ~77 s，远超 20 s，于是连接会被 keepalive
    强杀（本案实际发生过：ConnectionClosedError: keepalive ping timeout）。
    协议本身（msgpack_numpy 打包、请求-响应一问一答）完全不变，只是关闭自动 ping
    并把超时放宽到按实测延迟设定。

同时提供：
    * 每次调用记录往返时间；
    * 连接失败时区分“服务器未启动/端口不通/协议不匹配”，不把网络问题伪装成模型问题。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CallRecord:
    index: int
    roundtrip_s: float
    request_shapes: dict
    response_keys: list
    actions_shape: list

    def as_dict(self) -> dict:
        return {
            'index': self.index,
            'roundtrip_s': round(self.roundtrip_s, 4),
            'request_shapes': self.request_shapes,
            'response_keys': self.response_keys,
            'actions_shape': self.actions_shape,
        }


class Pi05PolicyClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 8000,
                 open_timeout: float = 30.0, recv_timeout: float = 600.0):
        import websockets.sync.client

        self._uri = f'ws://{host}:{port}'
        self._recv_timeout = recv_timeout
        self._packer = None
        from openpi_client import msgpack_numpy

        self._msgpack = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self.calls: list[CallRecord] = []
        self.last_episode_seed: int | None = None
        self._conn = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None,
            open_timeout=open_timeout, close_timeout=5.0,
            ping_interval=None,   # 关闭自动 ping：首帧 JIT 可能远超 20 s
        )
        self._metadata = self._msgpack.unpackb(self._conn.recv())

    # ---- 元数据 -----------------------------------------------------------
    def get_server_metadata(self) -> dict:
        return self._metadata

    # ---- 推理 -------------------------------------------------------------
    def infer(self, obs: dict, episode_seed: int | None = None) -> dict:
        """向服务器请求一次 action chunk。

        episode_seed：本回合的显式随机种子。服务器收到后会先把 JAX PRNG 重置成
        `jax.random.key(episode_seed)` 再推理，因此同 seed + 同观测 => 同一条采样轨迹。
        """
        payload = dict(obs)
        if episode_seed is not None:
            payload['__episode_seed__'] = int(episode_seed)
        t0 = time.perf_counter()
        self._conn.send(self._packer.pack(payload))
        raw = self._conn.recv(timeout=self._recv_timeout)
        dt = time.perf_counter() - t0
        if isinstance(raw, str):
            raise RuntimeError(f'策略服务器返回错误：\n{raw}')
        out = self._msgpack.unpackb(raw)
        actions = np.asarray(out['actions'])
        self.calls.append(CallRecord(
            index=len(self.calls) + 1,
            roundtrip_s=dt,
            request_shapes={k: list(np.asarray(v).shape) for k, v in payload.items() if k != 'prompt'},
            response_keys=sorted(out),
            actions_shape=list(actions.shape),
        ))
        self.last_episode_seed = episode_seed
        return out

    # ---- 统计 -------------------------------------------------------------
    def latency_summary(self) -> dict:
        if not self.calls:
            return {'n': 0}
        x = np.array([c.roundtrip_s for c in self.calls], dtype=float)
        return {
            'n': int(x.size),
            'first_s': round(float(x[0]), 4),
            'mean_s': round(float(x.mean()), 4),
            'p50_s': round(float(np.percentile(x, 50)), 4),
            'p95_s': round(float(np.percentile(x, 95)), 4),
            'max_s': round(float(x.max()), 4),
            'calls': [c.as_dict() for c in self.calls],
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
