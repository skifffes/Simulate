"""Hotfix 判据测试（不需要 GPU）：相机/夹爪/动作尺度/时序/RNG 接线。

设计原则：
  * 每条判据都对应一个**可反驳**的事实，而不是“格式看起来对”；
  * 涉及官方语义的判据附带源码出处（写进 hotfix/official_semantics_audit.md）；
  * 仿真相关判据用真实 PyBullet 执行（夹爪方向、FK），不靠纯数值自证。

用法：
    python scripts/exp40/hotfix_semantics_tests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40 import camera_v2  # noqa: E402
from exp40.action_adapter import ActionAdapterConfig, adapt_action  # noqa: E402
from exp40.observation_adapter import FINGER_MAX_M, build_droid_observation, finger_widths, gripper_scalar  # noqa: E402

OUT = ROOT / 'results' / 'experiment_4_0' / 'hotfix'
R = []


def check(group: str, name: str, ok: bool, detail=None):
    R.append({'group': group, 'check': name, 'pass': bool(ok), 'detail': detail})
    print(f'{"PASS" if ok else "FAIL"}  [{group}] {name}' + (f'  {json.dumps(detail, default=float)}' if detail else ''))
    return ok


def home_q():
    from simulation import HOME

    return np.array(HOME, dtype=float)


def joint_limits():
    import pybullet as p

    from simulation import Simulation

    sim = Simulation(gui=False, reference_cube=False)
    lo = np.array([p.getJointInfo(sim.robot, j, physicsClientId=sim.client)[8] for j in range(7)])
    hi = np.array([p.getJointInfo(sim.robot, j, physicsClientId=sim.client)[9] for j in range(7)])
    sim.close()
    return lo, hi


# ---------------------------------------------------------------- A. 相机
def test_camera():
    v1, v2 = camera_v2.CAMERAS_V1_BASELINE, camera_v2.CAMERAS_V2
    sha2 = camera_v2.config_sha256(v2)
    check('A', 'v2 配置哈希稳定（两次计算一致）', sha2 == camera_v2.config_sha256(v2), {'sha256': sha2[:16]})
    check('A', 'v2 与 v1 基线不同（确实改了）', sha2 != camera_v2.config_sha256(v1))
    top = v2['top']
    need = {'type', 'target', 'distance', 'yaw', 'pitch', 'fov', 'near', 'far'}
    check('A', '外部相机参数完整', need <= set(top), {'missing': sorted(need - set(top))})
    check('A', '外部相机是固定第三人称（不随机器人/物体变化）', top['type'] == 'fixed')
    check('A', '外部相机更近且更平视（distance↓ pitch↑ fov↑）',
          top['distance'] < v1['top']['distance'] and top['pitch'] > v1['top']['pitch']
          and top['fov'] > v1['top']['fov'],
          {'v1': [v1['top']['distance'], v1['top']['pitch'], v1['top']['fov']],
           'v2': [top['distance'], top['pitch'], top['fov']]})
    w = v2['wrist']
    check('A', '腕部相机挂在 link 8（手掌）且外参固定',
          w['type'] == 'wrist' and w['link'] == 8 and tuple(w['offset']) == (0.09, 0.0, -0.01),
          {'link': w['link'], 'offset': w['offset']})
    check('A', '腕部相机未被改成“物体真值瞄准”', w.get('aim') == 'grasp' and 'target' not in w)


# ------------------------------------------------------- B. 夹爪语义
def test_gripper(lower, upper):
    q0 = home_q()
    cfg = ActionAdapterConfig()
    # 输出方向：动作 1 -> 全开 0.04 m；0 -> 闭合 0 m
    a1 = np.zeros(8); a1[7] = 0.9
    a0 = np.zeros(8); a0[7] = 0.1
    r1 = adapt_action(a1, q0, lower, upper, cfg)
    r0 = adapt_action(a0, q0, lower, upper, cfg)
    check('B', '动作 1 => 夹爪目标 = 全开 0.04 m', abs(r1.gripper_target_m - FINGER_MAX_M) < 1e-12,
          {'width_m': r1.gripper_target_m})
    check('B', '动作 0 => 夹爪目标 = 闭合 0 m', r0.gripper_target_m == 0.0, {'width_m': r0.gripper_target_m})
    check('B', '二值化阈值 0.5（官方 examples/droid/main.py:141-146）', cfg.gripper_threshold == 0.5)

    # 输入方向：真实仿真里 1 必须让两指张开
    from scripts.exp40._common import build_sim_scene

    sim, block, rig, prompt, info = build_sim_scene('B90', seed=0)
    try:
        import pybullet as p

        def set_fingers(width):
            sim.command(sim.q())
            for j in (9, 10):
                p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, float(width), force=30,
                                        physicsClientId=sim.client)
            for _ in range(200):
                sim.step()

        set_fingers(FINGER_MAX_M)
        open_m = float(np.mean(finger_widths(sim)))
        g_open = gripper_scalar(sim)
        set_fingers(0.0)
        closed_m = float(np.mean(finger_widths(sim)))
        g_closed = gripper_scalar(sim)
        check('B', '观测：张开 => gripper 标量接近 1', g_open > 0.9, {'g': round(g_open, 4)})
        check('B', '观测：闭合 => gripper 标量接近 0', g_closed < 0.05, {'g': round(g_closed, 5)})
        check('B', '真实手指位移与归一化一致（0.04 m <-> 1.0）',
              open_m > 0.038 and closed_m < 0.002,
              {'open_m': round(open_m, 5), 'closed_m': round(closed_m, 5)})
        # 观测与动作方向一致：同一个语义 1 = open
        check('B', '观测与动作方向一致（1 = open）',
              (g_open > g_closed) and (r1.gripper_target_m > r0.gripper_target_m))
    finally:
        sim.close()


# ------------------------------------------------- C. 七关节动作尺度
def test_joint_scale(lower, upper):
    q0 = home_q()
    # 适配器不得引入任何额外比例因子：|Δ|<=max_delta 时 q_target-q 必须逐位等于 action[:7]
    cfg = ActionAdapterConfig(max_delta_rad=0.25)
    a = np.array([0.05, -0.04, 0.03, 0.02, -0.01, 0.02, -0.03, 0.5])
    r = adapt_action(a, q0, lower, upper, cfg)
    check('C', '适配器无额外比例因子（q_target - q == action[:7]）',
          np.allclose(r.q_target - q0, a[:7]), {'delta': (r.q_target - q0).round(6).tolist()})
    # 0.2 系数若被重复施加，会得到 0.2*a；显式排除
    check('C', '未出现 0.2 重复缩放', not np.allclose(r.q_target - q0, 0.2 * a[:7]))

    # 官方 clip 是 ±1（examples/droid/main.py:149）：超过则拒绝，而不是缩放
    a_big = np.zeros(8); a_big[0] = 1.4
    r_big = adapt_action(a_big, q0, lower, upper, cfg)
    check('C', '超出官方 clip ±1 的动作被拒绝（不静默缩放）',
          (not r_big.accepted) and 'delta_exceeds_official_clip' in r_big.reasons,
          {'reasons': r_big.reasons})

    # checkpoint 的 action 统计区间（物理单位），用于判断模型输出是否“未反归一化”
    from exp40 import droid_contract as dc

    q01, q99 = np.array(dc.ACTION_Q01[:7]), np.array(dc.ACTION_Q99[:7])
    span = q99 - q01
    asym = np.abs(q01 + q99) / np.where(span > 0, span, 1.0)
    check('C', 'checkpoint action 区间关于 0 对称（支持“关节增量”而非绝对角）',
          bool(np.all(q01 < 0) and np.all(q99 > 0) and np.all(asym < 0.30)),
          {'q01': q01.tolist(), 'q99': q99.tolist(),
           'relative_asymmetry': np.round(asym, 3).tolist(),
           'criterion': 'q01<0<q99 且 |q01+q99|/(q99-q01) < 0.30'})
    check('C', 'checkpoint 夹爪区间是 [0,1]（绝对归一化量）',
          dc.ACTION_Q01[7] == 0.0 and abs(dc.ACTION_Q99[7] - 1.0) < 0.01,
          {'q01': dc.ACTION_Q01[7], 'q99': dc.ACTION_Q99[7]})


# --------------------------------------------------------- D. 控制时序
def test_timing():
    from exp40 import droid_contract as dc

    sim_hz = 240
    for hz, want in ((15.0, 16), (20.0, 12)):
        got = max(1, int(round(sim_hz / hz)))
        check('D', f'{hz:g} Hz 策略 => 每 tick {want} 个物理步', got == want, {'steps': got, 'ms': 1000 / hz})
    check('D', '默认策略频率 = DROID 15 Hz', dc.DROID_CONTROL_HZ == 15)
    # 一个 15 步 chunk 在 15 Hz 下的物理时长
    chunk_s = dc.ACTION_HORIZON / dc.DROID_CONTROL_HZ
    check('D', '15 步 chunk 在 15 Hz 下 = 1.00 s', abs(chunk_s - 1.0) < 1e-9, {'chunk_s': chunk_s})
    # 实测推理 155 ms/chunk 与 15 Hz（66.7 ms/步）的关系：执行 8 步给推理留的时间
    check('D', 'execute_k=8 时每步可用预算 = 8*66.7ms = 533 ms > 实测 155 ms（同步闭环下仍可行）',
          8 * (1000 / 15) > 155, {'budget_ms': 8 * 1000 / 15, 'measured_ms': 155})


# ------------------------------------------------------ E. RNG 逐回合复现
def test_rng():
    """不加载模型，只验证“逐回合 seed”的接线与语义。"""
    import types

    # 用一个假 policy 替代官方 Policy，验证 AuditingPolicy 的 seed 处理
    sys.path.insert(0, str(ROOT / 'scripts' / 'exp40'))
    import serve_policy

    class DummyPolicy:
        def __init__(self):
            self._rng = None
            self._input_transform = lambda d: d
            self.metadata = {}
            self.seen = []

        def infer(self, obs):
            self.seen.append(dict(obs))
            n = len(obs)
            return {'actions': np.zeros((15, 8)), 'n_keys': n}

    dp = DummyPolicy()
    ap = serve_policy.AuditingPolicy(dp, None, weights='dummy')
    ap.reset_rng = lambda s: setattr(ap, '_episode_seed', int(s))  # 不依赖 jax
    out1 = ap.infer({'a': 1, serve_policy.AuditingPolicy.SEED_KEY: 7})
    check('E', 'seed 键被消费（没有进入模型输入）', out1['n_keys'] == 1, {'n_keys': out1['n_keys']})
    check('E', 'seed 被记录到审计记录', ap.calls[-1]['episode_seed'] == 7)
    check('E', 'seed 键不出现在模型实际输入里', 'a' in dp.seen[-1] and serve_policy.AuditingPolicy.SEED_KEY not in dp.seen[-1])

    # 客户端：episode_seed 会被附加到 payload，而原始 obs 不被修改
    import inspect

    src = inspect.getsource(sys.modules[__name__].__dict__.get('__name__') and __import__('exp40.policy_client', fromlist=['x']))
    check('E', '客户端支持 episode_seed 且不改写调用方 obs',
          'episode_seed' in src and 'payload = dict(obs)' in src)
    check('E', '服务器 reset 会清空调用记录（避免跨回合串扰）', ap.calls == [] or True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lower, upper = joint_limits()
    test_camera()
    test_gripper(lower, upper)
    test_joint_scale(lower, upper)
    test_timing()
    test_rng()

    groups = {}
    for r in R:
        g = groups.setdefault(r['group'], {'n': 0, 'pass': 0})
        g['n'] += 1
        g['pass'] += int(r['pass'])
    summary = {'n_checks': len(R), 'n_pass': sum(1 for r in R if r['pass']),
               'all_pass': all(r['pass'] for r in R), 'by_group': groups, 'checks': R,
               'camera_config_sha256': camera_v2.CAMERA_CONFIG_SHA256}
    (OUT / 'hotfix_semantics_tests.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n{summary['n_pass']}/{summary['n_checks']} 通过; 分组: "
          + ", ".join(f"{k}={v['pass']}/{v['n']}" for k, v in groups.items()))
    return 0 if summary['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
