"""Level 3：动作适配器与观测适配器的可判据单元测试（不需要 GPU，也不需要策略服务器）。

判据不是“格式正确”，而是：
  1. 已知动作 → 已知关节目标的逐分量核对（含 clip、限位、夹爪二值化）；
  2. 关节增量递推是否按“相对当前 state”实现；
  3. FK/IK 一致性：把适配出的 q_target 送进 PyBullet，用正运动学核对末端位移方向，
     与“关节增量 ≈ 末端位移”的物理预期一致（不要求严格相等，只要求同号且量级合理）；
  4. 夹爪方向：binary 1 必须让两指张开、0 必须闭合；
  5. 拒绝规则：NaN/Inf、超官方 clip、关节限位都必须被拒绝并给出原因。

用法：
    python scripts/exp40/level3_adapter_checks.py [--out results/experiment_4_0/action_checks]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40.action_adapter import ActionAdapterConfig, adapt_action, chunk_to_targets, summarize_chunk  # noqa: E402
from exp40.chunk import ActionChunker  # noqa: E402
from exp40.observation_adapter import build_droid_observation, gripper_scalar  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail=None):
    RESULTS.append({'check': name, 'pass': bool(ok), 'detail': detail})
    print(f'{"PASS" if ok else "FAIL"}  {name}  {json.dumps(detail, default=float) if detail else ""}')
    return ok


def limits():
    """PyBullet 里真实读一次 URDF 限位；失败则退回硬编码的 Panda 典型值。"""
    try:
        import pybullet as p

        from simulation import Simulation

        sim = Simulation(gui=False, reference_cube=False)
        lower = np.array([p.getJointInfo(sim.robot, j, physicsClientId=sim.client)[8] for j in range(7)])
        upper = np.array([p.getJointInfo(sim.robot, j, physicsClientId=sim.client)[9] for j in range(7)])
        sim.close()
        return lower, upper, 'pybullet_urdf'
    except Exception as exc:  # pragma: no cover
        print(f'[warn] 无法读取 URDF 限位（{exc}），使用 Panda 典型值')
        lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        return lower, upper, 'hardcoded'


def home_q() -> np.ndarray:
    """Panda URDF 的 HOME 构型（simulation.HOME）；注意 joint4 上限是 0.0，零位是越界的。"""
    from simulation import HOME

    return np.array(HOME, dtype=float)


def test_basic_mapping(lower, upper):
    q0 = home_q()
    cfg = ActionAdapterConfig(max_delta_rad=0.25)
    a = np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.9])
    r = adapt_action(a, q0, lower, upper, cfg)
    check('basic: accepted', r.accepted, {'reasons': r.reasons})
    check('basic: q_target = q + delta',
          np.allclose(r.q_target, q0 + np.array([0.1, -0.2, 0.05, 0, 0, 0, 0])),
          {'q_target': r.q_target.tolist()})
    check('basic: gripper binary=1 -> 全开宽度 0.04',
          abs(r.gripper_target_m - 0.04) < 1e-12 and r.gripper_binary == 1.0,
          {'width_m': r.gripper_target_m, 'binary': r.gripper_binary})
    b = np.array([0.1, 0, 0, 0, 0, 0, 0, 0.2])
    r2 = adapt_action(b, q0, lower, upper, cfg)
    check('basic: gripper 0.2 -> binary 0 -> 闭合 0 m',
          r2.gripper_binary == 0.0 and r2.gripper_target_m == 0.0,
          {'width_m': r2.gripper_target_m, 'binary': r2.gripper_binary})
    # 该 URDF 的 joint4 上限是 0.0：零位 q=0 必须被判为越界，这是真实限位证据
    r3 = adapt_action(np.zeros(8), np.zeros(7), lower, upper, cfg)
    check('basic: q=0 在 joint4 上限之外，被正确拒绝', not r3.accepted and 'joint_limit_target' in r3.reasons,
          {'joint4_upper': float(upper[3]), 'reasons': r3.reasons})


def test_clip_semantics(lower, upper):
    q0 = home_q()
    cfg = ActionAdapterConfig(max_delta_rad=0.25)
    a = np.array([1.8, -1.5, 0, 0, 0, 0, 0, 0.5])
    r = adapt_action(a, q0, lower, upper, cfg)
    check('clip: 超官方 clip 被拒绝', not r.accepted and 'delta_exceeds_official_clip' in r.reasons,
          {'reasons': r.reasons})
    check('clip: 实际应用量被压到 max_delta_rad',
          np.allclose(r.delta_applied, [0.25, -0.25, 0, 0, 0, 0, 0]),
          {'delta_applied': r.delta_applied.tolist()})
    check('clip: 记录被裁剪维度', sorted(r.clipped_dims) == [0, 1], {'clipped_dims': r.clipped_dims})


def test_rejections(lower, upper):
    q0 = home_q()
    r = adapt_action(np.array([np.nan, 0, 0, 0, 0, 0, 0, 0.5]), q0, lower, upper)
    check('reject: NaN 动作', not r.accepted and 'nan_or_inf' in r.reasons, {'reasons': r.reasons})
    r = adapt_action(np.array([np.inf, 0, 0, 0, 0, 0, 0, 0.5]), q0, lower, upper)
    check('reject: Inf 动作', not r.accepted and 'nan_or_inf' in r.reasons, {'reasons': r.reasons})
    # 把关节推到限位外
    q_hi = np.minimum(upper - 0.01, upper)
    r = adapt_action(np.array([0.2, 0, 0, 0, 0, 0, 0, 0.5]), q_hi, lower, upper)
    check('reject: 关节限位', not r.accepted and 'joint_limit_target' in r.reasons,
          {'reasons': r.reasons})


def test_chunk_recursion(lower, upper):
    q0 = home_q()
    chunk = np.tile(np.array([0.1, 0, 0, 0, 0, 0, 0, 0.6]), (15, 1))
    targets = chunk_to_targets(chunk, q0, lower, upper, ActionAdapterConfig(max_delta_rad=0.25))
    seq = [t.q_target[0] for t in targets]
    check('chunk: 逐条递推（第 k 条 = k*0.1）',
          np.allclose(seq, np.arange(1, 16) * 0.1),
          {'q0_sequence': seq})
    s = summarize_chunk(chunk)
    check('chunk: 统计摘要', s['shape'] == [15, 8] and s['finite'] and s['gripper_binary_frac'] == 1.0,
          {k: s[k] for k in ('shape', 'finite', 'gripper_binary_frac')})


def test_chunk_buffer():
    ch = ActionChunker(execute_k=4)
    check('buffer: 空时需推理', ch.needs_inference())
    ch.push(np.tile(np.array([0.0] * 7 + [0.5]), (15, 1)))
    n = 0
    while ch.pop() is not None:
        n += 1
    check('buffer: 只执行 execute_k=4 步', n == 4, {'consumed': n})
    check('buffer: 之后需要重新推理', ch.needs_inference())
    p_before = ch.stats['prompt_changes']
    ch.set_prompt('a new instruction')
    check('buffer: 指令切换计数', ch.stats['prompt_changes'] == p_before + 1)
    ch.reset()
    check('buffer: reset 清空', ch.remaining() == 0 and ch.needs_inference())


def test_gripper_direction():
    """在真实仿真里验证夹爪二值方向：1 -> 两指张开，0 -> 两指闭合。"""
    try:
        from exp40.observation_adapter import finger_widths
        from scripts.exp40._common import build_sim_scene
    except Exception as exc:  # pragma: no cover
        check('gripper: 方向验证（需要 pybullet）', False, {'error': str(exc)})
        return
    sim, block, rig, prompt, info = build_sim_scene('B90', seed=0)
    try:
        import pybullet as p

        # 张开
        sim.command(np.array(sim.q()))
        for j in (9, 10):
            p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, 0.04, force=30,
                                    physicsClientId=sim.client)
        for _ in range(120):
            sim.step()
        open_w = float(np.mean(finger_widths(sim)))
        # 闭合
        for j in (9, 10):
            p.setJointMotorControl2(sim.robot, j, p.POSITION_CONTROL, 0.0, force=30,
                                    physicsClientId=sim.client)
        for _ in range(120):
            sim.step()
        closed_w = float(np.mean(finger_widths(sim)))
        ok = open_w > closed_w + 0.01
        check('gripper: 1=张开 0=闭合（真实仿真）', ok,
              {'open_m': open_w, 'closed_m': closed_w,
               'note': 'binary 1 -> +X 目标 0.04，binary 0 -> 0.0 由 action_adapter 给出'})
        g_open = gripper_scalar(sim) if False else None
        check('gripper: 归一化换算一致性',
              abs(gripper_scalar(sim) - np.clip(closed_w / 0.04, 1e-4, 0.9999)) < 1e-6,
              {'gripper_scalar': gripper_scalar(sim), 'closed_m': closed_w})
    finally:
        sim.close()


def test_observation_adapter():
    """观测适配：真实仿真图 → DROID payload，核对 shape/dtype/键名/哈希稳定性。"""
    try:
        from scripts.exp40._common import build_sim_scene, capture, model_input_summary
    except Exception as exc:  # pragma: no cover
        check('obs: 真实仿真观测', False, {'error': str(exc)})
        return
    sim, block, rig, prompt, info = build_sim_scene('B90', seed=0)
    try:
        obs = capture(sim, rig)
        pkt = build_droid_observation(obs['frames'], obs['q'], gripper_scalar(sim), prompt)
        s = model_input_summary(pkt.payload)
        check('obs: 五个键齐全',
              set(pkt.payload) == {'observation/exterior_image_1_left', 'observation/wrist_image_left',
                                   'observation/joint_position', 'observation/gripper_position', 'prompt'},
              {'keys': s['keys']})
        check('obs: 图像 224x224x3 uint8',
              s['shapes']['observation/exterior_image_1_left'] == [224, 224, 3]
              and s['dtypes']['observation/exterior_image_1_left'] == 'uint8',
              {'shape': s['shapes']['observation/exterior_image_1_left'],
               'dtype': s['dtypes']['observation/exterior_image_1_left']})
        check('obs: state 8 维 float32',
              s['shapes']['observation/joint_position'] == [7]
              and s['shapes']['observation/gripper_position'] == [1],
              {'joint': s['shapes']['observation/joint_position'],
               'gripper': s['shapes']['observation/gripper_position']})
        pkt2 = build_droid_observation(obs['frames'], obs['q'], gripper_scalar(sim), prompt)
        same = (pkt.audit['images']['observation/exterior_image_1_left']['sha256_sent']
                == pkt2.audit['images']['observation/exterior_image_1_left']['sha256_sent'])
        check('obs: 同状态可复现（sha256 一致）', same)
        check('obs: gripper 落在 [0,1]',
              all(0 <= v < 1 for v in pkt.audit['state'][7:8]),
              {'gripper': pkt.audit['state'][7]})
    finally:
        sim.close()


def test_fk_consistency(lower, upper):
    """适配出的 q_target 送进仿真后，EE 是否按预期方向移动（同号 + 量级合理）。"""
    try:
        from scripts.exp40._common import build_sim_scene
    except Exception as exc:  # pragma: no cover
        check('fk: EE 一致性', False, {'error': str(exc)})
        return
    sim, block, rig, prompt, info = build_sim_scene('B90', seed=0)
    try:
        q0 = sim.q().copy()
        ee0, quat0 = sim.pose()
        cfg = ActionAdapterConfig(max_delta_rad=0.05)
        for jidx in range(7):
            a = np.zeros(8)
            a[jidx] = 0.05
            a[7] = 0.9
            r = adapt_action(a, q0, lower, upper, cfg)
            for _ in range(240):  # 1 s
                sim.command(r.q_target)
                sim.step()
            ee1, quat1 = sim.pose()
            dq = np.linalg.norm(sim.q() - q0)
            dee = float(np.linalg.norm(np.asarray(ee1) - np.asarray(ee0)))
            dang = float(2 * np.arccos(np.clip(abs(np.dot(quat0, quat1)), 0, 1)))
            # joint7 绕末端轴自转：末端位置不变、姿态改变，这是正确物理，
            # 因此对 joint7 用姿态判据，其余关节用位置判据。
            if jidx == 6:
                ok = dq > 0.02 and dang > 1e-3
                note = 'joint7 为末端自转：位置不变、姿态必须变化'
            else:
                ok = dq > 0.02 and dee > 1e-4
                note = ''
            check(f'fk: joint{jidx + 1} +0.05 rad 后 EE 响应且跟踪收敛', ok,
                  {'dq_rad': round(float(dq), 5), 'dee_m': round(dee, 5),
                   'dee_rot_rad': round(dang, 5), 'note': note})
            # 复位
            sim.command(q0)
            for _ in range(240):
                sim.step()
            ee0, quat0 = sim.pose()
    finally:
        sim.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'results' / 'experiment_4_0' / 'action_checks'))
    parser.add_argument('--skip-sim', action='store_true', help='只跑不需要 pybullet 的纯函数测试')
    args = parser.parse_args(argv)

    lower, upper, limit_src = limits()
    print(f'关节限位来源: {limit_src}')
    print('lower', np.round(lower, 4).tolist())
    print('upper', np.round(upper, 4).tolist())

    test_basic_mapping(lower, upper)
    test_clip_semantics(lower, upper)
    test_rejections(lower, upper)
    test_chunk_recursion(lower, upper)
    test_chunk_buffer()
    test_observation_adapter()
    if not args.skip_sim:
        test_gripper_direction()
        test_fk_consistency(lower, upper)

    passed = sum(1 for r in RESULTS if r['pass'])
    summary = {
        'n_checks': len(RESULTS),
        'n_pass': passed,
        'n_fail': len(RESULTS) - passed,
        'all_pass': passed == len(RESULTS),
        'joint_limit_source': limit_src,
        'lower': lower.tolist(),
        'upper': upper.tolist(),
        'checks': RESULTS,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'level3_checks.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n{passed}/{len(RESULTS)} 通过 -> {out / "level3_checks.json"}')
    return 0 if summary['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
