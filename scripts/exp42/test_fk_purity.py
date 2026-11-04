"""FK 纯净性回归测试（在 simulate venv 里跑）：

1. 纯 FK 不推进物理：自由落体方块在 N 次 FK 调用后 z 不变；
2. FK 顺序无关：fk(a) -> fk(b) -> fk(a) 两次 fk(a) 位姿逐位相同；
3. FK 结果与 pybullet 的 linkState 一致（同一 q 下与 sim.pose() 对齐）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EE_LINK = 11


def main() -> int:
    import pybullet as p

    from exp41 import scene41

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.40), obj_yaw=np.pi / 2, robot_q=None)
    sim = scene.sim

    def fk(q):
        for j in range(7):
            p.resetJointState(sim.robot, j, float(q[j]), targetVelocity=0.0,
                              physicsClientId=sim.client)
        link = p.getLinkState(sim.robot, EE_LINK, computeForwardKinematics=True,
                              physicsClientId=sim.client)
        return np.asarray(link[4], dtype=float)

    z0 = float(scene.object_state()['pos'][2])
    steps0 = int(getattr(sim, 'steps', -1))
    qa = np.asarray(sim.q(), dtype=float)
    qb = qa + 0.05
    for _ in range(10):
        fk(qa)
    steps1 = int(getattr(sim, 'steps', -1))
    z1 = float(scene.object_state()['pos'][2])
    pose_a1 = fk(qa)
    fk(qb)
    pose_a2 = fk(qa)
    order_ok = bool(np.array_equal(pose_a1, pose_a2))
    no_step = (steps1 == steps0) and abs(z1 - z0) < 1e-9

    # 与 sim.pose() 对照（同一 q，静止 scene）
    sim.command(qa)
    for _ in range(60):
        sim.step()
    sim_pose, _ = sim.pose()
    fk_pose = fk(np.asarray(sim.q(), dtype=float))
    agree = float(np.abs(sim_pose - fk_pose).max())

    print(f'sim.steps before/after 10 FK calls: {steps0} -> {steps1}  '
          f'(object z drift {z1 - z0:.3e} m)  -> {"PASS" if no_step else "FAIL"}')
    print(f'FK order independence: {"PASS" if order_ok else "FAIL"}')
    print(f'FK vs sim.pose agreement: {agree:.3e} m')
    scene.close()
    ok = no_step and order_ok and agree < 1e-6
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
