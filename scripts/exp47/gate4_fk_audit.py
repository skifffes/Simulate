"""实验 4.7 Gate 4：FK/EE 坐标系审计（解释实验 4.6 的 ~0.33 m 偏差）。

对 20 个合法关节构型，比较：
  1. scene.sim.pose()（其内部使用 self.ee link）
  2. 同一 link 的 getLinkState
  3. FK 用 link 11（panda_grasptarget，实验 4.6 用过的）
  4. link 11 的 getLinkState（真实物理状态）
输出 fk_consistency.csv 与 fk_frame_audit.md 的数值部分。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/scratch/lijh/exp40/repo')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))
OUT = REPO / 'results' / 'experiment_4_7'


def main() -> int:
    import pybullet as p

    from exp41 import scene41

    scene = scene41.Scene41('B90', obj_pos=(0.55, 0.21, 0.0201), obj_yaw=np.pi / 2, robot_q=None)
    sim = scene.sim
    ee_idx = int(sim.ee)
    names = {}
    for i in range(p.getNumJoints(sim.robot, physicsClientId=sim.client)):
        info = p.getJointInfo(sim.robot, i, physicsClientId=sim.client)
        names[i] = info[12].decode() if isinstance(info[12], bytes) else str(info[12])
    rng = np.random.default_rng(0)
    lower = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.1, -2.8])
    upper = np.array([2.8, 1.7, 2.8, -0.1, 2.8, 3.6, 2.8])
    rows = []
    for k in range(20):
        q = 0.5 * (lower + upper) + 0.35 * (upper - lower) * rng.uniform(-1, 1, 7)
        q = np.clip(q, lower, upper)
        for j in range(7):
            p.resetJointState(sim.robot, j, float(q[j]), targetVelocity=0.0,
                              physicsClientId=sim.client)
        pose_pos, pose_quat = sim.pose()
        ls_ee = p.getLinkState(sim.robot, ee_idx, computeForwardKinematics=True,
                               physicsClientId=sim.client)
        ls_11 = p.getLinkState(sim.robot, 11, computeForwardKinematics=True,
                               physicsClientId=sim.client)
        rows.append({'config': k, 'q': ';'.join(f'{v:.5f}' for v in q),
                     'ee_link_index': ee_idx, 'ee_link_name': names.get(ee_idx, ''),
                     'pose_pos': ';'.join(f'{v:.5f}' for v in pose_pos),
                     'getlink_ee_pos': ';'.join(f'{v:.5f}' for v in ls_ee[4]),
                     'pose_vs_getlink_ee': float(np.linalg.norm(pose_pos - ls_ee[4])),
                     'getlink_11_pos': ';'.join(f'{v:.5f}' for v in ls_11[4]),
                     'pose_vs_link11': float(np.linalg.norm(pose_pos - np.asarray(ls_11[4])))})
    with open(OUT / 'fk_consistency.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    d1 = np.array([r['pose_vs_getlink_ee'] for r in rows])
    d2 = np.array([r['pose_vs_link11'] for r in rows])
    summary = {'ee_link_index': ee_idx, 'ee_link_name': names.get(ee_idx, ''),
               'link_names': names,
               'pose_vs_same_link_max_m': float(d1.max()),
               'pose_vs_link11_max_m': float(d2.max()),
               'pose_vs_link11_mean_m': float(d2.mean()),
               'n_configs': len(rows)}
    (OUT / 'fk_frame_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'link_names'},
                     ensure_ascii=False, indent=2))
    print('links:', {i: names[i] for i in range(min(13, len(names)))})
    scene.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
