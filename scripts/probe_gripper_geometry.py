"""量一下夹爪几何：不同末端高度下指尖、手掌包围盒的实际 z。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pybullet as p  # noqa: E402

from manipulation import finger_tips  # noqa: E402
from robot_controller import RobotController  # noqa: E402
from vla.atomic_tasks.scene2 import build_scene, grasp_quat  # noqa: E402


def main():
    sim, block, wall, tray, info = build_scene()
    try:
        ctl = RobotController(sim)
        ctl.move_pose([0.52, 0.05, 0.15], grasp_quat(0.0))
        for z in (0.040, 0.031, 0.028, 0.024, 0.020, 0.016):
            ctl.move_pose([0.52, 0.05, z], grasp_quat(0.0))
            ee, _ = sim.pose()
            tips = finger_tips(sim)
            aabb = p.getAABB(sim.robot, 8, physicsClientId=sim.client)
            a9 = p.getAABB(sim.robot, 9, physicsClientId=sim.client)
            a10 = p.getAABB(sim.robot, 10, physicsClientId=sim.client)
            print(f'z={z:.3f} ee_z={ee[2]:.4f} '
                  f'tip_z={[round(t[2], 4) for t in tips]} '
                  f'hand_min_z={aabb[0][2]:.4f} '
                  f'finger_min_z={a9[0][2]:.4f}/{a10[0][2]:.4f} '
                  f'finger_max_z={a9[1][2]:.4f} '
                  f'finger_x=[{a9[0][0]:.4f},{a9[1][0]:.4f}] '
                  f'hand_x=[{aabb[0][0]:.4f},{aabb[1][0]:.4f}]')
    finally:
        sim.close()


if __name__ == '__main__':
    main()
