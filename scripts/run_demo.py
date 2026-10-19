"""四种技能交接的实时可视化，不写入数据或截图。"""
from pathlib import Path
import argparse
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulation import Simulation, HOME
from robot_controller import RobotController
from skills import Z1, Z2, B, C
import pybullet as p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='每个后继动作前先回到 HOME')
    args = parser.parse_args()
    sim = Simulation(gui=True, realtime=True)
    label = -1

    def status(text, color=(0, 0, 0)):
        nonlocal label
        label = p.addUserDebugText(
            text, [0, 0, 1.12], textColorRGB=color, textSize=1.3,
            replaceItemUniqueId=label, physicsClientId=sim.client)

    cases = [
        ('z1->B', Z1, B), ('z1->C', Z1, C),
        ('z2->B', Z2, B), ('z2->C', Z2, C),
    ]
    try:
        for index, (name, first, successor) in enumerate(cases, start=1):
            sim.reset()
            label = -1
            ctl = RobotController(sim)
            title = f'[{index}/4] {name}'
            mode = 'via HOME' if args.reset else 'direct'
            status(f'{title} ({mode}) | A: {first.name}')
            print(f'{title}: 执行 A ({first.name})')
            first_result = ctl.move_pose(first.position, first.orientation)
            print(f'  A: {first_result.success} ({first_result.reason})')
            status(f'{title} | A: {"OK" if first_result.success else first_result.reason}')
            time.sleep(1.5)
            if not first_result.success:
                status(f'{title} | STOP: A failed', (0.8, 0, 0))
                time.sleep(3)
                continue
            if args.reset:
                status(f'{title} | Recovery: HOME')
                print('  恢复: HOME')
                recovery = ctl.move(HOME)
                print(f'  恢复: {recovery.success} ({recovery.reason})')
                if not recovery.success:
                    status(f'{title} | STOP: {recovery.reason}', (0.8, 0, 0))
                    time.sleep(3)
                    continue
                time.sleep(1)
            status(f'{title} | Successor: {successor.name}')
            print(f'  执行后继 ({successor.name})')
            result = ctl.move_pose(successor.position, successor.orientation)
            print(f'  后继: {result.success} ({result.reason})')
            color = (0, 0.6, 0) if result.success else (0.8, 0, 0)
            status(f'{title} | {"SUCCESS" if result.success else "FAILED: " + result.reason}', color)
            if result.reason == 'ik_joint_limits':
                print('  IK 候选超出关节限位，拒绝执行；机械臂保持 A 终态，不是窗口卡住。')
            print('  当前画面保持 3 秒。')
            time.sleep(3)
        print('四种情况播放完毕，窗口保持 5 秒。')
        time.sleep(5)
    finally:
        sim.close()

if __name__ == '__main__':
    main()
