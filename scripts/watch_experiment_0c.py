"""实验 0C：在 PyBullet 实时窗口中观看真实接触抓取-移动-放置。

全部动作与离线实验使用同一个 GraspPlaceTask，因此画面中的夹持、抬升、
搬运和释放就是实验记录里的同一段物理。观察者（HUD/暂停）不参与控制。

按键: SPACE 开始, P 暂停/继续, ESC 退出。
"""
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from simulation import Simulation
from robot_controller import RobotController
from manipulation import (create_block, enable_contact_solver, finger_contact_forces,
                          finger_widths, object_state)
from grasp_task import DEFAULT_PARAMS, GraspPlaceTask

STAGE_CN = {
    'open_gripper': '张开夹爪', 'approach_pre': '接近(高位)', 'preclose': '预闭合',
    'approach': '下降到抓取位', 'close': '闭合夹紧', 'lift': '抬升 10cm',
    'transport': '移动到放置点', 'lower': '下降', 'release': '张开释放',
    'retreat': '撤离', 'settle': '静置检查',
}


class Hud:
    def __init__(self, sim):
        self.sim = sim
        self.ids = {}

    def show(self, stage, seconds, extra):
        lines = [
            f'阶段: {STAGE_CN.get(stage, stage)} ({stage})',
            f'仿真时间: {seconds:5.2f} s',
        ] + extra + ['P 暂停/继续 | ESC 退出']
        for i, text in enumerate(lines):
            self.ids[i] = p.addUserDebugText(
                text, [0.16, -0.30, 0.62 - 0.045 * i], textColorRGB=[0.05, 0.05, 0.05],
                textSize=1.25, lifeTime=0, replaceItemUniqueId=self.ids.get(i, -1),
                physicsClientId=self.sim.client)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=float, default=1.0, help='回放倍速（1=实时）')
    parser.add_argument('--hold', type=float, default=3.0, help='结束后停留秒数')
    parser.add_argument('--auto-exit', action='store_true', help='结束后自动关闭窗口')
    parser.add_argument('--no-wait', action='store_true', help='跳过开场的 SPACE 等待')
    parser.add_argument('--friction', type=float, default=None,
                        help='工件摩擦系数（默认 1.2；给 0.2 可复现摩擦不足的失败）')
    parser.add_argument('--force', type=float, default=None,
                        help='每指夹持力上限 N（默认 20；给 2 可复现夹持不足的失败）')
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.friction is not None:
        params['block_friction'] = float(args.friction)
    if args.force is not None:
        params['grasp_force'] = float(args.force)
    weight = params['block_mass'] * 9.81
    friction_cap = 2.0 * params['block_friction'] * params['grasp_force']

    sim = Simulation(gui=True, realtime=True, speed=args.speed, reference_cube=False)
    enable_contact_solver(sim)
    block = create_block(sim, params['block_half'], params['block_mass'],
                         params['block_start'], lateral_friction=params['block_friction'])
    sim.allowed_contacts = {(block, 9), (block, 10)}
    p.resetDebugVisualizerCamera(1.0, 55, -28, [0.47, 0.07, 0.13], physicsClientId=sim.client)

    hud = Hud(sim)
    status = {'paused': False}

    def on_step(stage, k, n):
        keys = p.getKeyboardEvents(physicsClientId=sim.client)
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            sim.close()
            raise SystemExit
        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            status['paused'] = not status['paused']
        while status['paused']:
            hud.show(stage, sim.steps * sim.dt, ['已暂停（P 继续）'])
            time.sleep(0.05)
            keys = p.getKeyboardEvents(physicsClientId=sim.client)
            if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
                status['paused'] = False
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                sim.close()
                raise SystemExit
        if sim.steps % 10 == 0:
            state = object_state(sim, block)
            f1, f2, active = finger_contact_forces(sim, block)
            widths = finger_widths(sim)
            hud.show(stage, sim.steps * sim.dt, [
                f'夹爪手指: {widths[0]:.4f} / {widths[1]:.4f} m',
                f'接触力: {f1:4.1f} N / {f2:4.1f} N  接触指: {len(active)}',
                f'工件: z={state["position"][2]:.3f} m  |v|={state["linear_speed"]:.3f} 倾角={state["tilt_deg"]:.1f}°',
                f'摩擦上限 2μF = {friction_cap:.2f} N  重力 = {weight:.2f} N  '
                f'{"够" if friction_cap > weight else "不够"}',
            ])

    if not args.no_wait:
        hud.show('ready', 0.0, [
            f'本轮参数: 工件摩擦 μ={params["block_friction"]}  每指夹持力 {params["grasp_force"]:.1f} N',
            f'摩擦上限 2μF={friction_cap:.2f} N vs 重力 {weight:.2f} N',
            '按 SPACE 开始（首次进入时点击窗口以获取按键焦点）',
        ])
        while True:
            keys = p.getKeyboardEvents(physicsClientId=sim.client)
            if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
                break
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                sim.close()
                return
            time.sleep(0.05)

    task = GraspPlaceTask(sim, RobotController(sim), block, on_step=on_step, params=params)
    task.run()
    result = task.result()
    status['result'] = result
    print(f"mu={params['block_friction']} force={params['grasp_force']}N "
          f"success={result['success']} failure={result['failure']} "
          f"lift={result['lift_height_m']:.3f} m slip={result['slip_m'] * 1000:.1f} mm")

    final_lines = [
        f'参数: 工件摩擦 μ={params["block_friction"]}  每指夹持力={params["grasp_force"]:.1f} N',
        f'摩擦上限 2μF={friction_cap:.2f} N vs 重力 {weight:.2f} N',
        f'结果: {"成功" if result["success"] else "失败"}  {result["failure_text"] or ""}',
        f'抬升 {result["lift_height_m"] * 100:.1f} cm  滑移 {result["slip_m"] * 1000:.1f} mm',
        '按 ESC 或关闭窗口退出',
    ]
    hud.show('done', result['sim_seconds'], final_lines)

    if args.auto_exit:
        sim.close()
        return
    start = time.time()
    while time.time() - start < args.hold:
        keys = p.getKeyboardEvents(physicsClientId=sim.client)
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.05)
    sim.close()


if __name__ == '__main__':
    main()
