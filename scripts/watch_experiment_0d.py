#!/usr/bin/env python
"""实验 0D 的实时可视化：在 PyBullet GUI 里按真实时间播放一个实验臂。

交互与 watch_experiment_0b/0c 一致：
    空格  开始 / 结束后重放（重建场景）
     P    暂停 / 继续
    ESC   退出

用法：
    python scripts/watch_experiment_0d.py                 # 默认 z1_B
    python scripts/watch_experiment_0d.py --arm z2_C      # 看几何失败
    python scripts/watch_experiment_0d.py --arm reset_C --speed 2
    python scripts/watch_experiment_0d.py --list
"""
import argparse
import math
import sys
import time
from pathlib import Path

import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from manipulation import (  # noqa: E402
    create_block, enable_contact_solver, finger_contact_forces, finger_widths,
    object_state,
)
from reset_primitives import reset_to_canonical  # noqa: E402
from robot_controller import RobotController  # noqa: E402
from run_experiment_0d import (  # noqa: E402
    ARM_ORDER, ARMS, CAMERA, HANDOVER_YAW, sample_config,
)
from simulation import Simulation  # noqa: E402
from successor_tasks import WALL_Y, PushTask, TopGraspTask, create_wall  # noqa: E402
from two_state_task import (  # noqa: E402
    HALF, HANDOVER, HandoverTask, object_yaw_deg,
)

NOTE = {
    'z1_B': 'A -> z1 -> B 侧推',
    'z2_B': 'A -> z2 -> B 侧推',
    'z1_C': 'A -> z1 -> C 顶抓（可抓）',
    'z2_C': 'A -> z2 -> C 顶抓（10cm 长边 vs 8cm 开口）',
    'reset_B': 'A -> z2 -> 复位 -> B 侧推',
    'reset_C': 'A -> z2 -> 复位 -> C 顶抓',
}

CN = {
    'open_gripper': '张开夹爪', 'approach_pre': '高位接近', 'preclose': '预闭合',
    'approach': '下降到抓取位', 'close': '闭合并夹紧', 'lift': '竖直抬升',
    'rotate': '腕部按 10 度步进转动', 'transport': '平移到交接点',
    'lower': '下放到台面', 'release': '张开释放', 'retreat': '撤离',
    'settle': '静置稳定', 'close_gripper': '夹爪并拢成推头',
    'push': '1cm 步进推动', 'approach_above': '到工件正上方',
    'descend': '下降到抓取高度', 'hold': '保持并观察',
}


class Hud:
    def __init__(self, sim, arm, episode, seed):
        self.sim, self.arm, self.episode, self.seed = sim, arm, episode, seed
        self.block = None
        self.phase, self.phase_key = 'ready', 'A'
        self.stage, self.k, self.n = '-', 0, 0
        self.paused = self.space_hit = False
        self.last_draw, self.hud_id = -10 ** 9, -1
        self.hint = 'SPACE run    P pause    ESC quit'
        self.extra = []

    def poll(self):
        for key, state in p.getKeyboardEvents(physicsClientId=self.sim.client).items():
            if not (state & p.KEY_WAS_TRIGGERED):
                continue
            if key == 27:
                self.sim.close()
                raise SystemExit(0)
            if key in (ord('p'), ord('P')):
                self.paused = not self.paused
                self.draw()
            if key == 32:
                self.space_hit = True

    def on_step(self, stage, k, n):
        if stage != self.stage:
            self.stage = stage
            print(f'   {CN.get(stage, stage):<14} [{stage}] t={self.sim.steps * self.sim.dt:6.2f}s')
        self.k, self.n = k, n
        self.poll()
        if self.paused:
            self.draw()
            while self.paused:
                self.poll()
                time.sleep(0.02)
        if self.sim.steps - self.last_draw >= 24:
            self.draw()

    def draw(self):
        self.last_draw = self.sim.steps
        prog = f' ({self.k}/{self.n})' if self.n else ''
        lines = [f'0D  arm {self.arm}  ep {self.episode}  seed {self.seed}',
                 f'phase {self.phase}',
                 f'stage {self.stage}{prog}',
                 f'sim {self.sim.steps * self.sim.dt:6.2f} s']
        if self.block is not None:
            obj = object_state(self.sim, self.block)
            pos = obj['position']
            yaw = float(object_yaw_deg(self.sim, self.block))
            w = finger_widths(self.sim)
            f1, f2, active = finger_contact_forces(self.sim, self.block)
            wt = '/'.join(f'{x * 1000:.1f}' for x in w)
            lines.append(f'obj {pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}  '
                         f'yaw {yaw:5.1f}  tilt {obj["tilt_deg"]:4.1f}')
            lines.append(f'dz {pos[2] - 0.021:+.3f}m  finger {wt}mm  '
                         f'contact {len(active)}  F {f1:.2f}/{f2:.2f}N')
        if self.extra:
            lines += ['-' * 46] + self.extra
        lines.append(self.hint)
        self.hud_id = p.addUserDebugText(
            '\n'.join(lines), [0.20, 0.00, 0.50], textSize=1.15,
            textColorRGB=[1, 1, 1], replaceItemUniqueId=self.hud_id,
            physicsClientId=self.sim.client)

    def wait(self, hint, timeout=None):
        self.hint = hint
        self.draw()
        self.space_hit = False
        start = time.perf_counter()
        while True:
            self.poll()
            if self.space_hit:
                self.space_hit = False
                return True
            if timeout is not None and time.perf_counter() - start > timeout:
                return False
            time.sleep(0.02)


def mark_scene(sim, target_yaw_deg, successor):
    """在画面里标出交接目标轮廓与挡板，让"计划条件"可见。"""
    yaw = math.radians(target_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [[HANDOVER[0] + c * sx * HALF[0] - s * sy * HALF[1],
            HANDOVER[1] + s * sx * HALF[0] + c * sy * HALF[1], 0.002]
           for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    for i in range(4):
        p.addUserDebugLine(pts[i], pts[(i + 1) % 4], [0.1, 0.95, 0.3], 3,
                           physicsClientId=sim.client)
    p.addUserDebugLine([HANDOVER[0], HANDOVER[1], 0.002],
                       [HANDOVER[0], HANDOVER[1], 0.22], [0.1, 0.95, 0.3], 1,
                       physicsClientId=sim.client)
    p.addUserDebugText('HANDOVER target', [HANDOVER[0], HANDOVER[1], 0.24],
                       textSize=1.0, textColorRGB=[0.1, 0.95, 0.3],
                       physicsClientId=sim.client)
    if successor == 'B':
        p.addUserDebugText('WALL', [0.45, WALL_Y + 0.02, 0.13], textSize=1.0,
                           textColorRGB=[0.95, 0.85, 0.2],
                           physicsClientId=sim.client)


def build_into(sim, cfg):
    sim.reset()
    enable_contact_solver(sim)
    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(cfg['yaw_deg'])])
    block = create_block(sim, HALF, cfg['mass'], cfg['start'],
                         orientation=orientation, lateral_friction=cfg['friction'],
                         color=(0.85, 0.35, 0.10, 1.0))
    sim.allowed_contacts = {(block, 9), (block, 10)}
    return RobotController(sim), block, create_wall(sim)


def run_arm(sim, ctl, block, wall, arm_key, hud):
    arm = ARMS[arm_key]
    hud.phase_key, hud.block = 'A', block
    hud.phase = f'A handover -> {arm["handover"]}'
    print(f'\n[A] 抓放到 {arm["handover"]}（目标 yaw {HANDOVER_YAW[arm["handover"]]:.0f} 度）')
    a_task = HandoverTask(sim, ctl, block, HANDOVER_YAW[arm['handover']],
                          on_step=hud.on_step)
    a_ok, a_res = a_task.run(), a_task.result()

    reset_ok, reset_res = None, None
    if arm['reset'] and a_ok:
        hud.phase_key, hud.phase = 'reset', 'reset -> canonical'
        print('\n[reset] 重新抓取并放回公共标准状态（yaw=0）')
        reset_ok, reset_res = reset_to_canonical(sim, ctl, block, on_step=hud.on_step)

    succ_ok, succ_res = False, {}
    if a_ok and (reset_ok is None or reset_ok):
        hud.phase_key = arm['successor']
        hud.phase = f'{arm["successor"]} ' + ('side push' if arm['successor'] == 'B'
                                              else 'top grasp')
        print(f'\n[{arm["successor"]}] ' + ('侧面推到挡板' if arm['successor'] == 'B'
                                            else '从上方顶抓并抬升'))
        if arm['successor'] == 'B':
            task = PushTask(sim, ctl, block, on_step=hud.on_step)
            task.wall = wall
        else:
            task = TopGraspTask(sim, ctl, block, on_step=hud.on_step)
        succ_ok, succ_res = task.run(), task.result()
    return a_ok, a_res, reset_ok, reset_res, succ_ok, succ_res


def num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def fnum(value, spec='.1f'):
    v = num(value)
    return 'nan' if v is None else format(v, spec)


def report(sim, block, arm_key, a_ok, a_res, reset_ok, reset_res, succ_ok, succ_res):
    """打印与 report.md 对齐的判定；指标取该阶段结束时的记录，不受后续扰动影响。"""
    total = float(a_res['sim_seconds'])
    pos = a_res['final_object_position']
    print('\n' + '=' * 58)
    print(f'A      : {"OK" if a_ok else "FAIL " + str(a_res["failure"])}')
    print(f'交接   : yaw {fnum(a_res["final_object_yaw_deg"])} 度'
          f'（误差 {fnum(a_res["yaw_error_deg"], "+.1f")}）  '
          f'tilt {fnum(a_res["final_tilt_deg"])}  z {pos[2]:.3f} m')
    if reset_ok is not None:
        total += float(reset_res['sim_seconds'])
        print(f'复位   : {"OK" if reset_ok else "FAIL " + str(reset_res["failure"])}'
              f'  yaw {fnum(reset_res["final_object_yaw_deg"], ".2f")} 度')
    if succ_res:
        total += float(succ_res.get('sim_seconds') or 0.0)
        print(f'后继 {ARMS[arm_key]["successor"]} : '
              f'{"OK" if succ_ok else "FAIL " + str(succ_res.get("failure"))}'
              f' ({succ_res.get("failure_stage")})')
        for key, label in (('displacement_m', '位移'), ('wall_gap_m', '距挡板'),
                           ('lift_rise_m', '抬升'), ('slip_m', '滑移')):
            value = num(succ_res.get(key))
            if value is not None:
                print(f'       {label} {value:.3f} m')
    print(f'总计   : {"成功" if succ_ok else "失败"}   {total:.2f} 仿真秒 / '
          f'{sim.steps} 步')
    print('=' * 58)


def main():
    parser = argparse.ArgumentParser(description='实验 0D 实时可视化')
    parser.add_argument('--arm', default='z1_B', help='实验臂')
    parser.add_argument('--episode', type=int, default=0, help='扰动 episode 序号')
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--speed', type=float, default=1.0, help='实时倍率')
    parser.add_argument('--hold', type=float, default=None, help='结束后自动退出秒数')
    parser.add_argument('--auto', action='store_true', help='不等待空格，立即播放一次')
    parser.add_argument('--list', action='store_true', help='列出所有臂')
    args = parser.parse_args()

    if args.list:
        for key in ARM_ORDER:
            print(f'{key:8s} {NOTE[key]}')
        return
    if args.arm not in ARMS:
        parser.error(f'未知实验臂 {args.arm}，可选：{", ".join(ARM_ORDER)}')

    seed = args.seed + args.episode
    cfg = sample_config(seed)
    sim = Simulation(gui=True, realtime=True, speed=args.speed, reference_cube=False)
    print(f'0D | 臂 {args.arm} | episode {args.episode} | seed {seed}')
    print(f'     {NOTE[args.arm]}')
    print(f'     扰动: 起点 y={cfg["start"][1]:+.4f} m  yaw={cfg["yaw_deg"]:+.2f} 度  '
          f'质量={cfg["mass"]:.4f} kg  摩擦={cfg["friction"]:.3f}')

    while True:
        ctl, block, wall = build_into(sim, cfg)
        # sim.reset() 会把相机复位成默认视角，这里重新对准工作区。
        p.resetDebugVisualizerCamera(CAMERA['distance'], CAMERA['yaw'], CAMERA['pitch'],
                                     CAMERA['target'], physicsClientId=sim.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=sim.client)
        p.removeAllUserDebugItems(physicsClientId=sim.client)
        mark_scene(sim, HANDOVER_YAW[ARMS[args.arm]['handover']], ARMS[args.arm]['successor'])
        hud = Hud(sim, args.arm, args.episode, seed)
        if not args.auto:
            hud.wait('SPACE run    P pause    ESC quit')
        out = run_arm(sim, ctl, block, wall, args.arm, hud)
        report(sim, block, args.arm, *out)
        reason = out[5].get('failure') if out[5] else out[1].get('failure')
        hud.extra = [f'RESULT  {"OK" if out[4] else "FAIL " + str(reason)}',
                     'metrics in terminal']
        if not hud.wait('SPACE replay    ESC quit', timeout=args.hold):
            break
    sim.close()


if __name__ == '__main__':
    main()
