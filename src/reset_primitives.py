"""实验 0D 的基线复位原语。

- ``go_home``：只把机械臂插值回 HOME，不改变工件状态（baseline 2 的
  "机器人复位" 代价测量，用于说明复位本身也需要真实动作）。
- ``reset_to_canonical``：baseline 3。用与阶段一完全相同的真实抓取/放置
  动作，把工件从当前交接状态恢复成公共标准状态（HANDOVER 点、yaw=0），
  再交给后继技能。它付出的代价是完整的一次抓取-抬升-旋转-放置。

所有数字都来自仿真执行，不做人为补偿。
"""
import numpy as np

from simulation import HOME
from two_state_task import HandoverTask, object_yaw_deg


def joint_travel(result):
    """把 stage_trace 里各段关节行程相加（含 IK 重试）。"""
    return float(sum(float(s.get('distance_rad') or 0.0)
                     for s in result.get('stage_trace', [])))


def go_home(sim, controller):
    q0 = sim.q()
    start = sim.steps
    result = controller.move(np.asarray(HOME, dtype=float))
    q1 = sim.q()
    return {
        'success': bool(result.success),
        'failure': None if result.success else result.reason,
        'seconds': float((sim.steps - start) * sim.dt),
        'distance_rad': float(result.distance),
        'joint_travel_rad': float(np.abs(q1 - q0).sum()),
    }


def reset_to_canonical(sim, controller, block, params=None, on_step=None):
    """baseline 3：重新抓取当前工件并放到公共标准姿态（HANDOVER, yaw=0）。"""
    current_yaw = float(object_yaw_deg(sim, block))
    reset_params = dict(params or {})
    reset_params['grasp_yaw_deg'] = current_yaw
    task = HandoverTask(sim, controller, block, 0.0, params=reset_params,
                        on_step=on_step)
    ok = task.run()
    result = task.result()
    result['reset_from_yaw_deg'] = current_yaw
    result['joint_travel_rad'] = joint_travel(result)
    result['contacts'] = task.contacts
    result['samples'] = task.samples
    return bool(ok), result
