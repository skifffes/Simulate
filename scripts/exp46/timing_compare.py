"""实验 4.6：用**可复现的时序量**定位首次偏离（不依赖 FK 参考系对齐）。

对每条 rollout：
  * A90：夹爪开始闭合的 tick（gripper_target_m < 0.03 首次出现）、首次 robot-block 接触 tick；
  * B90：首次 robot-block 接触 tick、首次 block-wall 接触 tick、物体 X 位移达到 1/2/5 cm 的 tick；
与专家记录（records 的 expert_metrics / n_control_steps）对比，给出"第一次偏离"的阶段与 tick。
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_6')
REC = Path('/scratch/lijh/exp40/repo/results/experiment_4_1/dataset/records')


def load_records():
    out = {}
    for f in REC.glob('*.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('stored_episode_index') is not None:
            out[(r['skill'], int(r['stored_episode_index']))] = r
    return out


def expert_steps(rec):
    """专家轨迹里关键事件的（近似）tick：接触步数/12 得控制步。"""
    m = rec.get('expert_metrics') or {}
    rb = m.get('robot_block_contact_steps')
    fw = m.get('two_finger_overlap_first_step')
    return {'expert_robot_block_steps': rb,
            'expert_two_finger_first_step': fw,
            'expert_advance_m': m.get('advance_x_m'),
            'expert_n_steps': rec.get('n_control_steps')}


def main() -> int:
    recs = load_records()
    rows = []
    for d in sorted((L / 'rollouts').glob('step*')):
        step = int(d.name.replace('step', ''))
        for f in sorted(d.glob('off*/*.json')):
            if f.name.endswith('_summary.json'):
                continue
            j = json.loads(f.read_text(encoding='utf-8'))
            skill, ep = j['skill'], int(j['episode'])
            sub = f.with_name(f.stem + '_substeps.csv')
            if not sub.exists():
                continue
            rows_s = list(csv.DictReader(open(sub, encoding='utf-8')))
            first_close = next((int(r['tick']) for r in rows_s
                                if float(r['gripper_target_m']) < 0.03), None)
            first_rb = next((int(r['tick']) for r in rows_s
                             if int(r['robot_block_contacts']) > 0), None)
            first_wall = next((int(r['tick']) for r in rows_s
                               if int(r['block_wall_contacts']) > 0), None)
            # 物体 X 位移（相对第一个子步）
            x0 = float(rows_s[0]['obj_before'].split(';')[0])
            xmax = max(float(r['obj_before'].split(';')[0]) +
                       float(r['obj_dx']) for r in rows_s)
            e = expert_steps(recs.get((skill, ep), {}))
            rows.append({'step': step, 'skill': skill, 'episode': ep,
                         'seed_offset': int(re.search(r'off(\d+)', str(f)).group(1)),
                         'success': j.get('strict_success'),
                         'failure_reason': j.get('failure_reason'),
                         'first_gripper_close_tick': first_close,
                         'first_robot_block_contact_tick': first_rb,
                         'first_block_wall_contact_tick': first_wall,
                         'obj_advance_m': xmax - x0,
                         **e})
    keys = ['step', 'skill', 'episode', 'seed_offset', 'success', 'failure_reason',
            'first_gripper_close_tick', 'first_robot_block_contact_tick',
            'first_block_wall_contact_tick', 'obj_advance_m',
            'expert_robot_block_steps', 'expert_two_finger_first_step',
            'expert_advance_m', 'expert_n_steps']
    with open(L / 'timing_comparison.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('rows', len(rows))
    for r in rows:
        print(f"{r['skill']} ep{r['episode']} off{r['seed_offset']} "
              f"ok={r['success']} reason={r['failure_reason']} "
              f"close={r['first_gripper_close_tick']} rb={r['first_robot_block_contact_tick']} "
              f"wall={r['first_block_wall_contact_tick']} "
              f"adv={r['obj_advance_m']:.3f} (expert adv={r['expert_advance_m']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
