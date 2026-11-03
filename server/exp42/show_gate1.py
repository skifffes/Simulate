"""打印 Gate1 四个回放摘要（measured / recorded 两种标签锚点）。"""
import json

base = '/scratch/lijh/exp40/repo/results/experiment_4_2/contract_replay_p0c'
for anchor in ('measured', 'recorded'):
    for skill in ('A90', 'B90'):
        p = f'{base}/expert_replay_results_{skill}_{anchor}.json'
        try:
            d = json.load(open(p, encoding='utf-8'))
        except FileNotFoundError:
            print('missing', p)
            continue
        print(f'== {skill} anchor={anchor}')
        for r in d:
            print(f"  ep{r['episode']} {r['mode']:8s} success={str(r['strict_success']):5s} "
                  f"reason={r['failure_reason']!s:20s} qerr={r['final_q_vs_expert_state_max_err_rad']:.4f} "
                  f"obj=({r['object_dx']:+.3f},{r['object_dy']:+.3f}) "
                  f"rb={r['robot_block_contact_steps']:5d} wall={r['block_wall_contact_steps']:4d} "
                  f"teacher={r['adapter_target_vs_expert_max_err_rad']}")
