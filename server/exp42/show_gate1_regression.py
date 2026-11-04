"""打印 Gate1 回归表（anchor vs target_increment）。"""
import json

base = '/scratch/lijh/exp40/repo/results/experiment_4_2/contract_replay_p0c'
rows = []
for mode in ('anchor', 'target_increment'):
    for skill in ('A90', 'B90'):
        with open(f'{base}/expert_replay_results_{skill}_{mode}.json', encoding='utf-8') as fh:
            for r in json.load(fh):
                rows.append((skill, mode, r))
for skill, mode, r in rows:
    te = r['adapter_target_vs_expert_max_err_rad']
    te_s = 'n/a' if te is None else f'{te:.2e}'
    print(f"{skill:4s} {mode:16s} ep{r['episode']} {r['mode']:7s} ok={str(r['strict_success']):5s} "
          f"reason={r['failure_reason']} qerr={r['final_q_vs_expert_state_max_err_rad']:.2e} "
          f"track={r['max_q_track_err_rad']:.4f} obj=({r['object_dx']:+.4f},{r['object_dy']:+.4f}) "
          f"rb={r['robot_block_contact_steps']} wall={r['block_wall_contact_steps']} teacher={te_s}")
print()
for mode in ('anchor', 'target_increment'):
    sub = [r for s, m, r in rows if m == mode and r['mode'] == 'adapter']
    te = [r['adapter_target_vs_expert_max_err_rad'] for r in sub]
    print(f'{mode}: adapter success {sum(1 for r in sub if r["strict_success"])}/{len(sub)}, '
          f'max teacher {max(te):.2e}, max track {max(r["max_q_track_err_rad"] for r in sub):.4f}')
