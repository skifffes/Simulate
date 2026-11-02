"""汇总实验 4.1 的最终数据事实（写报告用）。"""
import glob
import json
import os
import sys

repo = '/scratch/lijh/exp40/repo'
out = {}


def load(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


for name in ('collection_summary', 'diversity_audit', 'dataset_statistics',
             'readback_verification', 'split_manifest', 'fps_feasibility',
             'goal_conditioning_feasibility'):
    p = f'{repo}/results/experiment_4_1/dataset/{name}.json'
    if os.path.exists(p):
        out[name] = load(p)

res = {}
cs = out.get('collection_summary', {})
res['manifests'] = cs.get('manifests')
res['n_attempts_total'] = cs.get('n_attempts')
res['n_success_total'] = cs.get('n_success')

div = out.get('diversity_audit', {}).get('by_skill', {})
res['diversity'] = {k: {kk: v[kk] for kk in ('n_success', 'n_goals', 'object_terminal_uniques',
                                             'robot_terminal_uniques', 'joint_terminal_uniques')}
                    for k, v in div.items()}
res['per_goal'] = {k: v.get('n_per_goal') for k, v in div.items()}

st = out.get('dataset_statistics', {})
res['stats'] = {k: {'n_frames': v['n_frames'], 'fps': v['fps_from_timestamps'],
                    'action_delta_absmax': v['action_delta_7d']['absmax'],
                    'frac_abs_gt_1': v['action_delta_7d']['frac_abs_gt_1_rad'],
                    'gripper_mean': v['gripper_normalized']['mean'],
                    'gripper_frac_open': v['gripper_normalized']['frac_open']}
                for k, v in st.items()}

rb = out.get('readback_verification', {})
res['readback'] = {k: {'n_episodes': v['n_episodes'], 'n_valid': v['n_valid_episodes'],
                       'n_pass': v['n_pass'], 'n_checks': v['n_checks'],
                       'failed': [c['check'] for c in v['failed_checks']]}
                   for k, v in rb.items()}
phase = {}
for k, v in rb.items():
    vals = [e['detail'].get('action_vs_next_state_max_rad') for e in v.get('episodes', [])
            if e.get('detail', {}).get('action_vs_next_state_max_rad') is not None]
    means = [e['detail'].get('action_vs_next_state_mean_rad') for e in v.get('episodes', [])
             if e.get('detail', {}).get('action_vs_next_state_mean_rad') is not None]
    if vals:
        phase[k] = {'max_over_episodes': max(vals), 'mean_of_means': sum(means) / len(means)}
res['action_phase'] = phase
res['split'] = out.get('split_manifest')
res['fps_feasibility'] = {k: {kk: (v[kk]['n_success'], v[kk]['n']) for kk in v if kk != 'note'}
                          for k, v in out.get('fps_feasibility', {}).items() if k != 'note'}
res['goal_conditioning'] = out.get('goal_conditioning_feasibility', {}).get('by_goal')

# 任务分布
tasks = {}
for skill in ('A90', 'B90'):
    p = f'{repo}/data/exp41/{skill}/meta/tasks.jsonl'
    if os.path.exists(p):
        tasks[skill] = [json.loads(l)['task'] for l in open(p, encoding='utf-8')]
res['tasks'] = {k: len(v) for k, v in tasks.items()}
res['task_samples'] = {k: v[:3] for k, v in tasks.items()}
# 视频体积
tot = 0
for f in glob.glob(f'{repo}/data/exp41/*/videos/**/*.mp4', recursive=True):
    tot += os.path.getsize(f)
res['video_bytes'] = tot
res['dataset_bytes'] = sum(os.path.getsize(f) for f in glob.glob(f'{repo}/data/exp41/**/*', recursive=True)
                           if os.path.isfile(f))
print(json.dumps(res, ensure_ascii=False, indent=2))
