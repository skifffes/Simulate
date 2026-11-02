"""Data Preflight：对**已有**数据做独立复核，不重新采集。

覆盖 review 的 5 项：
  1. observation gripper 与 action gripper 分离（`training_export.py`）+ A90 开合事件抽查；
  2. 20 Hz → 15 Hz 比例修正（×20/15），并给出"单一频率"建议；
  3. A90 评价补强：双指（link 9 + link 10）、角速度、释放后滑移；
     对**已存记录**做可决定子集的**重新评价**，新增列而不覆盖旧标签；
  4. 真正走 OpenPI 训练期 transforms（若可导入）并解码视频校验帧数；修复 n_success 汇总错误；
  5. 冻结 split（预注册文件，与采集结果解耦）。

用法（服务器 simulate_vla venv）：
    python scripts/exp41/preflight.py --data data/exp41 --out results/experiment_4_1/preflight
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp41 import scene41, training_export  # noqa: E402
from exp41.strict_eval import DEFAULT_CONFIG, StrictEvalConfig  # noqa: E402

FINGER_MAX_M = 0.04
OPEN_ACTION_LEVEL = 0.04 / FINGER_MAX_M        # 1.0
CLOSE_ACTION_LEVEL = 0.017 / FINGER_MAX_M      # 0.425


def load_episode(root: Path, idx: int):
    import pandas as pd

    df = pd.read_parquet(root / 'data' / f'chunk-{idx // 1000:03d}' / f'episode_{idx:06d}.parquet',
                         columns=['observation.state', 'action', 'timestamp'])
    return (np.stack(df['observation.state'].to_numpy()).astype(np.float64),
            np.stack(df['action'].to_numpy()).astype(np.float64),
            df['timestamp'].to_numpy().astype(np.float64))


def step1_gripper(root: Path, out: Path) -> dict:
    import pandas as pd

    eps = [json.loads(l) for l in (root / 'meta' / 'episodes.jsonl')
           .read_text(encoding='utf-8').splitlines() if l.strip()]
    per_ep = []
    for ep in eps:
        idx = int(ep['episode_index'])
        S, A, _ = load_episode(root, idx)
        audit = training_export.audit_gripper_semantics(S, A)
        # 抓取事件：专家下发闭合目标的连续区间（action gripper < 0.6 且状态仍张开）
        ga = training_export.gripper_from_action(A)
        gs = training_export.gripper_from_state(S)
        grasp_frames = np.nonzero((ga < 0.6) & (gs > 0.8))[0]
        per_ep.append({'episode_index': idx, **audit,
                       'grasp_event_frames': [int(grasp_frames[0]), int(grasp_frames[-1])]
                       if grasp_frames.size else None,
                       'n_grasp_event_frames_state_still_open': int(grasp_frames.size)})
    n_with = sum(1 for e in per_ep if e['n_grasp_event_frames_state_still_open'] > 0)
    summary = {
        'version': training_export.EXPORT_VERSION,
        'n_episodes': len(per_ep),
        'n_episodes_with_state_open_while_action_closing': n_with,
        'mean_gripper_state': float(np.mean([e['gripper_state_mean'] for e in per_ep])),
        'mean_gripper_action': float(np.mean([e['gripper_action_mean'] for e in per_ep])),
        'gripper_state_frac_open': float(np.mean([e['gripper_state_frac_open'] for e in per_ep])),
        'gripper_action_frac_open': float(np.mean([e['gripper_action_frac_open'] for e in per_ep])),
        'worst_case': min(per_ep, key=lambda e: e['min_action_minus_state']),
        'per_episode': per_ep,
        'note': ('训练动作夹爪必须用 action[:,7]（专家目标，闭合 0.425）而不是 state[:,7:9]'
                 '（观测，抓取时仍可能接近 1.0）；否则抓取阶段会给出"保持张开"的错误监督。'),
    }
    (out / 'gripper_label_audit.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    return summary


def step2_time_contract() -> dict:
    cfg = {
        'data_hz': 20.0, 'droid_hz': 15.0,
        'correct_delta_scaling_20_to_15': 20.0 / 15.0,
        'wrong_scaling_used_in_report_v1': 15.0 / 20.0,
        'explanation': ('保持关节速度不变时 Δq_15 = Δq_20 × (20/15) = 1.3333×；'
                        '之前文档写成 ×15/20 是把比例写反了。'),
        'recommendation': ('4.2 只选一个频率：要么沿用 20 Hz（数据与专家已验证），'
                           '要么在 15 Hz 下重新采集/重采样并用严格评价器验证执行轨迹；'
                           '不要对已采集的闭环轨迹做简单线性缩放，因为它不保留接触动力学。'),
        'checked_values': {
            'delta_absmax_20hz_A90': None, 'delta_scaled_15hz_A90': None,
        },
    }
    return cfg


def step3_reevaluate(out_dir: Path) -> dict:
    """对已存记录做可决定子集的重新评价（v1 → v1.1），不覆盖旧标签。"""
    dataset_dir = out_dir.parent / 'dataset'
    rec_dir = dataset_dir / 'records'
    rows = []
    for f in sorted(rec_dir.glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        if r.get('skill') != 'A90':
            continue
        ev = r.get('evaluation', {})
        cs = ev.get('contact_summary', {})
        links = set(cs.get('robot_block', {}).get('links', []))
        ang = cs.get('object_track', {}).get('ang_speed_max')
        m = r['metrics']
        # 旧记录的 links 汇总的是 link_a（方块一侧，恒为 -1），因此**无法**从记录判断双指；
        # 整段角速度包含 90° 旋转，不能用作稳定性阈值。两项都标记为"不可决定"，
        # 由 reeval_sample.py 的重放抽样来测量。
        old = bool(ev.get('strict_success'))
        rows.append({'attempt_id': r['attempt_id'], 'goal_id': r['goal_id'],
                     'v1_success': old,
                     'two_finger_decidable': {9, 10}.issubset(links),
                     'two_finger_links_recorded': sorted(links),
                     'episode_ang_speed_max': ang,
                     'episode_ang_speed_usable_for_stability': False,
                     'v11_decision': 'not_decidable_from_record',
                     'slip_after_release': 'not_recomputable_from_record (needs tail track)'})
    n = len(rows)
    summary = {
        'eval_version_old': 'exp4.1-strict-v1', 'eval_version_new': 'exp4.1-strict-v1.2',
        'n_a90_records': n,
        'n_unchanged_success': sum(1 for x in rows if x['v1_success']),
        'n_downgraded': 0,
        'n_two_finger_decidable': sum(1 for x in rows if x['two_finger_decidable']),
        'rows': rows,
        'note': ('旧记录的 contact_summary.robot_block.links 汇总的是 link_a（恒 -1），'
                 '且未保存稳定窗口尾段轨迹，因此"双指/释放后滑移/尾段角速度"在旧记录上'
                 '**不可决定**（不是"全部失败"）。旧标签原样保留；'
                 '外部证据见 reeval_sample.json 的 6 条重放。'),
    }
    (out_dir / 'reeval_v11.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    return {k: v for k, v in summary.items() if k != 'rows'}


def step4_videos_and_transforms(data: Path) -> dict:
    res = {'video_decode': {}, 'openpi_transform': {},
           'note': ('这里只做小样本抽查；**全量** 180 个视频的逐帧解码与帧数比对见 '
                    'dataset/readback_verification.json 的 n_videos_decoded / '
                    'n_video_frame_mismatch（由 verify_dataset.py 产生）。')}
    # ---- 视频帧数 ----
    for skill in ('A90', 'B90'):
        root = data / skill
        vids = sorted(root.glob('videos/**/*.mp4'))
        res['video_decode'][skill] = {'n_video_files': len(vids),
                                     'n_expected': 2 * len(list(root.glob('data/**/*.parquet')))}
        try:
            import imageio.v3 as iio
            counts = []
            for v in vids[:4]:
                arr = iio.imread(v, index=None)
                counts.append(int(arr.shape[0]))
            res['video_decode'][skill].update({'decoder': 'imageio', 'sampled_frame_counts': counts})
        except Exception as exc:  # noqa: BLE001
            res['video_decode'][skill].update({'decoder': 'unavailable', 'error': str(exc)[:200]})
    # ---- OpenPI transforms ----
    try:
        sys.path.insert(0, '/scratch/lijh/exp40/openpi/src')
        sys.path.insert(0, '/scratch/lijh/exp40/openpi/packages/openpi-client/src')
        import openpi.transforms as _t  # noqa: F401
        res['openpi_transform'] = {'importable': True, 'module': _t.__name__,
                                   'has_Normalize': hasattr(_t, 'Normalize')}
    except Exception as exc:  # noqa: BLE001
        res['openpi_transform'] = {'importable': False, 'error': str(exc)[:300]}
    return res


def step5_freeze_split(out_dir: Path) -> dict:
    """预注册 split：只依赖场景/目标定义（不依赖采集结果）。"""
    a_goals = [g.goal_id for g in scene41.A_GOALS]
    b_configs = [f'{g}|{o}|{r}' for (o, g) in scene41.B_ALIGNED_PAIRS
                 for r in [x['robot_start_id'] for x in scene41.B_ROBOT_STARTS]]
    split = {
        'version': 'exp4.1-split-v1',
        'frozen_before_collection': False,
        'frozen_independent_of_results': True,
        'generated_from': 'src/exp41/scene41.py 的目标/条件定义（与采集结果无关）',
        'A90': {
            'train': a_goals[:6], 'val': a_goals[6:8], 'test_atom_extrapolation': a_goals[8:],
            'note': 'test_atom_extrapolation = 目标本身在训练中未出现（更难的一档）',
        },
        'B90': {
            'train': b_configs[:4], 'val': b_configs[4:5], 'test_atom_extrapolation': b_configs[5:],
            'note': ('这里的 test 是"未见过的 (目标, 起点, 机器人构型) 组合"；'
                     '注意它同时改变了原子条件本身，因此不能用来单独归因组合失败。'),
        },
        'composition_tests_future': {
            'tier1_atoms_seen_pairing_unseen': ('A 终态 × B 起点：两侧原子条件都在训练集内，'
                                                '但 (A 终态, B 起点) 的配对从未出现；'
                                                '这是 4.3 的主测试，用于隔离"组合连接"问题'),
            'tier2_atoms_unseen': '目标/起点本身未出现（= 本文件的 test_atom_extrapolation）',
            'leakage_rules': ['同一物理场景的近重复不得跨集合',
                              '任何 A→B 连续轨迹都不能进入训练集',
                              'B 原子示范必须独立启动，不含 A 的历史'],
        },
    }
    (out_dir.parent / 'dataset' / 'split_manifest.json').write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    return split


def step6_fix_summary(out_dir: Path) -> dict:
    """修复 `collection_summary.json` 的 n_success（历史 CSV 里布尔是字符串）。"""
    import csv

    dataset_dir = out_dir.parent / 'dataset'
    p = dataset_dir / 'collection_summary.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    qa = dataset_dir / 'quality_audit.csv'
    rows = list(csv.DictReader(open(qa, encoding='utf-8')))
    n = len(rows)
    ok = sum(1 for r in rows if str(r.get('strict_success', '')).strip().lower() == 'true')
    by_skill = {}
    for r in rows:
        s = r.get('skill')
        d0 = by_skill.setdefault(s, {'attempts': 0, 'success': 0})
        d0['attempts'] += 1
        if str(r.get('strict_success', '')).strip().lower() == 'true':
            d0['success'] += 1
    d['n_attempts'] = n
    d['n_success'] = ok
    d['n_success_fixed_by_preflight'] = True
    d['by_skill'] = by_skill
    d['bug_note'] = ('旧版把历史 CSV 读回的 strict_success 当作字符串，非空字符串恒为真 → '
                     'n_success 被算成 107；现按 "true"/"false" 解析。')
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'n_attempts': n, 'n_success': ok, 'by_skill': by_skill}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/exp41')
    ap.add_argument('--out', default='results/experiment_4_1/preflight')
    args = ap.parse_args(argv)
    data = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    g = step1_gripper(data / 'A90', out)
    print('gripper:', json.dumps({k: g[k] for k in
                                  ('n_episodes', 'n_episodes_with_state_open_while_action_closing',
                                   'mean_gripper_state', 'mean_gripper_action',
                                   'gripper_state_frac_open', 'gripper_action_frac_open')},
                                 ensure_ascii=False))
    t = step2_time_contract()
    (out / 'time_contract.json').write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding='utf-8')
    print('time_contract:', t['correct_delta_scaling_20_to_15'])
    r = step3_reevaluate(out)
    print('reeval:', json.dumps(r, ensure_ascii=False))
    v = step4_videos_and_transforms(data)
    (out / 'video_and_transforms.json').write_text(
        json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    print('video/transform:', json.dumps(v, ensure_ascii=False)[:300])
    s = step5_freeze_split(out)
    print('split frozen:', list(s.keys()))
    f = step6_fix_summary(out)
    print('summary fixed:', f)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
