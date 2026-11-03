"""复核独立审计指出的两个执行契约缺陷（纯 CPU，无需 PyBullet）。

缺陷 #1（锚点）：训练标签是 `a_k = q_expert[k] - q_t`（相对锚点），
部署若按递归累加，第 2 步起会把前面各行的偏移再加一遍。
缺陷 #2（夹爪）：`AdaptedAction.gripper_value` 是**归一化**值（0.425），
`gripper_target_m` 才是米（0.017）；把前者下发给手指电机会放大 25 倍。

运行：python scripts/exp42/verify_anchor_and_gripper.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp40.action_adapter import ActionAdapterConfig, chunk_to_targets  # noqa: E402


def main() -> int:
    q0 = np.zeros(7)
    lower = np.full(7, -2.9)
    upper = np.full(7, 2.9)
    # 专家轨迹：从 0.01 递增到 0.16（与审计的合成用例一致）
    q_expert = np.stack([np.full(7, 0.01 * (k + 1)) for k in range(16)])
    # 训练标签 = 相对锚点 q_t 的偏移（train_lora.make_item 的写法）
    labels = q_expert - q0[None, :]
    chunk = np.concatenate([labels, np.full((16, 1), 0.425)], axis=1)

    fixed = chunk_to_targets(chunk, q0, lower, upper, ActionAdapterConfig(gripper_mode='continuous'),
                             anchor_reference='fixed')
    rec = chunk_to_targets(chunk, q0, lower, upper, ActionAdapterConfig(gripper_mode='continuous'),
                           anchor_reference='recursive')

    q_fixed = np.stack([a.q_target for a in fixed])
    q_rec = np.stack([a.q_target for a in rec])
    err_fixed = float(np.abs(q_fixed - q_expert).max())
    err_rec = float(np.abs(q_rec - q_expert).max())

    cfg_cont = ActionAdapterConfig(gripper_mode='continuous')
    cfg_bin = ActionAdapterConfig(gripper_mode='binary')
    a_open = chunk_to_targets(np.tile(np.array([[0, 0, 0, 0, 0, 0, 0, 0.425]]), (1, 1)),
                              q0, lower, upper, cfg_cont, anchor_reference='fixed')[0]
    a_close = chunk_to_targets(np.tile(np.array([[0, 0, 0, 0, 0, 0, 0, 0.0]]), (1, 1)),
                               q0, lower, upper, cfg_bin, anchor_reference='fixed')[0]
    a_open_bin = chunk_to_targets(np.tile(np.array([[0, 0, 0, 0, 0, 0, 0, 1.0]]), (1, 1)),
                                  q0, lower, upper, cfg_bin, anchor_reference='fixed')[0]
    # binary 模式下 0.425 < 0.5 会被判为"闭合"，下发 0.0 m —— 与专家的 0.017 m 也不等价
    a_0425_bin = chunk_to_targets(np.tile(np.array([[0, 0, 0, 0, 0, 0, 0, 0.425]]), (1, 1)),
                                  q0, lower, upper, cfg_bin, anchor_reference='fixed')[0]

    res = {
        'case': 'synthetic expert ramp 0.01..0.16 rad, gripper 0.425',
        'anchor': {
            'fixed_anchor_max_err_rad': err_fixed,
            'recursive_max_err_rad': err_rec,
            'old_exec_last_joint_rad': float(q_rec[-1, 0]),
            'expert_last_joint_rad': float(q_expert[-1, 0]),
            'verdict': 'fixed 复现专家；recursive 累积放大（缺陷 #1 成立）' if err_fixed < 1e-9 and err_rec > 0.5
                       else '与预期不符，需复查',
        },
        'gripper': {
            'normalized_value_(bug_source)': a_open.gripper_value,
            'target_m_continuous': a_open.gripper_target_m,
            'target_m_binary_g1.0_(open)': a_open_bin.gripper_target_m,
            'target_m_binary_g0.425_(closed)': a_0425_bin.gripper_target_m,
            'target_m_binary_g0.0_(closed)': a_close.gripper_target_m,
            'expert_close_width_m': 0.017,
            'ratio_if_sending_normalized': a_open.gripper_value / 0.017,
            'verdict': 'gripper_target_m 在 continuous 下等于专家 0.017 m；下发 gripper_value 会放大 25 倍（缺陷 #2 成立）'
                       if abs(a_open.gripper_target_m - 0.017) < 1e-9 else '与预期不符',
        },
        'fixed_path_used_by': 'scripts/exp42/eval_rollouts.py (anchor_reference="fixed", gripper_mode=continuous)',
        'note': '默认 anchor_reference 仍为 recursive，以复现实验 4.0 归档结果；4.2 显式传 fixed',
    }
    ok = (err_fixed < 1e-9 and err_rec > 0.5
          and abs(a_open.gripper_target_m - 0.017) < 1e-9
          and abs(a_open_bin.gripper_target_m - 0.04) < 1e-9
          and abs(a_0425_bin.gripper_target_m - 0.0) < 1e-9
          and abs(a_close.gripper_target_m - 0.0) < 1e-9)
    res['all_checks_pass'] = bool(ok)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    out = ROOT / 'results' / 'experiment_4_2' / 'independent_audit' / 'agent_contract_verify.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
