"""P0c 契约单元测试（纯 CPU）：固定锚点、chunk 边界重锚、horizon 缩放裁剪、夹爪米单位。

运行：python scripts/exp42/test_contract_p0c.py
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

FAILS: list[str] = []


def check(name: str, cond: bool, detail=None) -> None:
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def main() -> int:
    lo = np.full(7, -2.9)
    hi = np.full(7, 2.9)
    q0 = np.zeros(7)
    cfg = ActionAdapterConfig(gripper_mode='continuous')

    # 1) 固定锚点：q_target[k] = q0 + a_k，不随 k 累积
    labels = np.concatenate([np.full((16, 7), 0.01), np.full((16, 1), 0.425)], axis=1)
    fixed = chunk_to_targets(labels, q0, lo, hi, cfg, anchor_reference='fixed')
    check('fixed anchor: target[k]=q0+a_k',
          all(np.allclose(a.q_target, np.full(7, 0.01)) for a in fixed))

    # 2) recursive：逐级累加（旧行为）
    rec = chunk_to_targets(labels, q0, lo, hi, cfg, anchor_reference='recursive')
    check('recursive: target[k]=sum',
          np.allclose(rec[3].q_target, np.full(7, 0.04)) and np.allclose(rec[15].q_target, np.full(7, 0.16)))

    # 3) chunk 边界重锚：第二个 chunk 用新的 q 作锚点
    q1 = np.full(7, 0.5)
    fixed2 = chunk_to_targets(labels, q1, lo, hi, cfg, anchor_reference='fixed')
    check('new chunk re-anchors at measured q',
          np.allclose(fixed2[0].q_target, np.full(7, 0.51)))

    # 4) horizon-scaled clip：第 k 步允许 (k+1)*max_delta_rad，不再一律 0.25
    big = np.zeros((16, 8))
    big[:, :7] = 0.3          # 每步 0.3 rad 的偏移（超过 0.25 的旧上限）
    fixed_big = chunk_to_targets(big, q0, lo, hi, cfg, anchor_reference='fixed')
    rec_big = chunk_to_targets(big, q0, lo, hi, cfg, anchor_reference='recursive')
    check('fixed anchor keeps 0.3 rad offset at k=0..3 (scaled clip)',
          np.allclose(fixed_big[3].q_target, np.full(7, 0.3)))
    check('recursive clips 0.3 rad to 0.25 (old per-step cap)',
          np.allclose(rec_big[0].q_target, np.full(7, 0.25)))

    # 5) 夹爪：continuous 米单位；binary 0.425 -> 0.0（闭合），1.0 -> 0.04
    g = np.zeros((1, 8))
    g[0, 7] = 0.425
    a = chunk_to_targets(g, q0, lo, hi, cfg, anchor_reference='fixed')[0]
    check('gripper continuous: 0.425 -> 0.017 m', abs(a.gripper_target_m - 0.017) < 1e-12,
          f'{a.gripper_target_m}')
    check('gripper normalized value is 0.425 (must NOT be sent as metres)',
          abs(a.gripper_value - 0.425) < 1e-12)
    b = chunk_to_targets(g, q0, lo, hi, ActionAdapterConfig(gripper_mode='binary'),
                         anchor_reference='fixed')[0]
    check('gripper binary: 0.425 -> 0.0 m (closed)', abs(b.gripper_target_m) < 1e-12)

    # 6) 专家目标恢复：a_k = A_k - S0，固定锚点应精确还原 A_k
    A = np.stack([np.full(7, 0.01 * (k + 1)) for k in range(16)])
    S0 = np.zeros(7)
    recovered = chunk_to_targets(np.concatenate([A - S0[None, :], np.ones((16, 1))], axis=1),
                                 S0, lo, hi, cfg, anchor_reference='fixed')
    err = float(max(np.abs(r.q_target - A[k]).max() for k, r in enumerate(recovered)))
    check('expert target recovery via fixed anchor', err < 1e-12, f'max_err={err:.2e}')

    out = ROOT / 'results' / 'experiment_4_2' / 'contract_replay_p0c' / 'contract_unit_tests.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'n_checks': 9, 'n_fail': len(FAILS), 'failures': FAILS},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n' + ('all contract checks passed' if not FAILS else f'{len(FAILS)} failures'))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    raise SystemExit(main())
