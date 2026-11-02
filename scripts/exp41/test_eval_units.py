"""小单元测试：布尔解析（回归入口）与双指同时接触判据（评价器 v1.2）。

不需要 PyBullet：`PhysicsMonitor.two_finger_overlap` 只读 `events['robot_block']`，
这里用 `object.__new__` 构造最小实例直接测逻辑。

运行：python scripts/exp41/test_eval_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp41.strict_eval import PhysicsMonitor  # noqa: E402

sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.exp41.collect41 import _is_true as _is_true_collect  # noqa: E402


def fake_monitor(events: list[dict]) -> PhysicsMonitor:
    m = object.__new__(PhysicsMonitor)
    m.events = {'robot_block': events, 'block_wall': [], 'block_plane': [],
                'block_tray': [], 'robot_wall': []}
    m.track = []
    return m


def main() -> int:
    failures = []

    def check(name, cond):
        print(('PASS ' if cond else 'FAIL ') + name)
        if not cond:
            failures.append(name)

    # ---- 1. 布尔解析 ----
    for v in (True, 'True', 'true', 'TRUE', '1', 1):
        check(f'_is_true({v!r}) is True', _is_true_collect(v) is True)
    for v in (False, 'False', 'false', '0', 0, '', None):
        check(f'_is_true({v!r}) is False', _is_true_collect(v) is False)
    check('collect41._is_true 与字符串规则一致',
          all(_is_true_collect(v) == (str(v).strip().lower() in ('true', '1', 'yes'))
              for v in (True, False, 'True', 'False', '1', '0', '', None)))

    # ---- 2. 双指同时接触 ----
    # 两指在同样的步上都有接触 → overlap > 0
    ev_same = ([{'step': s, 'link_b': 9, 'normal_force': 1.0} for s in range(1, 11)]
               + [{'step': s, 'link_b': 10, 'normal_force': 1.0} for s in range(1, 11)])
    ov = fake_monitor(ev_same).two_finger_overlap(0.05)
    check('两指同步接触 overlap=10', ov['overlap_steps'] == 10)

    # 交替接触（先后碰过，但从未同时）→ overlap = 0，必须被判失败
    ev_alt = ([{'step': s, 'link_b': 9, 'normal_force': 1.0} for s in range(1, 6)]
              + [{'step': s, 'link_b': 10, 'normal_force': 1.0} for s in range(6, 11)])
    ov2 = fake_monitor(ev_alt).two_finger_overlap(0.05)
    check('交替接触 overlap=0（应判双指未同时）', ov2['overlap_steps'] == 0)
    st = fake_monitor(ev_alt).finger_contact_stats(0.05)
    check('交替接触时两指各自步数都 ≥5（旧判据会误过）',
          st[9]['steps'] >= 5 and st[10]['steps'] >= 5)

    # 力阈值过滤：低力接触不计入
    ev_weak = ([{'step': s, 'link_b': 9, 'normal_force': 0.001} for s in range(1, 11)])
    st2 = fake_monitor(ev_weak).finger_contact_stats(0.05)
    check('低于力阈值的接触不计步数', st2[9]['steps'] == 0)

    print()
    print(f'{len(failures)} failures' if failures else 'all unit checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
