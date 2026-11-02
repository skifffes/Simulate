"""打印 experiment 4.1 的 pilot / collection 摘要（诊断与报告用）。"""
import json
import sys

path = sys.argv[1]
if path.endswith('pilot_summary.json'):
    d = json.load(open(path, encoding='utf-8'))
    s = d['summaries']
    print('plan', json.dumps(d['plan'], ensure_ascii=False))
    for k, v in s.items():
        print(f"\n== {k}: attempts={v['n_attempts']} success={v['n_success']} "
              f"rate={v['success_rate']:.3f} obj_terminal_uniques={v['object_terminal_uniques']} "
              f"q_terminal_uniques={v['robot_q_terminal_uniques']}")
        print('  failure_reasons:', json.dumps(v['failure_reasons'], ensure_ascii=False))
        for cfg, (ok, n) in sorted(v['by_config'].items()):
            print(f'   {cfg}: {ok}/{n}')
else:
    d = json.load(open(path, encoding='utf-8'))
    print(json.dumps(d, ensure_ascii=False, indent=2)[:4000])
