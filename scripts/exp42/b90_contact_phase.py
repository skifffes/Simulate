"""B90 接触阶段的 EE/物体方向对比：模型 rollout vs 专家回放（同阶段、同判据）。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_model(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    log = d['action_log']
    ticks = [r for r in log if r.get('robot_block_contacts', 0) > 0]
    def agg(rows, key):
        if not rows:
            return None
        x = np.asarray([[r[f'{key}_dx'], r[f'{key}_dy']] for r in rows])
        return {'n': len(x), 'sum_dx': float(x[:, 0].sum()), 'sum_dy': float(x[:, 1].sum()),
                'mean_dx': float(x[:, 0].mean()), 'mean_dy': float(x[:, 1].mean()),
                'frac_positive_x': float((x[:, 0] > 0).mean())}
    return {'file': str(path), 'skill': d['skill'], 'episode': d['episode'],
            'success': d['strict_success'], 'reason': d['failure_reason'],
            'n_contact_ticks': len(ticks),
            'contact_EE': agg(ticks, 'ee'), 'contact_object': agg(ticks, 'obj'),
            'all_EE': agg(log, 'ee'), 'all_object': agg(log, 'obj')}


def load_expert(path):
    rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['mode'] == 'adapter']
    out = []
    for ep in sorted({r['episode'] for r in rows}):
        sub = [r for r in rows if r['episode'] == ep]
        contact = [r for r in sub if int(r['robot_block_contacts']) > 0]
        def agg(rs, kind):
            if not rs:
                return None
            if kind == 'ee':
                x = np.asarray([[float(r['ee_x']) for r in rs],
                                [float(r['ee_y']) for r in rs]])
                # 用相邻差分求每 tick 位移
                x = np.diff(x, axis=1) if x.shape[1] > 1 else np.zeros((2, 1))
                return {'n': int(x.shape[1]), 'sum_dx': float(x[0].sum()),
                        'sum_dy': float(x[1].sum()), 'frac_positive_x': float((x[0] > 0).mean())}
            x = np.asarray([[float(r['obj_x']) for r in rs], [float(r['obj_y']) for r in rs]])
            x = np.diff(x, axis=1) if x.shape[1] > 1 else np.zeros((2, 1))
            return {'n': int(x.shape[1]), 'sum_dx': float(x[0].sum()),
                    'sum_dy': float(x[1].sum()), 'frac_positive_x': float((x[0] > 0).mean())}
        out.append({'file': str(path), 'episode': ep,
                    'n_contact_ticks': len(contact),
                    'contact_EE': agg(contact, 'ee'), 'contact_object': agg(contact, 'obj')})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-json', nargs='+', required=True)
    ap.add_argument('--expert-csv', required=True)
    ap.add_argument('--out', default='results/experiment_4_2/contract_replay_p0c')
    args = ap.parse_args(argv)
    res = {'model': [load_model(p) for p in args.model_json],
           'expert': load_expert(args.expert_csv)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'b90_contact_phase_direction.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2)[:3000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
