import json
import sys


def walk(o, path=''):
    if isinstance(o, dict):
        print((path or '<root>'), 'keys', list(o)[:6])
        for k, v in o.items():
            if isinstance(v, dict):
                walk(v, path + '/' + str(k))
            elif isinstance(v, (list, tuple)):
                print(path + '/' + str(k), 'len', len(v))
            else:
                print(path + '/' + str(k), 'scalar', type(v).__name__)


p = sys.argv[1] if len(sys.argv) > 1 else (
    '/scratch/lijh/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json')
walk(json.load(open(p)))
