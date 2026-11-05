"""实验 4.4：生成 artifact_manifest.json（服务器大工件清单 + GitHub 小文件清单）。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_4')
SRV = Path('/scratch/lijh/exp44')


def du(p: Path) -> str:
    r = subprocess.run(['du', '-sh', str(p)], capture_output=True, text=True)
    return r.stdout.split()[0] if r.stdout else '?'


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out = {'server_root': str(SRV), 'repo_results': str(L), 'checkpoints': {}, 'data': {}}
    for g in ('A', 'B'):
        d = SRV / g / 'ckpts'
        items = []
        for p in sorted(d.glob('*.npz')):
            items.append({'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
        out['checkpoints'][g] = items
    for name, path in (('A58', Path('/scratch/lijh/exp44_export_A58')),
                       ('v3', Path('/scratch/lijh/exp43/export_v3'))):
        out['data'][name] = {'path': str(path), 'size': du(path),
                             'meta_sha256': sha256(path / 'meta.jsonl')}
    logs = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(SRV.glob('logs_*.txt'))}
    out['logs'] = logs
    (L / 'artifact_manifest.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print('artifacts:', {g: len(v) for g, v in out['checkpoints'].items()},
          'logs', len(logs))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
