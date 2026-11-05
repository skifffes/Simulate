"""对合并导出目录 /scratch/lijh/exp43/export 的每个文件做 SHA256（大工件完整性归档）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/scratch/lijh/exp43/export')
OUT = Path('/scratch/lijh/exp40/repo/results/experiment_4_3/data_expansion/'
           'export_sha256.json')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = {}
    total = 0
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(ROOT))
            files[rel] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
            total += p.stat().st_size
    OUT.write_text(json.dumps({'root': str(ROOT), 'n_files': len(files),
                               'total_bytes': total, 'files': files},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print('hashed', len(files), 'files,', round(total / 1e9, 2), 'GB ->', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
