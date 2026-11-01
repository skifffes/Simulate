"""生成 results/experiment_4_0/artifact_manifest.json：本实验全部工件的清单与 sha256。

用法（仓库根目录）：
    python scripts/exp40/make_artifact_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'experiment_4_0'
SKIP_NAMES = {'artifact_manifest.json'}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        return ''


def main() -> int:
    files = []
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.name in SKIP_NAMES:
            continue
        files.append({
            'path': str(p.relative_to(OUT)).replace('\\', '/'),
            'size_bytes': p.stat().st_size,
            'sha256': sha256(p),
        })
    manifest = {
        'experiment': '4.0',
        'title': 'π₀.₅ 基座迁移与 PyBullet 原子技能环境适配',
        'simulate_commit': git('rev-parse', 'HEAD'),
        'simulate_branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'openpi_commit': '215abfb217dbac7d5f1273282331b9b1866c0479',
        'n_files': len(files),
        'total_bytes': sum(f['size_bytes'] for f in files),
        'files': files,
        'generated_at': subprocess.run(['date', '-Is'], capture_output=True, text=True).stdout.strip()
        if sys.platform != 'win32' else '',
    }
    path = OUT / 'artifact_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {path} ({len(files)} files, {manifest["total_bytes"] / 1024:.1f} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
