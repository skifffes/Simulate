"""生成 checkpoint_manifest.json：记录 pi05_droid 的每个文件、大小、md5 与本地路径。

在服务器上运行（下载完成后）：
    python3 make_ckpt_manifest.py --ckpt ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
        --list /scratch/lijh/exp40/pi05_droid_objects.tsv --out results/experiment_4_0/checkpoint_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--openpi-commit', default='')
    args = ap.parse_args()

    ckpt = pathlib.Path(os.path.expanduser(args.ckpt))
    files = []
    total = 0
    missing = []
    for line in pathlib.Path(args.list).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        key, size, _md5 = line.split('\t')
        rel = key.replace('checkpoints/pi05_droid/', '')
        p = ckpt / rel
        if not p.exists():
            missing.append(rel)
            continue
        s = p.stat().st_size
        total += s
        files.append({
            'path': rel,
            'gcs_object': key,
            'size_bytes': s,
            'size_matches_listing': str(s) == size,
            'md5': md5(p),
        })

    commit = args.openpi_commit
    if not commit:
        try:
            commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ckpt.parents[3],
                                    capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            commit = 'unknown'

    manifest = {
        'checkpoint': 'pi05_droid',
        'source_url': 'gs://openpi-assets/checkpoints/pi05_droid',
        'http_base': 'https://storage-download.googleapis.com/openpi-assets/checkpoints/pi05_droid',
        'local_cache_path': str(ckpt),
        'cache_layout': 'OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid',
        'config_name': 'pi05_droid (openpi src/openpi/training/config.py:631)',
        'asset_id': 'droid',
        'openpi_commit': commit,
        'n_files': len(files),
        'n_missing': len(missing),
        'total_bytes': total,
        'total_gib': round(total / 1024 ** 3, 3),
        'missing': missing,
        'files': sorted(files, key=lambda f: f['path']),
        'note': ('官方为 Orbax/OCDBT 分片格式，不能按“单文件权重”处理；'
                 '文件必须按相对路径原样落位，openpi 的 maybe_download 才能命中缓存。'),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out}: {len(files)} files, {manifest["total_gib"]} GiB, missing={len(missing)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
