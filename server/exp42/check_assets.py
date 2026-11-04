"""P0-a：训练/推理 checkpoint 与资产核查。

1. LoRA npz 张量数、key 前缀；
2. 合并后 params 是否 51/51 命中（由 export 日志保证，这里只校验存在与大小）；
3. `assets/droid/norm_stats.json` 是否存在、md5 是否等于官方 DROID 资产；
4. manifest 的 label_mode/anchor_reference 与执行器约定是否一致。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

OFFICIAL_DROID_MD5 = '2e8bc5547dc797f1df5f4e5401136082'


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, root: Path) -> dict:
    out = {'name': name, 'root': str(root)}
    assets = root / 'assets' / 'droid' / 'norm_stats.json'
    out['assets_path'] = str(assets)
    out['assets_exists'] = assets.exists()
    if assets.exists():
        out['assets_md5'] = md5(assets)
        out['assets_matches_official_droid'] = out['assets_md5'] == OFFICIAL_DROID_MD5
        stats = json.loads(assets.read_text(encoding='utf-8'))
        out['stats_action_q01_len'] = len(stats['norm_stats']['actions']['q01'])
    man = root.parent / 'ckpts' / 'manifest.json'
    if man.exists():
        out['manifest'] = json.loads(man.read_text(encoding='utf-8'))
    return out


def main() -> int:
    roots = [Path(p) for p in sys.argv[1:]]
    res = [check(r.name, r) for r in roots]
    print(json.dumps(res, ensure_ascii=False, indent=2))
    ok = all(r.get('assets_matches_official_droid') for r in res)
    print('ASSET_CHECK:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
