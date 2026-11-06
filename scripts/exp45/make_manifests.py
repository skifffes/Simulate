"""实验 4.5：生成 checkpoint_manifest.json 与 artifact_manifest.json（含 SHA256）。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

L = Path('/scratch/lijh/exp40/repo/results/experiment_4_5')
SRV = Path('/scratch/lijh/exp45')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ck = SRV / 'run' / 'ckpts'
    items = []
    for p in sorted(ck.glob('lora_step*.npz')):
        items.append({'step': int(p.stem.replace('lora_step', '')), 'path': str(p),
                      'bytes': p.stat().st_size, 'sha256': sha256(p)})
    opts = [{'step': int(p.stem.replace('opt_step', '')), 'path': str(p),
             'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(ck.glob('opt_step*.npz'))]
    lora_ref = Path('/scratch/lijh/exp44/B/ckpts/lora_step12000.npz')
    manifest = {
        'action_expert': {'mode': 'action_expert_full_ft', 'checkpoints': items,
                          'optimizer_state': opts,
                          'note': 'npz 只含 trainable 参数（430.1M，19 张量，1.59 GB）'},
        'lora_reference': {'path': str(lora_ref), 'bytes': lora_ref.stat().st_size,
                            'sha256': sha256(lora_ref),
                            'note': '实验 4.4 Group B step 12000，用于配对闭环对照'} if lora_ref.exists() else None,
    }
    (L / 'checkpoint_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                encoding='utf-8')

    def du(p: Path) -> str:
        r = subprocess.run(['du', '-sh', str(p)], capture_output=True, text=True)
        return r.stdout.split()[0] if r.stdout else '?'

    art = {
        'server_roots': {str(k): du(k) for k in (SRV / 'run', SRV / 'offline_placeholder')
                         if k.exists()},
        'data_root': {'path': '/scratch/lijh/exp43/export_v3',
                       'meta_sha256': sha256(Path('/scratch/lijh/exp43/export_v3/meta.jsonl'))},
        'repo_results': str(L),
        'n_ae_checkpoints': len(items), 'n_opt_states': len(opts),
        'logs': {p.name: p.stat().st_size for p in sorted(SRV.glob('*.log'))},
    }
    (L / 'artifact_manifest.json').write_text(json.dumps(art, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    print('ckpts', len(items), 'opts', len(opts), 'lora_ref', bool(manifest['lora_reference']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
