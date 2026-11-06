"""实验 4.5：补充 rescue 的 checkpoint/artifact manifest（含 SHA256）。"""
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


def collect(ck: Path, prefix: str = 'lora_step') -> dict:
    items = [{'step': int(p.stem.replace(prefix, '')), 'path': str(p),
              'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(ck.glob(f'{prefix}*.npz'))]
    opts = [{'step': int(p.stem.replace('opt_step', '')), 'path': str(p),
             'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(ck.glob('opt_step*.npz'))]
    return {'checkpoints': items, 'optimizer_state': opts}


def main() -> int:
    man = json.loads((L / 'checkpoint_manifest.json').read_text(encoding='utf-8'))
    man['rescue_lr_0_3'] = {'mode': 'action_expert_full_ft_lr0.3',
                            **collect(SRV / 'rescue_lr03' / 'ckpts')}
    (L / 'checkpoint_manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=2),
                                                encoding='utf-8')

    def du(p: Path) -> str:
        r = subprocess.run(['du', '-sh', str(p)], capture_output=True, text=True)
        return r.stdout.split()[0] if r.stdout else '?'

    art = json.loads((L / 'artifact_manifest.json').read_text(encoding='utf-8'))
    art['server_roots'] = {str(k): du(k) for k in (SRV / 'run', SRV / 'rescue_lr03') if k.exists()}
    art['n_ae_checkpoints'] = len(man['action_expert']['checkpoints'])
    art['n_rescue_checkpoints'] = len(man['rescue_lr_0_3']['checkpoints'])
    (L / 'artifact_manifest.json').write_text(json.dumps(art, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    print('main ckpts', art['n_ae_checkpoints'], 'rescue ckpts', art['n_rescue_checkpoints'])
    print(json.dumps(art['server_roots'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
