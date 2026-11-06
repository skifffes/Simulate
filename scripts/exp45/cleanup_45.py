"""实验 4.5 磁盘清理：只删**可再生成**的大件，全部带 guard，先打印再删。

删除清单：
  A. /scratch/lijh/exp42/**/params_step* 与 ckpt_{1200,2100,2400}
     → 需要同名 lora_stepNNNN.npz 存在（合并可再导出）。
  B. /scratch/lijh/exp44/{A,B}/ckpts 中**未被 4.5 对照选中**的 LoRA，
     以及全部 opt_state（4.4 训练已完成，hash 已记录在 checkpoint_manifest.json）。
     保留：A/4800、A/8000、B/10000、B/12000 的 LoRA。
不删：数据集、export/export_v3、任何 manifest/log、其他用户目录。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path('/scratch/lijh/exp40/repo')
HOME = Path('/scratch/lijh')


def du_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e9


def main() -> int:
    freed = 0.0
    # ---- A. exp42 可再生成的合并 params ----
    pats = ['exp42/**/params_step*', 'exp42/curve*/ckpt_1200', 'exp42/curve*/ckpt_2100',
            'exp42/curve*/ckpt_2400']
    seen = set()
    for pat in pats:
        for p in sorted(HOME.glob(pat)):
            if not p.is_dir() or p in seen:
                continue
            seen.add(p)
            m = re.search(r'(?:params_step|ckpt_)(\d{5}|\d{4})', p.name)
            step = int(m.group(1)) if m else None
            if step is None:
                print('SKIP (no step in name):', p)
                continue
            loras = list(p.parent.glob(f'lora_step{step:05d}.npz')) or \
                    list(p.parent.glob(f'lora_step0{step}.npz'))
            if not loras:
                print('REFUSE (no guard lora):', p, '-> need lora_step%05d.npz' % step)
                continue
            gb = du_gb(p)
            print(f'delete {p}  ({gb:.1f} GB)  guard={loras[0].name}')
            shutil.rmtree(p)
            freed += gb

    # ---- B. exp44 未选中的 LoRA 与全部 opt_state ----
    keep = {('A', 4800), ('A', 8000), ('B', 10000), ('B', 12000)}
    man = json.loads((REPO / 'results/experiment_4_4/checkpoint_manifest.json')
                     .read_text(encoding='utf-8'))
    listed = {(g, int(i['step'])) for g in man for i in man[g]['lora']}
    for g in ('A', 'B'):
        ck = HOME / 'exp44' / g / 'ckpts'
        for p in sorted(ck.glob('lora_step*.npz')):
            step = int(p.stem.replace('lora_step', ''))
            if (g, step) in keep:
                continue
            if (g, step) not in listed:
                print('REFUSE (not in manifest):', p)
                continue
            gb = p.stat().st_size / 1e9
            print(f'delete {p}  ({gb:.2f} GB)  [not selected for 4.5]')
            p.unlink()
            freed += gb
        for p in sorted(ck.glob('opt_step*.npz')):
            gb = p.stat().st_size / 1e9
            print(f'delete {p}  ({gb:.2f} GB)  [4.4 finished, hash recorded]')
            p.unlink()
            freed += gb

    print(f'\nTOTAL FREED ≈ {freed:.1f} GB')
    print(subprocess.run(['df', '-h', '/scratch'], capture_output=True, text=True).stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
