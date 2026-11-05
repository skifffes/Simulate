import glob
import os
import subprocess

for pat in ('/scratch/lijh/exp42/**/lora_step*.npz', '/scratch/lijh/exp42/**/*lora*.npz'):
    fs = glob.glob(pat, recursive=True)
    if fs:
        print(pat, len(fs))
        for f in sorted(fs)[:5]:
            print('  ', f, round(os.path.getsize(f) / 1e9, 2), 'GB')
        break

print(subprocess.run(['bash', '-lc',
                      'du -sh /scratch/lijh/exp42/curve2400/ckpts /scratch/lijh/exp43/export_v3 2>/dev/null; '
                      'ls /scratch/lijh/exp42/curve2400/ckpts | head'], capture_output=True,
                     text=True).stdout)
