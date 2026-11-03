"""汇总 4.2 的 checkpoint 事实（哈希、大小、benchmark）。"""
import hashlib
import json
import os

OUT = '/scratch/lijh/exp42'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def size(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


res = {'checkpoints': []}
for tag, npz, pdir in (
        ('step501', f'{OUT}/run/ckpts/lora_step00501.npz', f'{OUT}/run/ckpts/params_nested'),
        ('step901', f'{OUT}/run2/ckpts/lora_step00401.npz', f'{OUT}/run2/ckpts/params_nested')):
    item = {'tag': tag, 'lora_npz': npz, 'params_dir': pdir}
    if os.path.exists(npz):
        item['lora_md5'] = md5(npz)
        item['lora_bytes'] = os.path.getsize(npz)
    if os.path.exists(pdir):
        item['params_bytes'] = size(pdir)
    res['checkpoints'].append(item)

for tag, b in (('step501', f'{OUT}/run/training_speed_benchmark.json'),
               ('step901', f'{OUT}/run2/training_speed_benchmark.json')):
    if os.path.exists(b):
        d = json.load(open(b))
        res[tag + '_bench'] = {k: d[k] for k in ('steps', 'batch_size', 'seconds_per_optimizer_step',
                                                 'compile_seconds', 'steps_per_second',
                                                 'epochs_equivalent')}
        res[tag + '_bench']['val_loss_last'] = [x.get('eval') for x in d['log'] if 'eval' in x]
print(json.dumps(res, ensure_ascii=False, indent=2))
