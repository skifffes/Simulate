"""读取 A/B 训练日志关键行（避免远端二次运行）。"""
import re
import sys

for group, path in (('A(anchor)', '/scratch/lijh/exp42/train_anchor.log'),
                    ('B(target_increment)', '/scratch/lijh/exp42/train_B.log')):
    print('==', group)
    for line in open(path, encoding='utf-8', errors='replace'):
        if re.search(r'step (100|200|300)/300', line) or 'first step' in line \
                or 'exported full params' in line:
            print('  ', line.strip())
