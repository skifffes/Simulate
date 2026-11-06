"""实验 4.5 推理服务：注册 Action Expert-only 配置后委托给 exp40/serve_policy.py。"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp45 import registry  # noqa: E402

NAME = registry.register()
print(f'[exp45] registered config {NAME}', flush=True)
sys.argv = [str(ROOT / 'scripts' / 'exp40' / 'serve_policy.py')] + sys.argv[1:]
runpy.run_path(str(ROOT / 'scripts' / 'exp40' / 'serve_policy.py'), run_name='__main__')
