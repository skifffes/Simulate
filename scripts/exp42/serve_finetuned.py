"""用**微调后的** checkpoint 启动 4.0 的策略服务器（保持同一 websocket 协议）。

要点：微调用的是 LoRA 变体（`gemma_2b_lora` / `gemma_300m_lora`），参数树与官方
`pi05_droid`（全参数）不同，必须用相同模型配置构建策略，否则加载会失败。
本脚本先注册 `exp42_pi05_droid_lora`，再用与 `scripts/exp40/serve_policy.py` 相同的
参数解析与启动逻辑运行（直接 runpy 复用，避免复制协议代码）。

用法（openpi venv）：
  python scripts/exp42/serve_finetuned.py --mode checkpoint \
      --config exp42_pi05_droid_lora --checkpoint /scratch/lijh/exp42/run/ckpts/step00501 --port 8123
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp42 import registry  # noqa: E402

NAME = registry.register()
print(f'[exp42] registered config {NAME}', flush=True)
sys.argv = [str(ROOT / 'scripts' / 'exp40' / 'serve_policy.py')] + sys.argv[1:]
runpy.run_path(str(ROOT / 'scripts' / 'exp40' / 'serve_policy.py'), run_name='__main__')
