"""通过 SSH exec 上传文件（服务器的 SFTP 子系统只读，写入只能走 shell）。

用法：
    python server/push.py src/vla/atomic_tasks/scene2.py remote:~/simulate_vla/src/vla/atomic_tasks/scene2.py
    python server/push.py --list files.txt remote:~/simulate_vla/     # 每行 "本地相对路径"
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import posixpath

from remote import connect, _abs, _split_remote

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK = 6000


def push(client, local: pathlib.Path, remote: str) -> None:
    data = local.read_bytes()
    b64 = base64.b64encode(data).decode()
    part = remote + '.part'
    code, out, err = run(client, f'rm -f {part}')
    for i in range(0, len(b64), CHUNK):
        piece = b64[i:i + CHUNK]
        code, out, err = run(client, f"printf '%s' '{piece}' >> {part}")
        if code != 0:
            raise SystemExit(f'chunk failed: {err}')
    code, out, err = run(client, f'base64 -d {part} > {remote} && rm -f {part}')
    if code != 0:
        raise SystemExit(f'decode failed: {err}')
    print(f'pushed {local} -> {remote} ({len(data)} bytes)')


def run(client, command: str):
    _, stdout, stderr = client.exec_command(command, timeout=60)
    return (stdout.channel.recv_exit_status(),
            stdout.read().decode('utf-8', 'replace'),
            stderr.read().decode('utf-8', 'replace'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('local')
    ap.add_argument('remote')
    args = ap.parse_args()
    client = connect(need_password=False)
    try:
        sftp = client.open_sftp()
        remote = _abs(sftp, _split_remote(args.remote))
        sftp.close()
        if remote.endswith('/'):
            remote = posixpath.join(remote, pathlib.Path(args.local).name)
        push(client, (ROOT / args.local) if not pathlib.Path(args.local).is_absolute()
             else pathlib.Path(args.local), remote)
    finally:
        client.close()


if __name__ == '__main__':
    main()
