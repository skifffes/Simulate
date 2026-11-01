"""把服务器上的实验 4.0 工件拉回本地仓库（递归目录 + 指定文件重命名）。

用法（本地，仓库根目录）：
    python server/exp40/pull_artifacts.py
"""
from __future__ import annotations

import os
import posixpath
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.remote import connect  # noqa: E402

REMOTE_ROOT = '/scratch/lijh/exp40/repo/results/experiment_4_0'
LOCAL_ROOT = os.path.join('results', 'experiment_4_0')

# (远端文件, 本地相对路径)
EXTRA_FILES = [
    ('/scratch/lijh/exp40/install_openpi5.log', 'logs/install_openpi5.txt'),
    ('/scratch/lijh/exp40/install_openpi4.log', 'logs/install_openpi4.txt'),
    ('/scratch/lijh/exp40/install_openpi.log', 'logs/install_openpi_last_round.txt'),
    ('/scratch/lijh/exp40/download_ckpt6.log', 'logs/download_ckpt6.txt'),
    ('/scratch/lijh/exp40/download_ckpt5.log', 'logs/download_ckpt5.txt'),
    ('/scratch/lijh/exp40/run_all_checkpoint.log', 'logs/run_all_checkpoint.txt'),
    ('/scratch/lijh/exp40/server_checkpoint_8000.log', 'logs/policy_server_checkpoint.txt'),
    ('/scratch/lijh/exp40/server_random_8000.log', 'logs/policy_server_random.txt'),
    ('/scratch/lijh/exp40/recon_out.txt', 'logs/server_recon.txt'),
]


def walk(sftp, root, skip_dirs=('__pycache__',)):
    out = []
    for entry in sftp.listdir_attr(root):
        path = posixpath.join(root, entry.filename)
        if entry.st_mode & 0o040000:
            if entry.filename in skip_dirs:
                continue
            out.extend(walk(sftp, path, skip_dirs))
        else:
            out.append((path, entry.st_size))
    return out


def main() -> int:
    client = connect()
    try:
        sftp = client.open_sftp()
        files = walk(sftp, REMOTE_ROOT)
        n = 0
        for path, size in files:
            rel = posixpath.relpath(path, REMOTE_ROOT)
            dst = os.path.join(LOCAL_ROOT, *rel.split('/'))
            os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
            sftp.get(path, dst)
            n += 1
        print(f'results: {n} files')
        for remote, rel in EXTRA_FILES:
            dst = os.path.join(LOCAL_ROOT, *rel.split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                sftp.get(remote, dst)
                print(f'log: {rel}')
            except IOError:
                print(f'missing (skip): {remote}')
        sftp.close()
    finally:
        client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
