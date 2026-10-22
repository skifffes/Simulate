"""把服务器上的实验 1.1 结果目录递归拉到本地。

只用于本机；凭据走 server/remote.py 的密钥逻辑（~/.ssh/simulate_a40）。
用法：
    python server/pull_experiment_1_1.py --remote ~/simulate_vla/results/experiment_1_1 \
        --local results/experiment_1_1
"""

from __future__ import annotations

import argparse
import os
import posixpath
import stat

import paramiko

from remote import connect, _abs


def mirror(sftp: paramiko.SFTPClient, remote: str, local: str, skip_videos: bool) -> int:
    count = 0
    os.makedirs(local, exist_ok=True)
    for entry in sftp.listdir_attr(remote):
        name = entry.filename
        rpath = posixpath.join(remote, name)
        lpath = os.path.join(local, name)
        if stat.S_ISDIR(entry.st_mode):
            count += mirror(sftp, rpath, lpath, skip_videos)
        else:
            if skip_videos and (name.endswith('.mp4') or name.endswith('.npz')):
                continue
            sftp.get(rpath, lpath)
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--remote', required=True)
    ap.add_argument('--local', required=True)
    ap.add_argument('--skip-videos', action='store_true')
    args = ap.parse_args()

    client = connect(need_password=False)
    try:
        sftp = client.open_sftp()
        remote = _abs(sftp, args.remote)
        n = mirror(sftp, remote, args.local, args.skip_videos)
        sftp.close()
        print(f'downloaded {n} files -> {args.local}')
    finally:
        client.close()


if __name__ == '__main__':
    main()
