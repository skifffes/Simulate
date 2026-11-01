"""把远端 openpi 源码镜像到本地（只用于静态阅读与接口审计，不参与运行）。

用法（Windows 本地）：
    python server/exp40/fetch_ref.py --remote-dir /scratch/lijh/exp40/openpi \
        --local-dir .openpi_ref/openpi --pattern "src/openpi/**/*.py" ...

实现上只拉取文本类小文件，跳过 .git / tests 数据 / LFS 权重。
"""

from __future__ import annotations

import argparse
import os
import posixpath
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.remote import connect  # noqa: E402


def walk(sftp, root: str, skip_dirs=(".git", "__pycache__", "third_party")):
    out = []
    for entry in sftp.listdir_attr(root):
        name = entry.filename
        path = posixpath.join(root, name)
        if entry.st_mode & 0o040000:
            if name in skip_dirs:
                continue
            out.extend(walk(sftp, path, skip_dirs))
        else:
            out.append((path, entry.st_size))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--max-bytes", type=int, default=400_000)
    parser.add_argument("--exts", default=".py,.md,.toml,.txt,.json,.yaml,.yml,.cfg")
    args = parser.parse_args()

    exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())
    client = connect()
    try:
        sftp = client.open_sftp()
        files = walk(sftp, args.remote_dir)
        print(f"remote files: {len(files)}")
        n = 0
        skipped = 0
        for path, size in files:
            rel = posixpath.relpath(path, args.remote_dir)
            if not rel.endswith(exts):
                continue
            if size > args.max_bytes:
                skipped += 1
                continue
            dst = os.path.join(args.local_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sftp.get(path, dst)
            n += 1
        print(f"fetched {n} files, skipped {skipped} large files -> {args.local_dir}")
        sftp.close()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
