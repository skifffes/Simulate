"""极简 SSH 助手：给本项目提供非交互的远程命令 / 文件传输能力。

凭据一律从环境变量读取，不写入代码仓库：
    SIMULATE_SSH_HOST   (默认 222.29.51.96)
    SIMULATE_SSH_PORT   (默认 22)
    SIMULATE_SSH_USER   (默认 lijh)
    SIMULATE_SSH_PASSWORD

用法：
    python server/remote.py install-key                     # 首次：用密码登录并写入公钥
    python server/remote.py run "uname -a"
    python server/remote.py run "nvidia-smi" --timeout 60
    python server/remote.py put data\\lerobot_panda_pick_place remote:~/simulate/data/
    python server/remote.py get remote:~/simulate/train.log .

若存在本地私钥（SIMULATE_SSH_KEY 或 ~/.ssh/simulate_a40），优先用密钥认证，
此时不再需要 SIMULATE_SSH_PASSWORD。
"""

from __future__ import annotations

import argparse
import os
import posixpath
import stat
import sys
import time

import paramiko

DEFAULT_HOST = "222.29.51.96"
DEFAULT_PORT = 22
DEFAULT_USER = "lijh"


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"缺少环境变量 {name}")
    return value


KEY_PATH = os.path.expanduser(os.environ.get("SIMULATE_SSH_KEY", "~/.ssh/simulate_a40"))


def connect(need_password: bool = True) -> paramiko.SSHClient:
    host = _env("SIMULATE_SSH_HOST", DEFAULT_HOST)
    port = int(_env("SIMULATE_SSH_PORT", str(DEFAULT_PORT)))
    user = _env("SIMULATE_SSH_USER", DEFAULT_USER)
    password = os.environ.get("SIMULATE_SSH_PASSWORD")
    key_file = KEY_PATH if os.path.exists(KEY_PATH) and not need_password else None
    if key_file is None and password is None:
        raise SystemExit("既没有密钥也没有 SIMULATE_SSH_PASSWORD")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        key_filename=key_file,
        timeout=15,
        auth_timeout=15,
        banner_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def cmd_install_key(client: paramiko.SSHClient) -> None:
    pub_path = KEY_PATH + ".pub"
    if not os.path.exists(pub_path):
        raise SystemExit(f"找不到公钥 {pub_path}，先运行 ssh-keygen -t ed25519 -f \"{KEY_PATH}\" -N \"\"")
    with open(pub_path, "r", encoding="utf-8") as fh:
        pubkey = fh.read().strip()

    sftp = client.open_sftp()
    try:
        sftp.stat(".ssh")
    except IOError:
        sftp.mkdir(".ssh")
        sftp.chmod(".ssh", stat.S_IRWXU)
    auth_file = ".ssh/authorized_keys"
    existing = ""
    try:
        with sftp.open(auth_file, "r") as fh:
            existing = fh.read().decode("utf-8", "replace")
    except IOError:
        pass
    if pubkey not in existing:
        with sftp.open(auth_file, "a") as fh:
            fh.write(existing and not existing.endswith("\n") and "\n" or "")
            fh.write(pubkey + "\n")
        sftp.chmod(auth_file, stat.S_IRUSR | stat.S_IWUSR)
    sftp.close()
    print(f"公钥已写入 {auth_file}")


def cmd_run(client: paramiko.SSHClient, command: str, timeout: float) -> int:
    transport = client.get_transport()
    assert transport is not None
    channel = transport.open_session(timeout=timeout)
    channel.get_pty()
    channel.exec_command(command)
    channel.settimeout(timeout)

    deadline = time.time() + timeout
    while True:
        if channel.recv_ready():
            sys.stdout.write(channel.recv(65536).decode("utf-8", "replace"))
            sys.stdout.flush()
        if channel.recv_stderr_ready():
            sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", "replace"))
            sys.stderr.flush()
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if time.time() > deadline:
            channel.close()
            raise SystemExit(f"命令超时 ({timeout:.0f}s)")
        time.sleep(0.05)
    return channel.recv_exit_status()


def _split_remote(path: str) -> tuple[str, str]:
    if ":" not in path:
        raise SystemExit(f"远程路径需要 host:path 形式，收到 {path}")
    _, remote = path.split(":", 1)
    return remote


def _abs(sftp: paramiko.SFTPClient, path: str) -> str:
    """SFTP 不展开 ~，统一换成远端 home 的绝对路径。"""
    if path == "~" or path.startswith("~/"):
        home = sftp.normalize(".")
        return home if path == "~" else posixpath.join(home, path[2:])
    return path


def cmd_put(client: paramiko.SSHClient, local: str, remote: str) -> None:
    remote = _split_remote(remote)
    sftp = client.open_sftp()
    remote = _abs(sftp, remote)

    if os.path.isdir(local):
        _mkdirs(sftp, remote)
        _put_dir(sftp, local, remote)
    else:
        _mkdirs(sftp, posixpath.dirname(remote))
        sftp.put(local, remote)
    print(f"已上传 -> {remote}")
    sftp.close()


def _put_dir(sftp: paramiko.SFTPClient, local: str, remote: str) -> None:
    for name in sorted(os.listdir(local)):
        src = os.path.join(local, name)
        dst = posixpath.join(remote, name)
        if os.path.isdir(src):
            _mkdirs(sftp, dst)
            _put_dir(sftp, src, dst)
        else:
            sftp.put(src, dst)


def cmd_get(client: paramiko.SSHClient, remote: str, local: str) -> None:
    remote = _split_remote(remote)
    sftp = client.open_sftp()
    remote = _abs(sftp, remote)
    sftp.get(remote, local)
    sftp.close()


def _mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    if not path or path == "/":
        return
    parts = path.strip("/").split("/")
    current = "/" if path.startswith("/") else ""
    for part in parts:
        current = posixpath.join(current, part) if current else part
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("install-key", help="用密码登录并安装公钥")

    p_run = sub.add_parser("run", help="执行远程命令")
    p_run.add_argument("command")
    p_run.add_argument("--timeout", type=float, default=120.0)

    p_put = sub.add_parser("put", help="上传文件或目录")
    p_put.add_argument("local")
    p_put.add_argument("remote")

    p_get = sub.add_parser("get", help="下载文件")
    p_get.add_argument("remote")
    p_get.add_argument("local")

    args = parser.parse_args()

    client = connect(need_password=args.action == "install-key")
    try:
        if args.action == "install-key":
            cmd_install_key(client)
            code = 0
        elif args.action == "run":
            code = cmd_run(client, args.command, args.timeout)
        elif args.action == "put":
            cmd_put(client, args.local, args.remote)
            code = 0
        else:
            cmd_get(client, args.remote, args.local)
            code = 0
    finally:
        client.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
