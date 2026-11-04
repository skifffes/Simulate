#!/usr/bin/env bash
# 安全版 GPU 进程清理：**先核对 PID 属主与命令行**，默认 dry-run。
#
# 背景：上一版 `cleanup_gpu.sh` 直接 kill 四个固定 PID，没有校验属主。
# 在多人共用的 GPU 服务器上，PID 会被复用，固定 PID 可能指向别人的作业。
# 本脚本替代它：
#   * 不带参数：列出当前 GPU 上的进程及其属主/命令行
#   * --apply --pattern <子串>：只 kill **属于当前用户且命令行含该子串** 的进程
#
# 用法：
#   bash kill_gpu_procs_safe.sh
#   bash kill_gpu_procs_safe.sh --apply --pattern serve_finetuned
set -u
APPLY=0
PATTERN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ME=$(id -u)
echo "current uid=$ME user=$(whoami)"
echo '--- GPU compute apps ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true

TARGETS=""
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  [ -d "/proc/$P" ] || { echo "pid $P gone"; continue; }
  OWNER_UID=$(stat -c '%u' "/proc/$P" 2>/dev/null || echo 0)
  CMD=$(tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | cut -c1-120)
  MARK="other-user"
  [ "$OWNER_UID" = "$ME" ] && MARK="mine"
  printf 'pid=%-7s owner=%-4s %-10s %s\n' "$P" "$OWNER_UID" "$MARK" "$CMD"
  if [ "$OWNER_UID" = "$ME" ] && [ -n "$PATTERN" ] && echo "$CMD" | grep -q -- "$PATTERN"; then
    TARGETS="$TARGETS $P"
  fi
done

if [ "$APPLY" -eq 0 ]; then
  echo
  echo "dry-run：没有任何进程被结束。加 --apply --pattern <子串> 才会结束匹配且属于你的进程。"
  exit 0
fi

if [ -z "${PATTERN}" ]; then
  echo "拒绝执行：--apply 必须同时给 --pattern（避免误杀）" >&2
  exit 2
fi
if [ -z "$TARGETS" ]; then
  echo "没有匹配的、属于当前用户的进程。"
  exit 0
fi
echo "about to kill:$TARGETS (pattern=$PATTERN)"
for P in $TARGETS; do
  kill "$P" 2>/dev/null && echo "killed $P" || echo "failed $P"
done
