#!/usr/bin/env bash
# 华为 Atlas 上 π0.5 npu8 fixbase run 的 checkpoint 保留守护进程（在服务器宿主执行）。
#
# 为什么要它：scripts/train_pytorch.py 的 save_checkpoint 只写不删（keep_period 在这条
# 路径上根本没被引用），每个 ckpt ≈ 20GB。两个 8 卡 run 各 ~19 次存盘 = 760GB，而 /home
# 只剩 ~870GB —— 不裁剪必然在训练中途把盘写满，训练与别人的容器一起挂掉。
#
# 策略（保守，只碰明确点名的实验目录）：
#   保留  step % KEEP_EVERY == 0（默认 10000）∪ 最新 KEEP_LATEST 个 ∪ last 链接指向的
#   删除  其余，且只删 mtime 超过 30 分钟的（避开正在写的 tmp_*/目录）
#   兜底  /home 空闲低于 MIN_FREE_GB（默认 250）时，即使没到 keep 点也把最老的多余 ckpt 删掉
set -uo pipefail

TREE=${TREE:-/home/pi05/openpi_src}
EXPS=${EXPS:-"00036_PI05_npu8_bs128_lr7e-5_ah50_fixbase 00036_PI05_npu4_bs64_lr3.5e-5_ah16_fixbase 00036_PI05_npu4_bs64_lr3.5e-5_ah25_fixbase 00036_PI05_npu4_bs64_lr3.5e-5_ah75_fixbase 00036_PI05_npu4_bs64_lr3.5e-5_ah100_fixbase"}
KEEP_EVERY=${KEEP_EVERY:-10000}
KEEP_LATEST=${KEEP_LATEST:-3}
MIN_FREE_GB=${MIN_FREE_GB:-250}
INTERVAL=${INTERVAL:-600}
ONCE=${ONCE:-0}

log() { echo "[$(date '+%F %T')] $*"; }

free_gb() { df -Pk /home | awk 'NR==2{printf "%d", $4/1024/1024}'; }

prune_one() { # $1=exp 目录名
  local root=$TREE/checkpoints/$1/checkpoints
  [ -d "$root" ] || return 0
  local steps del step
  steps=$(ls -1 "$root" 2>/dev/null | grep -E '^[0-9]{6}$' | sort)
  [ -n "$steps" ] || return 0
  local keep_list=()
  while read -r step; do
    [ -n "$step" ] || continue
    local n=$((10#$step))
    if [ $((n % KEEP_EVERY)) -eq 0 ]; then keep_list+=("$step"); fi
  done <<< "$steps"
  # 最新 KEEP_LATEST 个也留
  while read -r step; do
    [ -n "$step" ] || continue
    keep_list+=("$step")
  done < <(printf '%s\n' "$steps" | tail -n "$KEEP_LATEST")

  del=0
  while read -r step; do
    [ -n "$step" ] || continue
    printf '%s\n' "${keep_list[@]}" | grep -qx "$step" && continue
    # 正在写的不删（safetensors + optimizer 落盘要几分钟）
    [ $(( $(date +%s) - $(stat -c%Y "$root/$step" 2>/dev/null || echo 0) )) -lt 1800 ] && continue
    log "$1: 删除 $step（不在保留集）"
    rm -rf "$root/$step" && del=$((del + 1))
  done <<< "$steps"
  [ "$del" -gt 0 ] && log "$1: 本轮删 $del 个"
  return 0
}

round() {
  local fg; fg=$(free_gb)
  log "磁盘空闲 ${fg}G；实验：$EXPS"
  local e
  for e in $EXPS; do prune_one "$e"; done
  # 空闲仍然吃紧时，把每个实验多余的旧 ckpt 从最老开始删（保留集之外已经删过了，这里再退一步）
  if [ "${fg:-0}" -lt "$MIN_FREE_GB" ]; then
    for e in $EXPS; do
      local root=$TREE/checkpoints/$e/checkpoints
      [ -d "$root" ] || continue
      ls -1 "$root" 2>/dev/null | grep -E '^[0-9]{6}$' | sort | head -n -"$KEEP_LATEST" | while read -r step; do
        local n=$((10#$step))
        [ $((n % KEEP_EVERY)) -eq 0 ] && continue
        [ $(( $(date +%s) - $(stat -c%Y "$root/$step" 2>/dev/null || echo 0) )) -lt 1800 ] && continue
        log "$e: 磁盘兜底删除 $step"; rm -rf "$root/$step"
      done
    done
  fi
}

log "ckpt 保留守护启动：keep_every=$KEEP_EVERY keep_latest=$KEEP_LATEST min_free=${MIN_FREE_GB}G interval=${INTERVAL}s"
while :; do
  round
  [ "$ONCE" = 1 ] && break
  sleep "$INTERVAL"
done
