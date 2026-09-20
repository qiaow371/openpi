#!/usr/bin/env bash
# 在华为 Atlas 800I A3 上执行（由 scripts/fix_huawei_pi05_base.sh 上传并调用，也可手工跑）：
#   基座权重替换 → 装基座完整性硬校验 + per-joint loss 探针 → 生成 *_fixbase 配置/启动脚本 → 停坏 run → 起串行重训链
#
# 前置：/home/pi05/models/model.safetensors.fixbase 已由驱动上传或在远端逐字节重建，且 md5 校验通过。
# 幂等：重复执行不会重复备份/重复打补丁；替换与生成都是覆盖式安全操作。
# 变量可覆盖：TREE / CONTAINER / EXPECT_KEYS / DOCKER=0（本机自测用，跳过 docker 与起训）
set -uo pipefail

TREE=${TREE:-/home/pi05/openpi_src}
CONTAINER=${CONTAINER:-pi05}
EXPECT_KEYS=${EXPECT_KEYS:-813}
BASE=/home/pi05/models/model.safetensors
STAGE=/home/pi05/models/model.safetensors.fixbase
BAKDIR=/home/pi05/models/_backup_bad_base
CHECKER=$TREE/pi05_ckpt_header_check.py
PATCHER=$TREE/patch_pi05_base_guard.py
PROBER=$TREE/patch_pi05_loss_probe.py
DOCKER=${DOCKER:-1}

log() { echo "[apply $(date '+%T')] $*"; }
die() { echo "[apply !!] $*"; exit 1; }
run() { if [ "$DOCKER" = 1 ]; then docker exec "$CONTAINER" bash -lc "$*"; else eval "$*"; fi; }

[ -f "$STAGE" ] || die "缺待上传文件 $STAGE（先跑本机驱动）"
[ -f "$CHECKER" ] || die "缺 $CHECKER（先跑本机驱动）"
[ -f "$PATCHER" ] || die "缺 $PATCHER（先跑本机驱动）"
[ -f "$PROBER" ] || die "缺 $PROBER（先跑本机驱动）"

# ------------------------------------------------- 1) 挂载一致性 + 文件门禁
LS_HOST=$(stat -c%s "$STAGE")
LS_CT=$(run "stat -c%s $STAGE")
[ "$LS_HOST" = "$LS_CT" ] || die "宿主机与容器看到的大小不同（$LS_HOST vs $LS_CT）—— /home/pi05 挂载与预期不符，停手"
log "挂载一致：$STAGE = $LS_HOST 字节"
python3 "$CHECKER" "$STAGE" --require-full || die "待换基座不完整，拒绝替换"
log "旧基座现状（留证据）："; python3 "$CHECKER" "$BASE" || true

# ------------------------------------------------- 2) 备份 + 原子替换
if [ ! -e "$BAKDIR/old_bad_base_DONE" ]; then
  mkdir -p "$BAKDIR"
  cp -a "$BASE" "$BAKDIR/old_bad_base_$(date +%Y%m%d_%H%M%S)" || die "备份失败"
  touch "$BAKDIR/old_bad_base_DONE"
  log "旧基座已备份到 $BAKDIR"
else
  log "已存在备份标记，跳过备份：$BAKDIR"
fi
mv -f "$STAGE" "$BASE" || die "替换失败"
python3 "$CHECKER" "$BASE" --require-full || die "替换后校验失败"
log "基座已换成完整 π0.5"

# ------------------------------------------------- 3) 装基座完整性硬校验 + per-joint loss 探针
run "cd $TREE && python3 $PATCHER scripts/train_pytorch.py" || die "硬校验补丁应用失败"
run "cd $TREE && python3 -m py_compile scripts/train_pytorch.py" || die "打补丁后语法校验失败"
N=$(grep -c "_assert_base_complete" "$TREE/scripts/train_pytorch.py")
[ "${N:-0}" -ge 2 ] || die "硬校验未生效（只找到 $N 处引用）"
log "硬校验已装入（$N 处引用）"

# 探针必须在硬校验之后装（两者都改 train_pytorch.py，锚点是按装完后的文件验证过的）
run "cd $TREE && python3 $PROBER scripts/train_pytorch.py" || die "loss 探针补丁应用失败"
run "cd $TREE && python3 -m py_compile scripts/train_pytorch.py" || die "装探针后语法校验失败"
M=$(grep -c "fm_probe\|_write_loss_probe" "$TREE/scripts/train_pytorch.py")
[ "${M:-0}" -ge 6 ] || die "loss 探针未生效（只找到 $M 处引用，期望 ≥6）"
log "loss 探针已装入（$M 处引用）：每 log_interval 写 <checkpoint_dir>/loss_probe.jsonl + 一行 PROBE"

# ------------------------------------------------- 4) 生成 *_fixbase 配置与启动脚本
# 用 sed 从现有可用文件派生，只改 exp_name / 配置指向 / 日志名，超参一字不动 —— 单变量对照。
gen_pair() { # $1=数据集 $2=现有 yaml 名(不含 .yaml) $3=现有启动脚本名(不含 .sh)
  local d=$1 cfg=$2 sh=$3
  [ -f "$TREE/configs/$cfg.yaml" ] || die "服务器缺 configs/$cfg.yaml"
  [ -f "$TREE/$sh.sh" ] || die "服务器缺 $sh.sh"
  sed -e "s|^exp_name: .*|exp_name: ${d}_PI05_npu16_bs256_lr1e-4_ah50_fixbase|" \
      "$TREE/configs/$cfg.yaml" > "$TREE/configs/${cfg}_fixbase.yaml" || die "生成 yaml 失败"
  sed -e "s|$cfg\.yaml|${cfg}_fixbase.yaml|" \
      -e "s|^LOG_FILE=.*|LOG_FILE=\"$TREE/checkpoints/train_${d}_16npu_fixbase.log\"|" \
      "$TREE/$sh.sh" > "$TREE/start_train_${d}_npu16_fixbase.sh" || die "生成启动脚本失败"
  chmod +x "$TREE/start_train_${d}_npu16_fixbase.sh"

  grep -q "^exp_name: ${d}_PI05_npu16_bs256_lr1e-4_ah50_fixbase$" "$TREE/configs/${cfg}_fixbase.yaml" \
    || die "$d 的 fixbase yaml exp_name 不对"
  diff <(grep -v "^exp_name:" "$TREE/configs/$cfg.yaml") <(grep -v "^exp_name:" "$TREE/configs/${cfg}_fixbase.yaml") \
    && log "$d：fixbase yaml 与原 yaml 除 exp_name 外完全一致 ✓"
  grep -q "configs/${cfg}_fixbase.yaml" "$TREE/start_train_${d}_npu16_fixbase.sh" \
    || die "$d 的启动脚本没指向 fixbase 配置"
  grep -q "^LOG_FILE=\"$TREE/checkpoints/train_${d}_16npu_fixbase.log\"$" "$TREE/start_train_${d}_npu16_fixbase.sh" \
    || die "$d 的启动脚本日志名不是确定值（门禁会抓不到日志）"
  grep -q "train_from_yaml" "$TREE/start_train_${d}_npu16_fixbase.sh" || die "$d 的启动脚本丢了训练入口"
  # 原 yaml 若已带 _fixbase 以外的绝对路径/相对 checkpoints 目录，这里保持一致
  log "$d：configs/${cfg}_fixbase.yaml + start_train_${d}_npu16_fixbase.sh 就绪"
}
gen_pair 00073 train_pi05_00073_npu16 start_train_00073_npu16
gen_pair 00036 train_pi05_npu16 start_train_npu16

cat > "$TREE/run_pi05_fixbase_chain.sh" <<CHENEOF
#!/bin/bash
# 串行重训 00073 -> 00036（完整基座 + 硬校验）。每步先过基座门禁，不过就终止链。
set -uo pipefail
cd $TREE
for spec in '00073:train_pi05_00073_npu16' '00036:train_pi05_npu16'; do
  d=\${spec%%:*}; cfg=\${spec#*:}
  LOG=$TREE/checkpoints/train_\${d}_16npu_fixbase.log
  mkdir -p $TREE/checkpoints
  echo "[chain] \$(date '+%F %T') 开始 \$d（configs/\${cfg}_fixbase.yaml）"
  bash start_train_\${d}_npu16_fixbase.sh
  rc=\$?
  echo "[chain] \$(date '+%F %T') \$d 退出码 \$rc"
  grep -q 'Base weight check OK: $EXPECT_KEYS' "\$LOG" || { echo "[chain] !! \$d 基座校验未通过，终止链"; tail -30 "\$LOG"; exit 1; }
  [ \$rc -eq 0 ] || { echo '[chain] 训练异常退出，终止链'; exit \$rc; }
done
echo "[chain] \$(date '+%F %T') ALL DONE"
CHENEOF
chmod +x "$TREE/run_pi05_fixbase_chain.sh"
log "重训链脚本就绪：$TREE/run_pi05_fixbase_chain.sh"

[ "${START:-1}" = 1 ] || { log "START=0，跳过停旧 run 与起训"; exit 0; }

# ------------------------------------------------- 5) 停坏 run
log "停止当前（半随机初始化的）训练进程"
if [ "$DOCKER" = 1 ]; then
  TP=$(docker top "$CONTAINER" -o pid,cmd 2>/dev/null | grep -E "torchrun|train_from_yaml|multiprocessing-fork" | awk '{print $1}')
  [ -n "$TP" ] && kill -9 $TP 2>/dev/null
  sleep 8
  LEFT=$(docker top "$CONTAINER" -o pid,cmd 2>/dev/null | grep -cE "torchrun|train_from_yaml")
  log "剩余训练进程数：$LEFT"
fi

# ------------------------------------------------- 6) 起训（detached，ssh 断开不受影响）
if [ "$DOCKER" = 1 ]; then
  docker exec -d -w "$TREE" "$CONTAINER" bash "$TREE/run_pi05_fixbase_chain.sh"
else
  nohup bash "$TREE/run_pi05_fixbase_chain.sh" > /tmp/fixbase_chain.log 2>&1 &
fi
log "已投递重训链：00073 -> 00036（日志 $TREE/checkpoints/train_00073_16npu_fixbase.log）"
