#!/usr/bin/env bash
# 在华为 Atlas 800I A3 宿主上执行（本机驱动 scripts/restart_pi05_npu8_pair.sh 上传并 detached 调用）：
#   等 HF 基座下完 → cast bf16 → 逐张量等价校验 → 换基座 → 装基座硬校验 → 校验 npu8 配置
#   → 停全部旧训练进程 → 并行起「00036@NPU0-7」+「00073@NPU8-15」→ 起 ckpt 保留守护
#
# 幂等：可重复执行（已备份则跳过备份，已 cast 则复用，进程停到 0 才起训）。
# 变量：TREE / CONTAINER=pi05 / START=0（只做基座+配置，不停旧 run 不起训）
#       / DRY_DL=0
set -uo pipefail

TREE=${TREE:-/home/pi05/openpi_src}
CONTAINER=${CONTAINER:-pi05}
MODELS=/home/pi05/models
FP32=$MODELS/pi05_base_hf_fp32.safetensors
EXPECT_FP32=14467165872
HF_URL=https://hf-mirror.com/lerobot/pi05_base/resolve/main/model.safetensors
BASE=$MODELS/model.safetensors
STAGE=$MODELS/model.safetensors.npu8stage
BAKDIR=$MODELS/_backup_bad_base
CHECKER=$TREE/pi05_ckpt_header_check.py
CAST=$TREE/cast_safetensors_bf16.py
PATCHER=$TREE/patch_pi05_base_guard.py
HASHES=$TREE/safetensors_tensor_hashes.py
GOLD=$TREE/base_hashes.gold.local.txt
NEW_HASHES=$TREE/base_hashes.new.txt
CKPT=$TREE/checkpoints
LOG36=$CKPT/train_00036_8npu_fixbase.log
LOG73=$CKPT/train_00073_8npu_fixbase.log
STATUS=$CKPT/npu8_apply.status
START=${START:-1}

EXP36=00036_PI05_npu8_bs128_lr7e-5_ah50_fixbase
EXP73=00073_PI05_npu8_bs128_lr7e-5_ah50_fixbase

mkdir -p "$CKPT"
log() { echo "[$(date '+%F %T')] [apply] $*"; }
die() { log "!!!!! $*"; echo "FAIL: $*" > "$STATUS"; exit 1; }
run() { docker exec "$CONTAINER" bash -lc "$*"; }

echo "RUNNING" > "$STATUS"
log "开始：TREE=$TREE 容器=$CONTAINER"

# ------------------------------------------------- 0) 前置文件
for f in "$CHECKER" "$CAST" "$PATCHER" "$HASHES" "$GOLD" \
         "$TREE/configs/train_pi05_npu8_00036.yaml" "$TREE/configs/train_pi05_npu8_00073.yaml" \
         "$TREE/start_train_npu8_00036.sh" "$TREE/start_train_npu8_00073.sh" \
         "$TREE/huawei_ckpt_prune.sh"; do
  [ -f "$f" ] || die "缺文件 $f（本机驱动没传全）"
done
run "test -f $TREE/scripts/train_from_yaml.py" || die "容器内看不到 $TREE —— /home/pi05 挂载与预期不符"
FREE=$(df -Pk /home | awk 'NR==2{print $4*1024}')
[ "${FREE:-0}" -gt $((120 * 1024 * 1024 * 1024)) ] || die "/home 空闲 ${FREE}B 太小（需要 ≥120G 放基座+日志）"

# ------------------------------------------------- 1) 等 HF fp32 基座下完（下载器挂了自动重启）
# 注意：下载器一开始就把目标 truncate 成满长度（稀疏文件），所以 stat -c%s 不能当进度用，
# 必须看 state.json 里「已完成分片数」是否等于总分片数。
log "=== 1/8 等基座下载：$FP32 ($EXPECT_FP32 B) ==="
PART_BYTES=$((256 * 1024 * 1024))
EXPECT_PARTS=$(( (EXPECT_FP32 + PART_BYTES - 1) / PART_BYTES ))
dl_done() {
  python3 - "$FP32.state.json" "$EXPECT_PARTS" <<'PY' 2>/dev/null
import json, sys
try:
    st = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
ok = len(st.get("done", [])) >= int(sys.argv[2]) and st.get("size") == 14467165872
print(f"{len(st.get('done', []))}/{sys.argv[2]}")
sys.exit(0 if ok else 1)
PY
}
for _i in $(seq 720); do
  if dl_done; then log "全部分片完成"; break; fi
  if ! pgrep -f 'hf_parallel_download.py' >/dev/null; then
    log "下载器不在，重启续传（已完成分片走 state.json）"
    nohup setsid python3 "$TREE/hf_parallel_download.py" "$HF_URL" "$FP32" \
      --size "$EXPECT_FP32" --parts-mib 256 --concurrency 8 \
      >> "$CKPT/dl_hf_base.log" 2>&1 < /dev/null &
    sleep 30
  else
    log "下载中 $(dl_done || true)"
    sleep 60
  fi
done
dl_done || die "等 12h 后基座仍未下完（见 $CKPT/dl_hf_base.log）"
[ "$(stat -c%s "$FP32")" = "$EXPECT_FP32" ] || die "基座大小不符：$(stat -c%s "$FP32") / $EXPECT_FP32"
log "fp32 基座到位：$EXPECT_FP32 字节"

# ------------------------------------------------- 2) fp32 头部门禁（必须是完整 π0.5，不是 PaliGemma 半成品）
log "=== 2/8 fp32 头部校验 ==="
python3 "$CHECKER" "$FP32" | tee /tmp/hw_fp32_head.txt
grep -q "KEYS=812" /tmp/hw_fp32_head.txt || die "HF 基座 key 数不是 812（上游换文件了？）：$(grep KEYS= /tmp/hw_fp32_head.txt)"
grep -q "VERDICT=FULL" /tmp/hw_fp32_head.txt || die "HF 基座判定不完整"
grep -q "'action_head': 8" /tmp/hw_fp32_head.txt || die "HF 基座缺动作头"

# ------------------------------------------------- 3) cast 成 bf16+fp32（与 GPU/PPU 侧同款产物）
log "=== 3/8 容器内 cast → $STAGE ==="
if [ -f "$STAGE" ] && [ "$(stat -c%s "$STAGE")" -gt 8000000000 ]; then
  log "STAGE 已存在（$(stat -c%s "$STAGE")B），复用"
else
  rm -f "$STAGE"
  run "cd $TREE && python3 $CAST $FP32 $STAGE" 2>&1 | tail -6 || die "cast 失败"
  [ -f "$STAGE" ] || die "cast 没产出 $STAGE"
fi
python3 "$CHECKER" "$STAGE" --require-full || die "cast 产物不完整"

log "逐张量与本机已验证基座比对（证明 HF 直下 == 已验证的那份权重）"
run "cd $TREE && python3 $HASHES $STAGE --out $NEW_HASHES" || die "算新基座哈希失败"
python3 "$HASHES" --diff "$GOLD" "$NEW_HASHES" | tee /tmp/hw_hash_diff.txt
grep -q "内容不一致=0" /tmp/hw_hash_diff.txt || die "逐张量不一致 —— HF 基座与已验证基座不是同一份，停手"
grep -qE "只在 B=0" /tmp/hw_hash_diff.txt || log "提示：新基座缺 gold 独有 key（预期是 1 个 tied embed_tokens）"

# ------------------------------------------------- 4) 备份坏基座 + 原子替换
log "=== 4/8 换基座 ==="
if [ ! -e "$BAKDIR/old_bad_base_DONE" ]; then
  mkdir -p "$BAKDIR"
  cp -a "$BASE" "$BAKDIR/old_bad_base_$(date +%Y%m%d_%H%M%S)" || die "备份失败"
  touch "$BAKDIR/old_bad_base_DONE"
  log "旧基座已备份 $BAKDIR"
else
  log "已有备份标记，跳过备份"
fi
[ "$(python3 "$CHECKER" "$BASE" | grep -c 'KEYS=812\|KEYS=813')" = 1 ] && log "注意：现用基座看起来已经是完整的"
mv -f "$STAGE" "$BASE" || die "替换失败"
python3 "$CHECKER" "$BASE" --require-full || die "替换后校验失败"
log "基座已换成完整 π0.5（812 张量，bf16+fp32 混合）"

# ------------------------------------------------- 5) 装基座完整性硬校验
log "=== 5/8 装 _assert_base_complete 硬校验 ==="
run "cd $TREE && python3 $PATCHER scripts/train_pytorch.py" || die "打补丁失败"
run "cd $TREE && python3 -m py_compile scripts/train_pytorch.py" || die "打补丁后语法校验失败"
N=$(grep -c "_assert_base_complete" "$TREE/scripts/train_pytorch.py")
[ "${N:-0}" -ge 2 ] || die "硬校验未生效（只 $N 处引用）"
log "硬校验已装入（$N 处引用）"

# ------------------------------------------------- 6) 配置自检（dry-run 解析步数/数据集）
log "=== 6/8 配置 dry-run 校验 ==="
check_cfg() { # $1=yaml $2=期望 steps
  local y=$1 want=$2 out
  # 必须带上 PALIGEMMA_TOKENIZER_PATH，否则 tokenizer.py 会去 gs:// 下载并 ImportError
  out=$(docker exec -e PALIGEMMA_TOKENIZER_PATH=$MODELS/tokenizer.model -e TORCHDYNAMO_DISABLE=1 "$CONTAINER" \
        bash -lc "cd $TREE && timeout 900 python3 scripts/train_from_yaml.py --config configs/$y.yaml --dry-run" 2>&1) \
    || { printf '%s\n' "$out" | tail -20; die "dry-run $y 失败"; }
  printf '%s\n' "$out" | grep -E '^(exp_name|batch_size|epochs|steps/ep|steps|save_every|repo_id):' | sed 's/^/    /'
  local got; got=$(printf '%s\n' "$out" | awk -F': *' '/^steps:/{print $2; exit}')
  [ "${got:-0}" = "$want" ] || die "$y 解析出的 steps=$got != 期望 $want"
  printf '%s\n' "$out" | grep -qi "error\|Traceback" && die "$y dry-run 输出里有报错"
  log "$y：steps=$got ✓"
}
check_cfg train_pi05_npu8_00036 39400
check_cfg train_pi05_npu8_00073 37925
grep -q "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" "$TREE/start_train_npu8_00036.sh" || die "00036 卡段不对"
grep -q "ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15" "$TREE/start_train_npu8_00073.sh" || die "00073 卡段不对"
grep -q "nproc_per_node=8" "$TREE/start_train_npu8_00036.sh" || die "00036 脚本不是 8 卡"
grep -q "nproc_per_node=8" "$TREE/start_train_npu8_00073.sh" || die "00073 脚本不是 8 卡"
[ "$(grep -oE 'HCCL_IF_BASE_PORT=[0-9]+' "$TREE"/start_train_npu8_000*.sh | awk -F= '{print $2}' | sort -u | wc -l)" = 2 ] \
  || die "两个 run 的 HCCL 端口没分开（会抢 61000）"
log "卡段/端口/nproc 校验通过"

[ "$START" = 1 ] || { log "START=0：只完成换基座与校验，不停训不起训"; echo "OK(no-start)" > "$STATUS"; exit 0; }

# ------------------------------------------------- 7) 解除旧自动化 + 停掉所有旧训练进程并确认卡空
log "=== 7/8 停旧 run ==="
# 2026-09-14 教训：旧的「隧道传完就串行重训 16 卡」链一旦 armed，会在几十分钟后自己接管并把
# 8 卡 run 挤掉。发起端进程死了不等于触发端死了，必须显式 de-arm。
docker exec "$CONTAINER" pkill -f 'fixbase_chain' 2>/dev/null
for s in run_pi05_fixbase_chain.sh; do
  if [ -f "$TREE/$s" ]; then
    mv -f "$TREE/$s" "$TREE/$s.DISABLED_by_npu8" && log "已停用 $s"
  fi
done
TP=$(docker top "$CONTAINER" -o pid,cmd 2>/dev/null | grep -E "torchrun|train_from_yaml" | awk '{print $1}')
if [ -n "$TP" ]; then
  log "杀掉：$(echo "$TP" | tr '\n' ' ')"
  docker exec "$CONTAINER" pkill -TERM -f 'torchrun|train_from_yaml' 2>/dev/null
  kill -TERM $TP 2>/dev/null
  sleep 15
  docker exec "$CONTAINER" pkill -9 -f 'torchrun|train_from_yaml' 2>/dev/null
  TP2=$(docker top "$CONTAINER" -o pid,cmd 2>/dev/null | grep -E "torchrun|train_from_yaml" | awk '{print $1}')
  [ -n "$TP2" ] && kill -9 $TP2 2>/dev/null
  sleep 8
fi
LEFT=$(docker top "$CONTAINER" -o pid,cmd 2>/dev/null | grep -cE "torchrun|train_from_yaml")
log "剩余训练进程：$LEFT"
for _i in $(seq 10); do
  MAXHBM=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+/ 65536' | awk -F/ '{if($1+0>m) m=$1+0} END{print m+0}')
  log "NPU 最大 HBM 占用：${MAXHBM:-0} MB（进程 $LEFT）"
  [ "${MAXHBM:-99999}" -lt 2000 ] && [ "${LEFT:-1}" = 0 ] && break
  sleep 20
done
MAXHBM=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+/ 65536' | awk -F/ '{if($1+0>m) m=$1+0} END{print m+0}')
[ "${MAXHBM:-99999}" -lt 2000 ] || die "仍有 NPU 占卡（max HBM ${MAXHBM}MB），不起训以免抢卡失败"

# ------------------------------------------------- 8) 起两个 8 卡 run + ckpt 保留守护
log "=== 8/8 并行起训 ==="
ensure_mon() { for e in "$EXP36" "$EXP73"; do mkdir -p "$CKPT/$e/resource_monitor"; done; }
ensure_mon
for d in 00036 00073; do
  [ -s "$CKPT/train_${d}_8npu_fixbase.log" ] && cp -f "$CKPT/train_${d}_8npu_fixbase.log" "$CKPT/train_${d}_8npu_fixbase.log.prev"
  docker exec -d -w "$TREE" "$CONTAINER" bash "$TREE/start_train_npu8_${d}.sh"
  log "已投递 start_train_npu8_${d}.sh"
done
# overwrite:true 会在初始化时 rmtree 掉实验目录（连 resource_monitor 一起），
# 监控线程随后每 5s 抛一次 FileNotFoundError（daemon 线程，不致命但刷屏）。
# 起训后 20 分钟内反复补建，让监控线程能正常落盘。
nohup setsid bash -c "for i in \$(seq 40); do
    mkdir -p '$CKPT/$EXP36/resource_monitor' '$CKPT/$EXP73/resource_monitor'; sleep 30; done" \
  >> "$CKPT/mon_dir_fix.log" 2>&1 < /dev/null &

if ! pgrep -f 'huawei_ckpt_prune.sh' >/dev/null; then
  nohup setsid bash "$TREE/huawei_ckpt_prune.sh" >> "$CKPT/ckpt_prune.log" 2>&1 < /dev/null &
  log "ckpt 保留守护已启动"
else
  log "ckpt 保留守护已在跑"
fi

log "两个 8 卡 run 已投递：00036@NPU0-7（$LOG36）、00073@NPU8-15（$LOG73）"
echo "OK" > "$STATUS"
exit 0
