#!/usr/bin/env bash
# 华为 Atlas 800I A3（16×Ascend 910）π0.5「8+8 双 run」重训驱动（在本机执行）
#
# 目标：00036 用 NPU 0-7、00073 用 NPU 8-15，两个独立 8 卡 DDP run 并行，都从完整 π0.5
#       基座热启动（原来 16 卡 run 的基座是只含 PaliGemma 的 604 张量半成品，动作专家随机初始化）。
#
# 为什么要本机驱动：UniVPN 隧道单向只有 ~0.2MB/s，8.5G 基座上传要 12h+；改成服务器自己
#   从 hf-mirror 分段并发拉 fp32 基座（~30MB/s），再用同一份 cast 脚本转 bf16，最后拿
#   逐张量 sha256 与本机已验证基座做全量比对来证明「同一份权重」。所有长活儿都在服务器
#   侧 detached 跑，本机只负责上传素材 + 轮询门禁，隧道断了也不影响训练。
#
# 用法：
#   bash scripts/restart_pi05_npu8_pair.sh            # 全流程（等下载→换基座→停旧→起 8+8→门禁）
#   bash scripts/restart_pi05_npu8_pair.sh --watch    # 只做进度/门禁复查，不改任何东西
#   START=0 bash scripts/restart_pi05_npu8_pair.sh    # 只换基座+校验，不停旧 run 不起训
set -uo pipefail

R=${HW_HOST:-root@10.228.7.117}
TREE=/home/pi05/openpi_src
CONTAINER=pi05
HERE=$(cd "$(dirname "$0")" && pwd)
REC=$HERE/../recipes/pi05/train/huawei_npu8
GOLD_SRC=${GOLD_SRC:-/data/xk_dataDriven_tmp/pi05_base_fix/pi05_base_full_bf16.safetensors}
GOLD_LOCAL=/tmp/pi05_base_hashes.gold.txt
APPLY_LOG=$TREE/checkpoints/npu8_apply.log
STATUS=$TREE/checkpoints/npu8_apply.status
LOG36=$TREE/checkpoints/train_00036_8npu_fixbase.log
LOG73=$TREE/checkpoints/train_00073_8npu_fixbase.log
PW=${HUAWEI_PW:-$(grep -hoP "PW='\\K[^']+" /home/xk/.pull_00073_ckpts_v2.sh 2>/dev/null | head -1)}
START=${START:-1}
GATE_MIN=${GATE_MIN:-90}

WATCH=0
for a in "$@"; do case "$a" in --watch) WATCH=1 ;; *) echo "未知参数 $a"; exit 2 ;; esac; done

command -v sshpass >/dev/null || { echo "需要 sshpass"; exit 1; }
[ -n "$PW" ] || { echo "取不到华为密码，请 export HUAWEI_PW=..."; exit 1; }

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo; echo "[$(date '+%F %T')] !!!!! 中止：$* !!!!!"; exit 1; }
hw() { sshpass -p "$PW" ssh -p 22 -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o ServerAliveInterval=20 "$R" "$@" </dev/null 2>/dev/null; }
up() { sshpass -p "$PW" scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no -q "$@"; }

gate_report() { # 只读门禁 + 进度快照
  hw "
    echo \"===== \$(date '+%F %T') 华为 π0.5 8+8 =====\"
    echo -n 'apply 状态: '; cat $STATUS 2>/dev/null || echo '(无)'
    echo -n '容器: '; docker ps --filter name=$CONTAINER --format '{{.Status}}'
    echo -n '训练 rank 总数: '; docker top $CONTAINER -o pid,cmd 2>/dev/null | grep -c train_from_yaml
    # 抢卡哨兵：本场只允许两个 npu8 run。看到 npu16 / fixbase_chain 就说明有别的自动化在接管。
    echo -n '非法并行(应为0): '; docker top $CONTAINER -o pid,cmd 2>/dev/null | grep -cE 'npu16|fixbase_chain'
    echo -n '基座: '; python3 $TREE/pi05_ckpt_header_check.py /home/pi05/models/model.safetensors 2>/dev/null | grep -E '^KEYS=|^VERDICT' | tr '\n' ' '; echo
    for d in 00036 00073; do
      L=$TREE/checkpoints/train_\${d}_8npu_fixbase.log
      echo \"--- \$d ---\"
      [ -f \"\$L\" ] || { echo '  还没日志'; continue; }
      grep -aE 'Base weight check OK|基座权重不完整|Missing weight keys|Enabled memory optimizations' \"\$L\" | head -2 | sed 's/^/  /'
      grep -aoE 'step:[0-9]+ epch:[0-9.]+ loss:[0-9.]+' \"\$L\" | head -1 | sed 's/^/  首步: /'
      grep -aoE 'step:[0-9]+ epch:[0-9.]+ loss:[0-9.]+' \"\$L\" | tail -2 | sed 's/^/  最新: /'
      grep -aoE '[0-9.]+s/it' \"\$L\" | tail -1 | sed 's/^/  步速: /'
      tb=\$(grep -ac 'Traceback' \"\$L\"); rmf=\$(grep -ac 'Resource monitor sample failed' \"\$L\")
      echo \"  异常计数: Traceback=\$tb 其中监控线程=\$rmf\"
      grep -aE 'RuntimeError|OutOfMemory|HCCL|KeyError|ValueError|Error:' \"\$L\" | grep -avE 'cpu_mem|gpu_util|Resource monitor' | tail -2 | sed 's/^/  ! /'
    done
    echo '--- NPU 功耗/AICore（前 4 行）---'
    npu-smi info 2>/dev/null | grep -oE '[0-9]+[0-9.]*W|[0-9]+%' | head -8 | tr '\n' ' '; echo
    echo '--- ckpt ---'
    for e in 00036_PI05_npu8_bs128_lr7e-5_ah50_fixbase 00073_PI05_npu8_bs128_lr7e-5_ah50_fixbase; do
      echo -n \"  \$e: \"; ls -1 $TREE/checkpoints/\$e/checkpoints 2>/dev/null | tr '\n' ' '; echo
    done
    echo -n '磁盘空闲: '; df -Pk /home | awk 'NR==2{printf \"%dG\n\", \$4/1024/1024}'
    echo '--- apply 日志尾 ---'; tail -6 $APPLY_LOG 2>/dev/null | sed 's/^/  /'
  " | grep -viE 'authorized only|monitored and reported'
}

if [ "$WATCH" = 1 ]; then gate_report; exit 0; fi

# ---------------------------------------------------------------- 0) SSH
hw "echo ok" | grep -q ok || die "华为 SSH 不通（UniVPN 隧道未起），先用 --watch 隔一会儿再跑"
log "已连上：$(hw hostname)"

# ---------------------------------------------------------------- 1) 本机算 gold 逐张量哈希
[ -f "$GOLD_SRC" ] || die "本机已验证基座不存在：$GOLD_SRC"
if [ ! -s "$GOLD_LOCAL" ]; then
  log "计算本机基座逐张量哈希（8.5G，一次约 1-2 分钟）：$GOLD_SRC"
  python3 "$HERE/safetensors_tensor_hashes.py" "$GOLD_SRC" --out "$GOLD_LOCAL" || die "算哈希失败"
fi
log "gold 哈希 $(grep -c . "$GOLD_LOCAL") 条"

# ---------------------------------------------------------------- 2) 上传素材
log "=== 上传工具 / 配置 / 启动脚本 ==="
up "$HERE/hf_parallel_download.py" "$HERE/cast_safetensors_bf16.py" "$HERE/pi05_ckpt_header_check.py" \
   "$HERE/patch_pi05_base_guard.py" "$HERE/safetensors_tensor_hashes.py" \
   "$HERE/huawei_ckpt_prune.sh" "$HERE/huawei_pi05_npu8_apply.sh" "$R:$TREE/" || die "上传工具失败"
up "$REC/train_pi05_npu8_00036.yaml" "$REC/train_pi05_npu8_00073.yaml" "$R:$TREE/configs/" || die "上传 yaml 失败"
up "$REC/start_train_npu8_00036.sh" "$REC/start_train_npu8_00073.sh" "$R:$TREE/" || die "上传启动脚本失败"
up "$GOLD_LOCAL" "$R:$TREE/base_hashes.gold.local.txt" || die "上传 gold 哈希失败"
hw "chmod +x $TREE/start_train_npu8_000*.sh $TREE/huawei_ckpt_prune.sh $TREE/huawei_pi05_npu8_apply.sh"
log "素材就位"

# ---------------------------------------------------------------- 3) 确保下载器在跑
if ! hw "pgrep -f hf_parallel_download.py" | grep -q .; then
  log "启动 HF 基座分段并发下载（14.4G fp32）"
  hw "nohup setsid python3 $TREE/hf_parallel_download.py \
        'https://hf-mirror.com/lerobot/pi05_base/resolve/main/model.safetensors' \
        /home/pi05/models/pi05_base_hf_fp32.safetensors --size 14467165872 --parts-mib 256 --concurrency 8 \
        >> $TREE/checkpoints/dl_hf_base.log 2>&1 < /dev/null & sleep 3; pgrep -c -f hf_parallel_download.py"
fi

# ---------------------------------------------------------------- 4) 投远端 apply（detached，隧道断了也不受影响）
if hw "cat $STATUS 2>/dev/null" | grep -q '^OK'; then
  log "apply 已经是 OK，跳过重投（要重跑请删 $STATUS）"
else
  log "=== 投递远端 apply（换基座→校验→停旧→起 8+8） ==="
  hw "START=$START nohup setsid bash $TREE/huawei_pi05_npu8_apply.sh >> $APPLY_LOG 2>&1 < /dev/null & sleep 5; tail -3 $APPLY_LOG"
fi

# ---------------------------------------------------------------- 5) 门禁：等 apply OK + 两个 run 首步
log "=== 等门禁（最长 ${GATE_MIN} 分钟；含下载/cast/dry-run/冷启动首步）==="
probe_gate() {
  hw "
    for d in 00036 00073; do
      L=$TREE/checkpoints/train_\${d}_8npu_fixbase.log
      if [ ! -f \"\$L\" ]; then echo \"\$d none\"; continue; fi
      if grep -q '基座权重不完整' \"\$L\"; then echo \"\$d bad_base\"; continue; fi
      s=\$(grep -aoE 'step:[0-9]+ epch:[0-9.]+ loss:[0-9.]+' \"\$L\" | head -1)
      if [ -n \"\$s\" ]; then echo \"\$d ok \$s\"; continue; fi
      # resource_monitor 的建目录竞态会在 daemon 线程里每 5s 抛一次栈（无害），
      # 只有「Traceback 总数 > 监控失败数」才是训练本体炸了。
      tb=\$(grep -ac 'Traceback' \"\$L\"); rm=\$(grep -ac 'Resource monitor sample failed' \"\$L\")
      if [ \"\$tb\" -gt \"\$rm\" ] && [ \"\$rm\" -gt 0 ]; then echo \"\$d crashed \$(grep -a -m1 -A4 Traceback \"\$L\" | tr '\n' '|')\"; continue; fi
      if [ \"\$tb\" -gt 0 ] && [ \"\$rm\" = 0 ]; then echo \"\$d crashed \$(grep -a -m1 -A4 Traceback \"\$L\" | tr '\n' '|')\"; continue; fi
      echo \"\$d waiting\"
    done"
}
ok36=0; ok73=0
for _i in $(seq "$GATE_MIN"); do
  sleep 60
  ST=$(hw "cat $STATUS 2>/dev/null")
  case "$ST" in
    FAIL*) die "远端 apply 失败：$ST（详见 $APPLY_LOG）" ;;
  esac
  G=$(probe_gate)
  log "status=${ST:-RUNNING} | $(printf '%s' "$G" | tr '\n' ' ')"
  printf '%s\n' "$G" | grep -q 'bad_base' && die "基座硬校验报错（基座仍不完整），见 $APPLY_LOG"
  printf '%s\n' "$G" | grep -q 'crashed' && die "有 run 起训即抛异常：$(printf '%s\n' "$G" | grep crashed | head -1)"
  # 冷启动理论值 1.12：动作专家随机初始化时首步 loss 必然 ≥1.0
  while read -r d st _step _ep fl rest; do
    [ "$st" = ok ] || continue
    v=${fl#loss:}
    case "$v" in ''|*[!0-9.]*) continue ;; esac
    awk -v l="$v" 'BEGIN{exit !(l >= 1.0)}' && die "$d 首步 loss=$v ≥ 冷启动理论值 1.12 ⇒ 动作专家仍是随机初始化，基座没热加载成功"
    log "$d 首步 loss=$v（warm start 区间 0.19~0.57 合格）"
  done <<< "$G"
  case "$G" in *"00036 ok"*) ok36=1 ;; esac
  case "$G" in *"00073 ok"*) ok73=1 ;; esac
  [ "$ok36" = 1 ] && [ "$ok73" = 1 ] && break
done
[ "$ok36" = 1 ] && [ "$ok73" = 1 ] || die "${GATE_MIN} 分钟内没等到两个 run 都过门禁，去服务器查 $APPLY_LOG"

log "两个 8 卡 run 均已过基座校验并进入真实训练步"
echo
gate_report

cat <<EOF
=========================================================================
✔ 华为 16 NPU 已切成两个 8 卡 π0.5 run（完整基座热启动）
  00036 → NPU 0-7，8 rank，bs=128(16/卡)，100 epoch = 39400 步
          configs/train_pi05_npu8_00036.yaml · start_train_npu8_00036.sh
          日志 $LOG36
  00073 → NPU 8-15，8 rank，bs=128(16/卡)，25 epoch = 37925 步
          configs/train_pi05_npu8_00073.yaml · start_train_npu8_00073.sh
          日志 $LOG73
  基座    /home/pi05/models/model.safetensors = 完整 π0.5（812 张量，bf16+fp32）
          来源 hf-mirror lerobot/pi05_base → cast，逐张量 sha256 与本机已验证基座全等
          旧 604 张量半成品备份在 /home/pi05/models/_backup_bad_base/
  硬校验  train_pytorch.py 已装 _assert_base_complete（基座不完整直接停训）
  磁盘    scripts/huawei_ckpt_prune.sh 常驻：保 10k 倍数 + 最新 3 个 ckpt
  复跑    bash $HERE/restart_pi05_npu8_pair.sh --watch
  判读    首步 loss 落在 warm start 区间 0.19~0.57 为热加载；≥1.0 视为冷启动
=========================================================================
EOF
