#!/usr/bin/env bash
# 监控单个 OpenPI 训练 run：写 monitor.log / progress.json；进程退出或检测到致命错误时写 failure_report。
set -u

OUT_DIR="${1:?usage: monitor_openpi_training.sh OUTPUT_DIR [TRAIN_PID] [INTERVAL_SEC] [HOST_TAG]}"
TRAIN_PID="${2:-}"
INTERVAL="${3:-60}"
HOST_TAG="${4:-$(hostname)}"
MONITOR_LOG="${OUT_DIR}/monitor.log"
PROGRESS_JSON="${OUT_DIR}/progress.json"
FAILURE_MD="${OUT_DIR}/failure_report.md"
FAILURE_JSON="${OUT_DIR}/failure_report.json"

mkdir -p "${OUT_DIR}"
exec >>"${MONITOR_LOG}" 2>&1

echo "[$(date '+%F %T %z')] monitor started host=${HOST_TAG} pid=${TRAIN_PID:-unknown} interval=${INTERVAL}s out=${OUT_DIR}"

extract_progress() {
  python3 - "$OUT_DIR" <<'PY'
import json, re, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
out = Path(sys.argv[1])
log = out / "train.log"
hist = out / "train_loss_history.json"
TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ).isoformat()
step = None
epoch = None
loss = None
last_line = ""
errors = []
if log.exists():
    raw = log.read_bytes()[-800000:].decode("utf-8", "replace")
    lines = raw.splitlines()
    if lines:
        last_line = lines[-1][:500]
    for line in lines:
        m = re.search(r"step:(\d+)\s+epch:([0-9.]+)\s+loss:([0-9.eE+-]+)", line)
        if m:
            step, epoch, loss = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if re.search(r"(?i)traceback|out of memory|cuda error|killed|segmentation fault|nan|exception:", line):
            errors.append(line[:400])
    errors = errors[-20:]
if step is None and hist.exists():
    try:
        h = json.loads(hist.read_text())
        if h.get("steps"):
            step = int(h["steps"][-1])
        if h.get("losses"):
            loss = float(h["losses"][-1])
    except Exception:
        pass
ckpts = []
ck_dir = out / "checkpoints"
if ck_dir.is_dir():
    for p in ck_dir.iterdir():
        name = p.name
        if p.is_dir() and (name.isdigit() or name.endswith("-tmp-0") or name.replace(".", "").isdigit()):
            ckpts.append(name)
ckpts = sorted(ckpts)
payload = {
    "updated_at": now,
    "step": step,
    "epoch": epoch,
    "loss": loss,
    "latest_checkpoint": ckpts[-1] if ckpts else None,
    "checkpoints": ckpts[-10:],
    "error_hits": len(errors),
    "recent_errors": errors,
    "last_train_line": last_line,
}
(out / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"step={step if step is not None else 'unknown'} epoch={epoch if epoch is not None else 'unknown'} loss={loss if loss is not None else 'unknown'} checkpoint={ckpts[-1] if ckpts else 'none'} errors={len(errors)}")
PY
}

write_failure_report() {
  local reason="$1"
  local exit_code="${2:-}"
  python3 - "$OUT_DIR" "$HOST_TAG" "$TRAIN_PID" "$reason" "$exit_code" <<'PY'
import json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

out = Path(sys.argv[1])
host = sys.argv[2]
pid = sys.argv[3]
reason = sys.argv[4]
exit_code = sys.argv[5]
TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ).isoformat()

def run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=20)[-8000:]
    except Exception as e:
        return f"<failed: {e}>"

progress = {}
pj = out / "progress.json"
if pj.exists():
    try:
        progress = json.loads(pj.read_text())
    except Exception:
        pass

train_log = out / "train.log"
tail = ""
tb = []
if train_log.exists():
    raw = train_log.read_bytes()[-500000:].decode("utf-8", "replace")
    lines = raw.splitlines()
    tail = "\n".join(lines[-80:])
    # extract last traceback block
    idxs = [i for i, ln in enumerate(lines) if "Traceback" in ln]
    if idxs:
        i = idxs[-1]
        tb = lines[i:i + 60]

gpu = run("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv")
disk = run("df -h / /data /home 2>/dev/null | head -20")
procs = run(f"ps -o pid,etime,stat,cmd -p {pid} 2>/dev/null || true")
dmesg = run("dmesg -T 2>/dev/null | grep -iE 'oom|kill|nvidia|xid' | tail -30 || true")

cfg_snap = {}
for name in ("train_pi05.yaml", "train.yaml", "run_meta.json", "train_hardware.json"):
    p = out / name
    if p.exists():
        try:
            cfg_snap[name] = p.read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception as e:
            cfg_snap[name] = f"<read failed: {e}>"

report = {
    "failed_at": now,
    "host": host,
    "run_dir": str(out),
    "train_pid": pid,
    "exit_code": exit_code,
    "reason": reason,
    "progress": progress,
    "gpu": gpu,
    "disk": disk,
    "process": procs,
    "dmesg_hits": dmesg,
    "traceback_tail": tb,
    "train_log_tail": tail.splitlines()[-80:],
    "config_snapshots": {k: (v[:500] + "...") if len(v) > 500 else v for k, v in cfg_snap.items()},
}
(out / "failure_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

md = []
md.append(f"# OpenPI 训练失败报告")
md.append("")
md.append(f"- **时间**: `{now}`")
md.append(f"- **主机**: `{host}`")
md.append(f"- **run_dir**: `{out}`")
md.append(f"- **pid**: `{pid}`  exit: `{exit_code or 'n/a'}`")
md.append(f"- **原因**: {reason}")
md.append("")
md.append("## 进度快照")
md.append("```json")
md.append(json.dumps(progress, ensure_ascii=False, indent=2))
md.append("```")
md.append("")
md.append("## GPU")
md.append("```")
md.append(gpu.strip())
md.append("```")
md.append("")
md.append("## 磁盘")
md.append("```")
md.append(disk.strip())
md.append("```")
md.append("")
if tb:
    md.append("## Traceback")
    md.append("```")
    md.append("\n".join(tb))
    md.append("```")
    md.append("")
md.append("## train.log 尾部")
md.append("```")
md.append(tail[-6000:])
md.append("```")
md.append("")
md.append("## dmesg 相关")
md.append("```")
md.append(dmesg.strip() or "(none)")
md.append("```")
(out / "failure_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"[failure] wrote {out / 'failure_report.md'} and failure_report.json")
PY
}

FATAL_PAT='out of memory|CUDA error|Traceback \(most recent call last\)|Segmentation fault|Killed|XLA runtime error|RESOURCE_EXHAUSTED|jaxlib.*Error'

while :; do
  now="$(date '+%F %T %z')"
  prog_line="$(extract_progress || true)"
  echo "[$now] host=${HOST_TAG} ${prog_line}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 \
      | sed 's/^/[gpu] /'
  fi

  if [[ -f "${OUT_DIR}/train.log" ]]; then
    tail -n 3 "${OUT_DIR}/train.log" | sed 's/^/[train] /'
    # 过滤常见无害 warning，只保留更可能致命的行
    grep -iE "${FATAL_PAT}" "${OUT_DIR}/train.log" 2>/dev/null | tail -n 8 | sed 's/^/[error] /' || true
  fi

  # 磁盘告警
  root_avail_g="$(df -BG / --output=avail | tail -1 | tr -dc '0-9' || true)"
  if [[ -n "${root_avail_g}" && "${root_avail_g}" -lt 20 ]]; then
    echo "[warn] root disk avail=${root_avail_g}G (<20G)"
  fi

  if [[ -n "${TRAIN_PID}" ]]; then
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      wait_code=""
      # 无法 wait 非本 shell 子进程；用 /proc 退出信息尽力取
      if [[ -r "/proc/${TRAIN_PID}/stat" ]]; then
        :
      fi
      echo "[$(date '+%F %T %z')] training pid ${TRAIN_PID} exited; writing failure/completion report"
      # 若已正常跑完（End of training / 达到 num_train_steps），标为 completed 而非 failure
      if grep -qE 'End of training|Training completed|saved final checkpoint' "${OUT_DIR}/train.log" 2>/dev/null; then
        write_failure_report "process_exited_after_completion" "${wait_code}" || true
        # rename semantically? keep failure_report but reason says completion
        mv -f "${FAILURE_MD}" "${OUT_DIR}/completion_report.md" 2>/dev/null || true
        mv -f "${FAILURE_JSON}" "${OUT_DIR}/completion_report.json" 2>/dev/null || true
        echo "[$(date '+%F %T %z')] training appears completed; monitor stopping"
      else
        write_failure_report "process_exited_unexpectedly" "${wait_code}" || true
        echo "[$(date '+%F %T %z')] failure report written; monitor stopping"
      fi
      break
    fi
  fi

  # 日志出现明确致命错误且进程还在：也落一份报告（不退出，继续盯）
  if [[ -f "${OUT_DIR}/train.log" ]] && grep -qiE 'Out of memory|CUDA error:|RESOURCE_EXHAUSTED' "${OUT_DIR}/train.log"; then
    if [[ ! -f "${FAILURE_JSON}" ]]; then
      write_failure_report "fatal_error_detected_in_log_while_running" "" || true
    fi
  fi

  sleep "${INTERVAL}"
done
