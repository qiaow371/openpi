#!/usr/bin/env bash
set -u

OUT_DIR="${1:?usage: monitor_openpi_training.sh OUTPUT_DIR [TRAIN_PID] [INTERVAL_SEC]}"
TRAIN_PID="${2:-}"
INTERVAL="${3:-60}"
MONITOR_LOG="${OUT_DIR}/monitor.log"

mkdir -p "${OUT_DIR}"
exec >>"${MONITOR_LOG}" 2>&1

echo "[$(date '+%F %T %z')] monitor started pid=${TRAIN_PID:-unknown} interval=${INTERVAL}s"
while :; do
  now="$(date '+%F %T %z')"
  latest_step="$(grep -Eo 'step[=:][0-9]+' "${OUT_DIR}/train.log" 2>/dev/null | grep -Eo '[0-9]+' | tail -1 || true)"
  latest_ckpt="$(find "${OUT_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -n | tail -1 || true)"
  echo "[$now] step=${latest_step:-unknown} checkpoint=${latest_ckpt:-none}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 \
      | sed 's/^/[gpu] /'
  fi

  if [[ -f "${OUT_DIR}/train.log" ]]; then
    tail -n 3 "${OUT_DIR}/train.log" | sed 's/^/[train] /'
    grep -iE 'out of memory|cuda|traceback|error|exception|killed|segmentation fault' \
      "${OUT_DIR}/train.log" | tail -n 5 | sed 's/^/[error] /' || true
  fi

  if [[ -n "${TRAIN_PID}" ]] && ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[$(date '+%F %T %z')] training pid ${TRAIN_PID} exited; monitor stopping"
    break
  fi
  sleep "${INTERVAL}"
done
