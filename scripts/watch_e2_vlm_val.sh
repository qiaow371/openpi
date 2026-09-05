#!/usr/bin/env bash
# 等 e2 第一次 step ckpt，再跑满 155 段 VLM val。宿主机跑，不占训练 GPU。
set -euo pipefail
# 正确 e2 是 subtask CE；val 权重默认 best/，不再盯 FM e2_pg_only。
CKPT="${CKPT:-/nvme1n1/openpi/cigai_train/openpi-jiangsuan/checkpoints/e2_subtask_ce_00036/best}"
STAMP="/nvme1n1/openpi/outputs/annotate/00036/e2_vlm_val/watched_step.txt"
LOG="/nvme1n1/openpi/e2_vlm_val_watch.log"
mkdir -p "$(dirname "$STAMP")"
echo "[watch] start $(date -Iseconds) ckpt=$CKPT (CE best; not FM e2_pg_only)" >>"$LOG"
if [[ -f "${CKPT}/model.safetensors" ]]; then
  step="ce_best"
  if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$step" ]]; then
    echo "[watch] run val step=$step $(date -Iseconds)" >>"$LOG"
    docker exec openpi_train bash -lc "
set -euo pipefail
cd /workspace/openpi/cigai_train/openpi-jiangsuan
export PYTHONPATH=\$(pwd):\$(pwd)/src:\$(pwd)/packages/openpi-client/src:\$(pwd)/cigia_lerobot/src
export CUDA_VISIBLE_DEVICES=
export PALIGEMMA_TOKENIZER_PATH=/workspace/openpi/models/pi05_fixed/tokenizer.model
python3 scripts/eval_e2_vlm_val.py --device cuda:0 --weight-dir ${CKPT}
" >>"$LOG" 2>&1 || true
    echo "$step" >"$STAMP"
    echo "[watch] done step=$step $(date -Iseconds)" >>"$LOG"
  fi
else
  echo "[watch] missing ${CKPT}/model.safetensors" >>"$LOG"
fi
