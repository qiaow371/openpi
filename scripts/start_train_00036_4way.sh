#!/bin/bash
# 宿主机启动 00036 π0.5 正确四路（禁止再拉起「FM 训 PG」旧 e2/e3/e4）。
#
# 正确配方:
#   e1 GPU0-3  : 动作 FM 共训，prompt_from_subtask
#   e2         : subtask 语言 CE（scripts/train_pi05_subtask_ce.py），不在本脚本里 FM
#   e3 GPU8-11 : CE 拼回的 π0.5 → 只训 action FM
#   e4 GPU12-15: CE 拼回的 π0.5 → PG+action 联合 FM
#
# ssh -F ssh_config jiangsuan-16
# bash /nvme1n1/openpi/cigai_train/openpi-jiangsuan/scripts/start_train_00036_4way.sh
set -euo pipefail

OPENPI_DOCKER="${OPENPI_DOCKER:-openpi_train}"
OPENPI_REPO="${OPENPI_REPO:-/workspace/openpi/cigai_train/openpi-jiangsuan}"
HOST_OPENPI="${HOST_OPENPI:-/nvme1n1/openpi}"
STOP_16GPU="${STOP_16GPU:-0}"
LAUNCH_E1="${LAUNCH_E1:-1}"
LAUNCH_E3="${LAUNCH_E3:-1}"
LAUNCH_E4="${LAUNCH_E4:-1}"
# 默认不杀已在跑的正确任务
FORCE="${FORCE:-0}"

launch_one() {
  local gpus="$1"
  local port="$2"
  local yaml="$3"
  local log="$4"
  echo "[launch] gpus=${gpus} port=${port} ${yaml} -> ${log}"
  docker exec -d "${OPENPI_DOCKER}" bash -lc "
set -euo pipefail
cd ${OPENPI_REPO}
export PYTHONPATH=\$(pwd)/src:\$(pwd)/packages/openpi-client/src:\$(pwd)/cigia_lerobot/src
export PALIGEMMA_TOKENIZER_PATH=/workspace/openpi/models/pi05_fixed/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_VISIBLE_DEVICES=${gpus}
setsid python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --master_port=${port} \
  scripts/train_from_yaml.py --config ${yaml} \
  > ${log} 2>&1
"
}

if [[ "${STOP_16GPU}" == "1" ]]; then
  echo "[stop] 16-gpu joint train"
  docker exec "${OPENPI_DOCKER}" bash -lc '
pkill -f "train_from_yaml.py --config configs/train_pi05_00036_16gpu.yaml" || true
pkill -f "nproc_per_node=16" || true
' || true
fi

# 绝不启动错误配方 yaml
for bad in e2_pg_only e3_pg_then_action e4_pg_then_joint; do
  if docker exec "${OPENPI_DOCKER}" bash -lc "pgrep -af train_from_yaml | grep -q ${bad}"; then
    echo "[kill-wrong] ${bad}"
    docker exec "${OPENPI_DOCKER}" bash -lc "pkill -9 -f ${bad} || true"
  fi
done

if [[ "${FORCE}" != "1" ]] && docker exec "${OPENPI_DOCKER}" bash -lc 'pgrep -f "train_pi05_00036_e1_cotrain\|e3_ce_then\|e4_ce_then" >/dev/null'; then
  echo "[skip] correct jobs already running (FORCE=1 to relaunch)"
  docker exec "${OPENPI_DOCKER}" bash -lc 'pgrep -af "train_from_yaml.py|train_pi05_subtask_ce" | grep -v pgrep || true'
  exit 0
fi

if [[ ! -f "${HOST_OPENPI}/models/pi05_e2_ce_spliced/model.safetensors" ]]; then
  echo "[error] missing pi05_e2_ce_spliced — 先跑 e2 CE 再 splice"
  echo "  python3 scripts/train_pi05_subtask_ce.py ..."
  echo "  python3 scripts/splice_pi05_paligemma.py --base .../pi05_fixed --pg .../e2_subtask_ce_00036/best --out .../pi05_e2_ce_spliced"
  exit 1
fi

echo "[note] e2 = subtask CE（不在此脚本 FM）。权重应已在 checkpoints/e2_subtask_ce_00036/best"

if [[ "${LAUNCH_E1}" == "1" ]]; then
  launch_one 0,1,2,3 29501 configs/train_pi05_00036_e1_cotrain.yaml /workspace/openpi/train_00036_e1_cotrain.log
fi
if [[ "${LAUNCH_E3}" == "1" ]]; then
  launch_one 8,9,10,11 29503 configs/train_pi05_00036_e3_ce_then_action.yaml /workspace/openpi/train_00036_e3_ce_then_action.log
fi
if [[ "${LAUNCH_E4}" == "1" ]]; then
  launch_one 12,13,14,15 29504 configs/train_pi05_00036_e4_ce_then_joint.yaml /workspace/openpi/train_00036_e4_ce_then_joint.log
fi

sleep 12
echo "[logs]"
for f in train_00036_e1_cotrain.log train_00036_e3_ce_then_action.log train_00036_e4_ce_then_joint.log; do
  echo "--- ${HOST_OPENPI}/${f} ---"
  grep -E "trainable_modules|prompt_from_subtask|subtask_sidecar|error|Error|Traceback" "${HOST_OPENPI}/${f}" 2>/dev/null | tail -8 || echo "(not yet)"
done
echo "[OK] correct 4-way recipe launched $(date -Is)"
echo "禁止: configs/train_pi05_00036_e{2,3,4}_pg_*.yaml （DEPRECATED wrong FM-PG）"
