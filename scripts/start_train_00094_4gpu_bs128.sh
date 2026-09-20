#!/bin/bash
# 00094 π0.5 训练（4×PPU GPU 0,13,14,15，在 openpi_train 容器内跑）：
#   configs/train_pi05_00094_4gpu_bs256.yaml  bs256 / action_horizon=50 / 固定 prompt
# 宿主机执行：bash /nvme1n1/openpi/cigai_train/openpi-jiangsuan/scripts/start_train_00094_4gpu_bs128.sh
set -euo pipefail

OPENPI_DOCKER="${OPENPI_DOCKER:-openpi_train}"
OPENPI_REPO="${OPENPI_REPO:-/workspace/openpi/cigai_train/openpi-jiangsuan}"
# 日志在容器里重定向，必须用容器内路径（宿主机 /nvme1n1/openpi）。
CONT_LOG_DIR="${CONT_LOG_DIR:-/workspace/openpi}"
GPUS="${GPUS:-0,13,14,15}"
PORT="${PORT:-29794}"
YAML="configs/train_pi05_00094_4gpu_bs256.yaml"
LOG="${CONT_LOG_DIR}/train_00094_4gpu_bs256_ah50.log"

# 已有一模一样的任务在跑就不重复拉起
if docker exec "${OPENPI_DOCKER}" bash -lc "pgrep -af 'train_from_yaml.py --config ${YAML}' | grep -v pgrep || true" | grep -q train_from_yaml; then
  echo "[skip] 00094 已在跑："
  docker exec "${OPENPI_DOCKER}" bash -lc "pgrep -af 'train_from_yaml.py --config ${YAML}' | grep -v pgrep || true"
  exit 0
fi

echo "[launch] gpus=${GPUS} port=${PORT} ${YAML} -> ${LOG}"
docker exec -d "${OPENPI_DOCKER}" bash -lc "
set -euo pipefail
cd ${OPENPI_REPO}
export PYTHONPATH=\$(pwd)/src:\$(pwd)/packages/openpi-client/src:\$(pwd)/cigia_lerobot/src
export PALIGEMMA_TOKENIZER_PATH=/workspace/openpi/models/pi05_fixed/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_VISIBLE_DEVICES=${GPUS}
setsid python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --master_port=${PORT} \\
  scripts/train_from_yaml.py --config ${YAML} \\
  > ${LOG} 2>&1
"

echo "[log]   容器 ${LOG}  (宿主机 /nvme1n1/openpi/train_00094_4gpu_bs256_ah50.log)"
echo "[ckpt]  ${OPENPI_REPO}/checkpoints/pi05_00094_4gpu_bs256_nw16_ep30_sv2ep_20260910"
echo "[tail]  docker exec ${OPENPI_DOCKER} tail -f ${LOG}"
