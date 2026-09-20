#!/bin/bash
# 00086 π0.5 训练（4×PPU GPU 8,9,10,11，在 openpi_train 容器内跑）：
#   configs/train_pi05_00086_4gpu_bs384_ah100.yaml  bs384 / action_horizon=100 / 100ep / sv10ep
#   与 ah15 版仅两处不同：① action_horizon 15→100；② exp_name 后缀 ah100（对照实验）。
# 宿主机执行：bash /nvme1n1/openpi/cigai_train/openpi-jiangsuan/scripts/start_train_00086_4gpu_ah100.sh
set -euo pipefail

OPENPI_DOCKER="${OPENPI_DOCKER:-openpi_train}"
OPENPI_REPO="${OPENPI_REPO:-/workspace/openpi/cigai_train/openpi-jiangsuan}"
# 日志在容器里重定向，必须用容器内路径（宿主机 /nvme1n1/openpi）。
CONT_LOG_DIR="${CONT_LOG_DIR:-/workspace/openpi}"
GPUS="${GPUS:-8,9,10,11}"
PORT="${PORT:-29798}"
YAML="configs/train_pi05_00086_4gpu_bs384_ah100.yaml"
LOG="${CONT_LOG_DIR}/train_00086_4gpu_bs384_ah100.log"

# 已有一模一样的任务在跑就不重复拉起
if docker exec "${OPENPI_DOCKER}" bash -lc "pgrep -af 'train_from_yaml.py --config ${YAML}' | grep -v pgrep || true" | grep -q train_from_yaml; then
  echo "[skip] 00086-ah100 已在跑："
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

echo "[log]   容器 ${LOG}  (宿主机 /nvme1n1/openpi/train_00086_4gpu_bs384_ah100.log)"
echo "[ckpt]  ${OPENPI_REPO}/checkpoints/pi05_00086_4gpu_bs384_ah100_20260915"
echo "[tail]  docker exec ${OPENPI_DOCKER} tail -f ${LOG}"
