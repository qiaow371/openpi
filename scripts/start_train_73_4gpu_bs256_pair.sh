#!/bin/bash
# 00073 π0.5 双路对照实验（各 4×PPU，在 openpi_train 容器内跑）：
#   A GPU 5,6,7,8     : bs256 action_horizon=15  configs/train_pi05_73_4gpu_bs256_ah15.yaml
#   B GPU 9,10,11,12  : bs256 action_horizon=75  configs/train_pi05_73_4gpu_bs256_ah75.yaml
# 其余超参照抄 configs/train_pi05_73_16gpu_bs1024.yaml（num_epochs 100 / save_every_epochs 10 / lr 1e-4→1e-5 / wu 2000）。
#
# 宿主机执行：bash /nvme1n1/openpi/cigai_train/openpi-jiangsuan/scripts/start_train_73_4gpu_bs256_pair.sh
# 只跑其中一路：ROLES=A 或 ROLES=B
set -euo pipefail

OPENPI_DOCKER="${OPENPI_DOCKER:-openpi_train}"
OPENPI_REPO="${OPENPI_REPO:-/workspace/openpi/cigai_train/openpi-jiangsuan}"
# 日志是在容器里重定向的，必须用容器内路径（宿主机 /nvme1n1/openpi）。
CONT_LOG_DIR="${CONT_LOG_DIR:-/workspace/openpi}"
ROLES="${ROLES:-AB}"

launch_one() {
  local gpus="$1" port="$2" yaml="$3" log="$4"
  if [[ "$log" != /* ]]; then log="${CONT_LOG_DIR}/${log}"; fi
  echo "[launch] gpus=${gpus} port=${port} ${yaml} -> ${log}"
  docker exec -d "${OPENPI_DOCKER}" bash -lc "
set -euo pipefail
cd ${OPENPI_REPO}
export PYTHONPATH=\$(pwd)/src:\$(pwd)/packages/openpi-client/src:\$(pwd)/cigia_lerobot/src
export PALIGEMMA_TOKENIZER_PATH=/workspace/openpi/models/pi05_fixed/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_VISIBLE_DEVICES=${gpus}
setsid python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --master_port=${port} \\
  scripts/train_from_yaml.py --config ${yaml} \\
  > ${log} 2>&1
"
}

# 已有一模一样的任务在跑就不重复拉起（[.] 是为了不让 pgrep 匹配到自身所在的 bash -lc）
already_running() {
  local key="$1"
  docker exec "${OPENPI_DOCKER}" bash -lc "pgrep -af 'train_pi05_73_4gpu_bs256_[^ ]*[.]yaml' | grep ${key} || true"
}

if [[ "${ROLES}" == *A* ]]; then
  if [[ -n "$(already_running ah15)" ]]; then
    echo "[skip] ah15 已在跑："; already_running ah15
  else
    launch_one 5,6,7,8 29731 configs/train_pi05_73_4gpu_bs256_ah15.yaml train_00073_4gpu_bs256_ah15.log
  fi
fi

if [[ "${ROLES}" == *B* ]]; then
  if [[ -n "$(already_running ah75)" ]]; then
    echo "[skip] ah75 已在跑："; already_running ah75
  else
    launch_one 9,10,11,12 29732 configs/train_pi05_73_4gpu_bs256_ah75.yaml train_00073_4gpu_bs256_ah75.log
  fi
fi

echo
echo "[log] 容器 ${CONT_LOG_DIR}/train_00073_4gpu_bs256_ah15.log  (宿主机 /nvme1n1/openpi/...)"
echo "[log] 容器 ${CONT_LOG_DIR}/train_00073_4gpu_bs256_ah75.log  (宿主机 /nvme1n1/openpi/...)"
echo "[ckpt] ${OPENPI_REPO}/checkpoints/pi05_73_4gpu_bs256_chunk15_20260909 | .../pi05_73_4gpu_bs256_chunk75_20260909"
