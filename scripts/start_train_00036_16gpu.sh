#!/bin/bash
# 在 openpi_train 容器内跑。宿主机也可: docker exec openpi_train bash /workspace/openpi/cigai_train/openpi-jiangsuan/scripts/start_train_00036_16gpu.sh
cd /workspace/openpi/cigai_train/openpi-jiangsuan
export PYTHONPATH=$(pwd)/src:$(pwd)/packages/openpi-client/src:$(pwd)/cigia_lerobot/src
export PALIGEMMA_TOKENIZER_PATH=/workspace/openpi/models/pi05_fixed/tokenizer.model
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

setsid nohup python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=16 \
  scripts/train_from_yaml.py --config configs/train_pi05_00036_16gpu.yaml \
  > /workspace/openpi/train_00036_16gpu_subtask.log 2>&1 &
echo $!
