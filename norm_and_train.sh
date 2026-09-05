#!/bin/bash
# 遇到任何错误立即退出
set -e

echo "================================================"
echo "Step 1: 计算数据集归一化统计量"
echo "================================================"
uv run scripts/compute_norm_stats.py \
  --config-name pi05_aloha \
  --dataset-dir /home/cqcigai/datasets/00063_20260813_RM_01_YJ_21

echo "================================================"
echo "Step 1 完成，开始 Step 2: 训练模型"
echo "================================================"
uv run scripts/train_from_yaml.py \
  --config configs/train_pi05.yaml

echo "================================================"
echo "全部任务执行完毕！"
echo "================================================"