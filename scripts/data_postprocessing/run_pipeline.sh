#!/bin/bash
# LeRobot v2.1 数据集后处理 Pipeline 快捷脚本
#
# 用法:
#   ./run_pipeline.sh <dataset_path> [options]
#
# 示例:
#   # 完整 pipeline（检查 + 修复 + 统计）
#   ./run_pipeline.sh /nvme2n1/dataset/00070_20260817_agx01_gzy
#
#   # 仅检查
#   ./run_pipeline.sh /nvme2n1/dataset/00070_20260817_agx01_gzy --check-only
#
#   # 高并行度
#   ./run_pipeline.sh /nvme2n1/dataset/00070_20260817_agx01_gzy -j 8
#
#   # 远程执行（江算）
#   ssh gpu-master "ssh root@172.30.61.36 \\
#     '/home/ubuntu/lerobot/.venv/bin/python3 \\
#     /home/ubuntu/cigai_train/openpi-jiangsuan/scripts/data_postprocessing/run_pipeline.sh \\
#     /nvme2n1/dataset/<dataset_name>'"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if [ $# -eq 0 ]; then
    echo "用法: $0 <dataset_path> [options]"
    echo ""
    echo "Options:"
    echo "  --check-only    仅检查，不修复"
    echo "  --fix           自动修复损坏视频"
    echo "  --verify        修复后验证"
    echo "  --no-backup     不保留 .bak 备份"
    echo "  -j N            并行 worker 数 (default: 4)"
    exit 1
fi

DATASET_PATH="$1"
shift

# 默认启用 fix + verify（除非 --check-only）
ARGS=("--fix" "--verify")

for arg in "$@"; do
    if [ "$arg" = "--check-only" ]; then
        ARGS=()  # 清空 fix/verify
    fi
    ARGS+=("$arg")
done

echo "============================================"
echo "  LeRobot v2.1 Dataset Post-Processing"
echo "============================================"
echo "  Dataset: $(basename "$DATASET_PATH")"
echo "  Path:    $DATASET_PATH"
echo "  Args:    ${ARGS[*]:-check-only}"
echo "============================================"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/pipeline.py" "$DATASET_PATH" "${ARGS[@]}"
