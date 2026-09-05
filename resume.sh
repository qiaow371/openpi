#!/bin/bash

# 训练配置文件
CONFIG="configs/train_pi05_00060.yaml"

# 循环直到手动停止（Ctrl+C）
while true; do
    echo "========================================="
    echo "启动训练（如果存在检查点则自动恢复）"
    echo "时间: $(date)"
    echo "========================================="

    # 执行训练命令，如果崩溃则退出码非0，循环会重试
    uv run scripts/train_from_yaml.py --config "$CONFIG" --resume 

    # 捕获退出码
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "训练正常完成（所有 epoch 结束），脚本退出。"
        break
    else
        echo "训练异常退出（退出码: $EXIT_CODE），可能发生了 OOM 或其他错误。"
        echo "等待 10 秒后自动恢复..."
        sleep 10
        # 如果是因为 OOM 导致，建议额外等待系统回收内存
        # 可以加上清理缓存的命令（可选）
        # echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1
    fi
done