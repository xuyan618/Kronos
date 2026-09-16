#!/usr/bin/env bash
# CPU 中等压测：在 config_sequoia_cpu 基础上通过环境变量放大规模，
# 验证「扩展维度微调」在较大数据量下训练循环稳定（非真实训练）。
#   - 120 只标的（SMOKE 的 6 倍）
#   - 3 epoch
#   - 训练/验证样本上限设很大 = 使用满全部自然样本（不再封顶 200）
# 预计 CPU 单进程耗时 1~3 小时。
# 用法:  bash run_sequoia_cpu_medium.sh
set -e

export SEQUOIA=1
export SEQUOIA_CPU=1
# 规模覆盖（被 finetune/config_sequoia_cpu.py 读取）
export SEQUOIA_CPU_DATASET_PATH="./data/sequoia_cpu_medium"
export SEQUOIA_CPU_MAX_SYMBOLS=120
export SEQUOIA_CPU_EPOCHS=3
export SEQUOIA_CPU_TRAIN_ITER=1000000
export SEQUOIA_CPU_VAL_ITER=200000
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):$PYTHONPATH"

echo "== 1. 生成中等规模 CPU 数据集（约 120 只标的）=="
python -u finetune/sequoia_db_loader.py

echo "== 2. 微调扩展 tokenizer (CPU 单进程, 120 标的 × 3 epoch) =="
python -u finetune/train_tokenizer.py

echo "== 3. 微调 predictor (CPU 单进程, 120 标的 × 3 epoch) =="
python -u finetune/train_predictor.py

echo "CPU 中等压测完成。checkpoint 在:"
echo "  tokenizer:  ./outputs/models/sequoia_tokenizer/checkpoints/best_model"
echo "  predictor:  ./outputs/models/sequoia_predictor/checkpoints/best_model"
