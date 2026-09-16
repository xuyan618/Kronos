#!/usr/bin/env bash
# CPU 冒烟测试：在 Mac/CPU 上验证「扩展维度微调」整条流程可端到端执行。
# 使用极小数据(约 20 只标的) + 极小训练预算，数分钟内跑完，仅用于验证，非真实训练。
# 用法:  bash run_sequoia_cpu.sh
set -e

export SEQUOIA=1
export SEQUOIA_CPU=1
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):$PYTHONPATH"

echo "== 1. 生成极小 CPU 数据集（约 20 只标的）=="
python finetune/sequoia_db_loader.py

echo "== 2. 微调扩展 tokenizer (CPU 单进程) =="
python finetune/train_tokenizer.py

echo "== 3. 微调 predictor (CPU 单进程) =="
python finetune/train_predictor.py

echo "CPU 冒烟测试完成。checkpoint 在:"
echo "  tokenizer:  ./outputs/models/sequoia_tokenizer/checkpoints/best_model"
echo "  predictor:  ./outputs/models/sequoia_predictor/checkpoints/best_model"
