#!/usr/bin/env bash
# 在预训练 Kronos 权重基础上引入新维度并微调（方向 1-A）。
# 用法:  bash run_sequoia_finetune.sh
set -e

# 启用 Sequoia 扩展路径（model_factory 据此加载 KronosTokenizerExtended）
export SEQUOIA=1
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):$PYTHONPATH"

echo "== 1. 从 sequoia_v2.db 生成 pickle 数据集 =="
python finetune/sequoia_db_loader.py

echo "== 2. 微调扩展 tokenizer（复用原权重，新增维度走并行支路）=="
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py

echo "== 3. 微调 predictor（token 空间不变，复用原权重）=="
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py

echo "完成。checkpoint 在 ./outputs/models/sequoia_tokenizer 与 sequoia_predictor"
