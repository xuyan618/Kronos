#!/usr/bin/env bash
# 前期融合（Early Fusion）CPU 冒烟测试：
#   价格 token + 文本 token(因子文本/新闻) 拼进 predictor 输入，跨模态联合预测。
#
# 三步：
#   1. 生成价格 pickle
#   2. 构建按 symbol+date 对齐的文本嵌入（因子文本 + 可选外部新闻）
#   3. 微调 KronosFusion（文本通道随训练学习，backbone 复用预训练权重）
#
# 用法:  bash run_sequoia_fusion_cpu.sh
#
# 可选环境变量：
#   SEQUOIA_CPU_MAX_SYMBOLS=20      标的数量
#   TEXT_ENCODER_MODE=hash          文本编码器：hash(离线兜底) / auto / finbert
#                                   （真实训练建议改成 auto 或 finbert，需 pip install transformers）
#   FUSION_MODE=interleave          融合布局：interleave(因果正确) / prepend(消融，有泄漏)
set -e

export SEQUOIA=1
export SEQUOIA_CPU=1
export SEQUOIA_FUSION=1
export SEQUOIA_CPU_DATASET_PATH="./data/sequoia_fusion_cpu"
export SEQUOIA_CPU_MAX_SYMBOLS=${SEQUOIA_CPU_MAX_SYMBOLS:-20}
export TEXT_ENCODER_MODE=${TEXT_ENCODER_MODE:-hash}
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):$PYTHONPATH"

# 现有因子文本只覆盖 2026-05-07 ~ 2026-09-11（约 90 个交易日），
# 与默认训练区间(2024-01~2025-06)完全不重叠，会导致文本覆盖率 0%。
# 因此融合冒烟把区间指到「与文本重叠」的最近时段，让文本真正参与融合。
# 注意：本冒烟的 train/val 区间有重叠，仅用于验证通路，不用于评估模型质量。
export TRAIN_RANGE_START=${TRAIN_RANGE_START:-2026-02-01}
export TRAIN_RANGE_END=${TRAIN_RANGE_END:-2026-09-11}
export VAL_RANGE_START=${VAL_RANGE_START:-2026-03-01}
export VAL_RANGE_END=${VAL_RANGE_END:-2026-09-11}

echo "== 1. 生成价格数据（约 ${SEQUOIA_CPU_MAX_SYMBOLS} 只标的）=="
python -u finetune/sequoia_db_loader.py

echo "== 2. 构建文本嵌入（因子文本 + 外部新闻，mode=${TEXT_ENCODER_MODE}）=="
python -u finetune/build_text_embeddings.py

echo "== 3. 前期融合微调 predictor (CPU 单进程) =="
python -u finetune/train_predictor_fusion.py

echo "前期融合 CPU 冒烟完成。checkpoint 在:"
echo "  ./outputs/models/sequoia_predictor_fusion/checkpoints/best_model"
