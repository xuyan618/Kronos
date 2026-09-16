#!/bin/bash
# 云端融合训练：FinBERT 编码 -> 多 epoch 融合微调
# 用法：FUSION_EPOCHS=15 FUSION_TRAIN_ITER=1500 bash run_cloud_train.sh
set -e
cd /workspace
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python

export SEQUOIA_CPU=1
export SEQUOIA_FUSION=1
export SEQUOIA_CPU_DATASET_PATH=./data/sequoia_fusion_cpu
export TEXT_ENCODER_MODE=finbert
export PYTHONPATH=.
export FUSION_EPOCHS=${FUSION_EPOCHS:-15}
export FUSION_TRAIN_ITER=${FUSION_TRAIN_ITER:-1500}
export FUSION_VAL_ITER=${FUSION_VAL_ITER:-200}

echo "=== 环境 ==="
$PY --version
$PY -c "import torch; print('torch', torch.__version__)"

echo "=== Step1: FinBERT 编码 (dim=768) ==="
$PY -u finetune/build_text_embeddings.py

echo "=== Step2: 融合训练 ==="
$PY -u finetune/train_predictor_fusion.py

echo "=== DONE ==="
ls -lh outputs/models/sequoia_predictor_fusion/checkpoints/ 2>/dev/null
