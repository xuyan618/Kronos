#!/usr/bin/env bash
# 任务1：严格时间外（temporal OOS）融合模型训练 + 回测
#
# ✅ 数据现状（已回补）：
#   - 因子文本（factor_dragon_tiger/factor_theme）已回溯覆盖 2024-01-02 ~ 2026-09-15
#   - 新闻（东财 7×24 + 原 fetch_news）已回溯覆盖 2024-01-02 ~ 2026-09-14
#   因此三段「均可含文本」，可做干净的非重叠严格时间切分（无未来泄漏）：
#     - 训练：2024-01-02 ~ 2025-09-30  （价格 + 文本，主训练）
#     - 验证：2025-10-01 ~ 2026-03-31  （价格 + 文本，早停）
#     - 测试：2026-04-01 ~ 2026-09-14  （价格 + 文本，OOS 评测）
#   所有区间均可通过环境变量覆盖（见下方 RANGE / MIN_LEN 段）。
#
# ⚠️ 运行前提：先等后台链「新闻回补」跑完，使 sequoia_v2.db 含 2024~2026 全历史新闻；
#   本脚本会自行从 DB 重建全历史文本嵌入（含因子+新闻），无需外部 pkl。
#
# 用法:
#   bash run_sequoia_fusion_temporal.sh
# 可选环境变量：
#   SEQUOIA_DB             数据库路径（默认 BigAData v2）
#   SEQUOIA_CPU_MAX_SYMBOLS  训练标的数量（默认 300；CPU 200~400，GPU 可调大）
#   FUSION_TRAIN_ITER     训练迭代步数上限（默认 5000，到达即停；设极大=用满全量）
#   FUSION_VAL_ITER       验证批次数上限（默认 200）
#   FUSION_BATCH_SIZE     批大小（默认 16；CPU 建议 16~32，看内存而定）
#   FUSION_EPOCHS         训练轮数（默认 1；n_train_iter 先到则提前停）
#   MAX_WINDOW            回测收集交易日数（默认 250，需 ≥ 测试段长度）
#   SEQUOIA_DETERMINISTIC 1=固定线程+种子，保证 OOS 可复现（默认 1）
set -e

export SEQUOIA_FUSION=1
export SEQUOIA_DB="${SEQUOIA_DB:-/Users/xuyan/Desktop/BigAData/sequoia_v2.db}"
export SEQUOIA_CPU_DATASET_PATH="./data/sequoia_fusion_temporal"
export SEQUOIA_CPU_MAX_SYMBOLS=${SEQUOIA_CPU_MAX_SYMBOLS:-300}
export FUSION_TRAIN_ITER=${FUSION_TRAIN_ITER:-5000}
export FUSION_VAL_ITER=${FUSION_VAL_ITER:-200}
export FUSION_BATCH_SIZE=${FUSION_BATCH_SIZE:-16}
export FUSION_EPOCHS=${FUSION_EPOCHS:-1}
export SEQUOIA_DETERMINISTIC=${SEQUOIA_DETERMINISTIC:-1}
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):$PYTHONPATH"

# 时间切分区间（全历史严格时间切分：三段均有文本）
export TRAIN_RANGE_START=${TRAIN_RANGE_START:-2024-01-02}
export TRAIN_RANGE_END=${TRAIN_RANGE_END:-2025-09-30}
export VAL_RANGE_START=${VAL_RANGE_START:-2025-10-01}
export VAL_RANGE_END=${VAL_RANGE_END:-2026-03-31}
export TEST_RANGE_START=${TEST_RANGE_START:-2026-04-01}
export TEST_RANGE_END=${TEST_RANGE_END:-2026-09-14}

# loader 最小行数：训练段保留默认下限(101)，验证/测试段放开（0=不丢样本）
export TRAIN_MIN_LEN=${TRAIN_MIN_LEN:-101}
export VAL_MIN_LEN=${VAL_MIN_LEN:-0}
export TEST_MIN_LEN=${TEST_MIN_LEN:-0}

echo "== 1. 生成价格数据（全历史严格时间切分：train/val/test 各自非空）=="
mkdir -p "$SEQUOIA_CPU_DATASET_PATH"
python -u finetune/sequoia_db_loader.py

echo "== 2. 重建全历史文本嵌入（因子+新闻，从 DB 直接读；覆盖 loader 生成的全部标的）=="
python -u finetune/build_text_embeddings.py

echo "== 3. 微调 KronosFusion（严格时间外）=="
python -u finetune/train_predictor_fusion.py

echo "== 4. 确定性回测（OOS 测试集全量）=="
MAX_WINDOW=250 python -u finetune/backtest_predictor_fusion.py

echo "任务1 完成。OOS 结果见 ./outputs/models/sequoia_predictor_fusion/backtest/"
