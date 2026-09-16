"""
CPU 冒烟测试配置：在 config_sequoia 基础上大幅压缩训练预算与数据规模，
使整条「扩展维度微调」流程能在 Mac/CPU 上数分钟内跑通。

注意：这不是真实训练，仅用于验证「原有权重 + 新维度」流程在 CPU 上可端到端执行。
"""
from config_sequoia import get_config as _base_get_config
import os as _os


def _env_int(name, default):
    v = _os.environ.get(name)
    return int(v) if v is not None else default


def _env_str(name, default):
    v = _os.environ.get(name)
    return v if v is not None else default


def get_config():
    c = _base_get_config()
    # 以下均可通过环境变量覆盖，方便在不改动代码的前提下做不同规模的 CPU 验证：
    #   SEQUOIA_CPU_DATASET_PATH   数据输出/读取目录
    #   SEQUOIA_CPU_MAX_SYMBOLS    选取的标的数量（默认 20 = 冒烟）
    #   SEQUOIA_CPU_EPOCHS         epoch 数
    #   SEQUOIA_CPU_TRAIN_ITER     单 epoch 训练样本上限（设很大 = 用满全部自然样本）
    #   SEQUOIA_CPU_VAL_ITER       单 epoch 验证样本上限
    c.dataset_path = _env_str("SEQUOIA_CPU_DATASET_PATH", "./data/sequoia_cpu")
    c.max_symbols = _env_int("SEQUOIA_CPU_MAX_SYMBOLS", 20)
    c.epochs = _env_int("SEQUOIA_CPU_EPOCHS", 1)
    c.n_train_iter = _env_int("SEQUOIA_CPU_TRAIN_ITER", 200)
    c.n_val_iter = _env_int("SEQUOIA_CPU_VAL_ITER", 40)
    c.batch_size = 8
    c.num_workers = 0
    c.tokenizer_learning_rate = 5e-4
    c.predictor_learning_rate = 5e-4
    return c
