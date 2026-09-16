"""
Sequoia (A 股日频) 微调配置 —— 在预训练权重上引入新维度。

与官方 config.py 的区别:
- dataset_path 指向由 sequoia_db_loader.py 生成的 pickle
- pretrained_* 指向 HuggingFace 上的 Kronos 预训练权重
- feature_list 在原始 6 维基础上增加了 outstanding_share（流通股本，日频、全量覆盖）
- 新增 d_in_orig / d_in_new 描述扩展维度
"""

from config import Config


def get_config():
    c = Config()

    c.instrument = "sequoia_a_share"
    c.dataset_path = "./data/sequoia_processed"

    # 预训练权重（首次运行会自动从 HF 下载）
    c.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
    c.pretrained_predictor_path = "NeoQuasar/Kronos-base"

    # 保存目录
    c.tokenizer_save_folder_name = "sequoia_tokenizer"
    c.predictor_save_folder_name = "sequoia_predictor"
    c.finetuned_tokenizer_path = f"{c.save_path}/{c.tokenizer_save_folder_name}/checkpoints/best_model"
    c.finetuned_predictor_path = f"{c.save_path}/{c.predictor_save_folder_name}/checkpoints/best_model"

    # —— 扩展维度 ——
    c.d_in_orig = 6                       # 原始: open/high/low/close/vol/amt
    c.d_in_new = 1                        # 新增: outstanding_share
    c.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt', 'outstanding_share']
    c.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

    # 关掉 comet 避免外部依赖
    c.use_comet = False

    return c
