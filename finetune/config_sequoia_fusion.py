"""
前期融合（Early Fusion）配置 —— 在 config_sequoia_cpu 基础上增加「文本模态」。

文本来源（可同时开启，即「两者结合」）：
    factor  因子文本：factor_dragon_tiger.reason、factor_theme.reason/tags
            这些字段已按 symbol+date 对齐，来自现有 sequoia_v2.db，无需额外语料。
    news    外部新闻：可插拔。若数据库中存在 `news(symbol, date, title, content)`
            表则自动读取合并；不存在则静默跳过。

文本编码器：
    TEXT_ENCODER_MODE=auto|finbert|hash
    - finbert 真实 FinBERT [CLS] 嵌入（768 维，需要 pip install transformers）
    - hash    离线确定性哈希兜底（无依赖、无联网，用于跑通通路）
    - auto    优先 finbert，不可用时自动降级为 hash（默认）
    默认 FinBERT 模型为「中文金融」yiyanghkust/finbert-tone-chinese
    （bert-base-chinese，768 维 [CLS]），对 A 股因子文本比英文 ProsusAI/finbert 更贴合；
    可用 TEXT_FINBERT_MODEL 覆盖。

融合布局：
    FUSION_MODE=interleave|prepend
    - interleave（默认，因果正确）[t_0,p_0,t_1,p_1,...]
    - prepend（仅消融对照，会让价格 token 看到未来文本，存在泄漏）
"""
import os

from config_sequoia_cpu import get_config as _base_get_config


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_str(name, default):
    v = os.environ.get(name)
    return v if v is not None else default


def get_config():
    c = _base_get_config()

    # 数据与权重目录与非融合流程隔离，避免互相覆盖
    c.dataset_path = _env_str("SEQUOIA_CPU_DATASET_PATH", "./data/sequoia_fusion_cpu")
    c.predictor_save_folder_name = "sequoia_predictor_fusion"

    # ---- 文本来源 ----
    c.text_sources = ("factor", "news")
    c.text_embeddings_file = _env_str("TEXT_EMBEDDINGS_FILE", "text_embeddings.pkl")

    # ---- 文本编码器 ----
    c.text_encoder_mode = _env_str("TEXT_ENCODER_MODE", "auto")
    c.text_finbert_model = _env_str("TEXT_FINBERT_MODEL", "yiyanghkust/finbert-tone-chinese")
    c.text_hash_dim = _env_int("TEXT_HASH_DIM", 128)
    c.text_max_length = _env_int("TEXT_MAX_LENGTH", 64)
    c.text_encode_batch_size = _env_int("TEXT_ENCODE_BATCH", 16)

    # ---- 融合 ----
    c.fusion_mode = _env_str("FUSION_MODE", "interleave")
    c.use_modality_emb = True

    # ---- CPU 冒烟预算 ----
    c.epochs = _env_int("FUSION_EPOCHS", 1)
    c.n_train_iter = _env_int("FUSION_TRAIN_ITER", 200)
    c.n_val_iter = _env_int("FUSION_VAL_ITER", 40)
    c.batch_size = _env_int("FUSION_BATCH_SIZE", 8)
    c.num_workers = _env_int("FUSION_NUM_WORKERS", 0)
    return c
