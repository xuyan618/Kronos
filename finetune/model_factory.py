"""
model_factory: 根据环境变量 SEQUOIA=1 切换 tokenizer 实现。

- SEQUOIA 未设置: 使用原始 KronosTokenizer（官方 csi300 / 通用流程）。
- SEQUOIA=1     : 使用 KronosTokenizerExtended（在原有权重上引入新维度）。

Predictor (Kronos) 的 token 词表空间不变，因此无论走哪条路都用 Kronos.from_pretrained。
"""

import os

from model.kronos import KronosTokenizer, Kronos
from model.extended_tokenizer import KronosTokenizerExtended


def build_tokenizer(config, finetuned=False):
    if os.environ.get("SEQUOIA") != "1":
        path = config['finetuned_tokenizer_path'] if finetuned else config['pretrained_tokenizer_path']
        return KronosTokenizer.from_pretrained(path)

    # ---- Sequoia 扩展路径 ----
    d_in_orig = config.get('d_in_orig', 6)
    d_in_new = config.get('d_in_new', 1)
    if finetuned:
        fp = config['finetuned_tokenizer_path']
        if os.path.exists(fp):
            # 训练 tokenizer 阶段已保存为 extended 模型，直接按子类加载即可
            # （save_pretrained 写入的 config.json 含 d_in_orig / d_in_new）
            return KronosTokenizerExtended.from_pretrained(fp)
        # 微调 tokenizer 产物缺失（例如全新环境尚未跑 tokenizer 微调）：
        # 自动回退为 base 扩展版（新维度随机初始化，冻结），保证流水线可跑通。
        # 若后续把微调产物放到 finetuned_tokenizer_path，会优先使用。
        print(f"[model_factory] 微调 tokenizer 缺失({fp})，回退为 base 扩展版"
              f"(新维度 d_in_new={d_in_new} 随机初始化)")
    return KronosTokenizerExtended.from_pretrained_extended(
        config['pretrained_tokenizer_path'], d_in_new, d_in_orig
    )


def build_predictor(config):
    # Predictor 结构不变，token 空间不变，直接复用预训练权重后微调
    return Kronos.from_pretrained(config['pretrained_predictor_path'])
