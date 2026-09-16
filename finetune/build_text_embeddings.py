"""
build_text_embeddings.py —— 构建并缓存「按 symbol+date 对齐」的文本嵌入。

流程：
    1. 从价格 pickle 中取到标的集合；
    2. text_source.build_text_by_symbol() 汇总因子文本（+ 可选外部新闻）；
    3. text_encoder 编码（FinBERT 或离线 hash 兜底），相同文本只编码一次；
    4. 落到 <dataset_path>/text_embeddings.pkl，供 QlibFusionDataset 直接读取。

用法:
    python finetune/build_text_embeddings.py
（需先跑过 sequoia_db_loader.py 生成价格 pickle）
"""

import os
import pickle
import sys

import numpy as np

sys.path.append('../')
from config_sequoia_fusion import get_config  # noqa: E402
from text_source import build_text_by_symbol, DB_PATH_DEFAULT  # noqa: E402
from text_encoder import build_encoder  # noqa: E402


def main():
    cfg = get_config()
    ds_path = cfg.dataset_path

    # 1) 标的集合
    symbols = set()
    for name in ('train_data.pkl', 'val_data.pkl', 'test_data.pkl'):
        p = os.path.join(ds_path, name)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                symbols |= set(pickle.load(f).keys())
        else:
            print(f"[text-emb] WARNING: 缺少 {p}")
    symbols = sorted(symbols)
    print(f"[text-emb] 标的数: {len(symbols)}, 数据集目录: {ds_path}")

    # 2) 文本来源（因子文本 + 可选新闻）
    # 若已本地抽取好 raw_text.pkl（无 db 环境，如云端子训练），直接加载
    raw_path = os.path.join(ds_path, "raw_text.pkl")
    if os.path.exists(raw_path):
        print(f"[text-emb] 从 raw_text.pkl 加载文本（无需 db）: {raw_path}")
        with open(raw_path, "rb") as f:
            text_by_symbol = pickle.load(f)
    else:
        text_by_symbol = build_text_by_symbol(
            DB_PATH_DEFAULT, symbols=symbols, sources=cfg.text_sources
        )

    # 3) 编码（去重）
    enc = build_encoder(
        mode=cfg.text_encoder_mode,
        hash_dim=cfg.text_hash_dim,
        finbert_model=cfg.text_finbert_model,
    )

    unique_texts = []
    text_index = {}
    for _sym, dmap in text_by_symbol.items():
        for _d, t in dmap.items():
            if t not in text_index:
                text_index[t] = len(unique_texts)
                unique_texts.append(t)

    if unique_texts:
        print(f"[text-emb] 编码 {len(unique_texts)} 条去重文本...")
        embs = enc.encode(unique_texts, batch_size=cfg.text_encode_batch_size)
    else:
        print("[text-emb] WARNING: 没有任何文本，将生成空嵌入表（全零文本向量）")
        embs = np.zeros((0, enc.dim_out), dtype=np.float32)

    # 4) 组装并保存
    data = {}
    for sym, dmap in text_by_symbol.items():
        dates = sorted(dmap.keys())
        arr = np.stack([embs[text_index[dmap[d]]] for d in dates]).astype(np.float32)
        data[sym] = {'dates': dates, 'emb': arr}

    out = {
        'dim': int(embs.shape[1]) if embs.ndim == 2 and embs.shape[0] > 0 else int(enc.dim_out),
        'meta': {
            'encoder_mode': cfg.text_encoder_mode,
            'finbert_model': cfg.text_finbert_model,
            'hash_dim': cfg.text_hash_dim,
            'sources': list(cfg.text_sources),
            'n_symbols': len(data),
            'n_unique_texts': len(unique_texts),
        },
        'data': data,
    }

    os.makedirs(ds_path, exist_ok=True)
    out_path = os.path.join(ds_path, cfg.text_embeddings_file)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)

    n_pairs = sum(len(v['dates']) for v in data.values())
    print(f"[text-emb] 已保存: {out_path}")
    print(f"[text-emb] 覆盖 {len(data)} 只标的 / {n_pairs} 个 (symbol, date) 对, dim={out['dim']}")


if __name__ == '__main__':
    main()
