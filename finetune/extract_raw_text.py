"""
extract_raw_text.py —— 仅本地使用：把「按 symbol+date 对齐」的原始文本
从 sequoia_v2.db 抽出来，存为 <dataset_path>/raw_text.pkl（仅几 MB）。

目的：云端子训练不需要 588MB 的 db，只要这个小 pickle 即可做 FinBERT 编码。
格式与 text_source.build_text_by_symbol 的返回值一致：
    {symbol: {date_str(YYYY-MM-DD): text_string}}

用法：
    SEQUOIA_CPU_DATASET_PATH=./data/sequoia_fusion_cpu python finetune/extract_raw_text.py
"""
import os
import pickle
import sys

sys.path.append("../")
from text_source import build_text_by_symbol, DB_PATH_DEFAULT  # noqa: E402


def main():
    ds = os.environ.get("SEQUOIA_CPU_DATASET_PATH", "./data/sequoia_fusion_cpu")
    symbols = set()
    for name in ("train_data.pkl", "val_data.pkl", "test_data.pkl"):
        p = os.path.join(ds, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                symbols |= set(pickle.load(f).keys())
    print(f"[extract] 标的数: {len(symbols)}  数据集目录: {ds}")

    text_by_symbol = build_text_by_symbol(
        DB_PATH_DEFAULT, symbols=symbols, sources=("factor", "news")
    )

    out = os.path.join(ds, "raw_text.pkl")
    with open(out, "wb") as f:
        pickle.dump(text_by_symbol, f)
    n_pairs = sum(len(v) for v in text_by_symbol.values())
    print(f"[extract] 已保存: {out}")
    print(f"[extract] 覆盖 {len(text_by_symbol)} 只标的 / {n_pairs} 个 (symbol, date) 对")


if __name__ == "__main__":
    main()
