"""临时诊断：验证 FinBERT 情感 -> 融合层 端到端跑通（不依赖 Kronos 模型）。

运行：在 Kronos 根目录  python check_fusion.py
"""
from app.decision import generate_trade_signal
from app.sentiment import FinbertSentimentScorer

CACHE_PATH = "app/sentiment/sentiment_cache.pkl"

# 用之前分布验证里已知方向的两组 + 一个对照
PAIRS = [
    ("600603", "2024-04-15"),   # 之前 POS 样例（涨幅偏离）
    ("300262", "2024-06-04"),   # 之前 NEG 样例（跌幅偏离）
    ("000001", "2024-01-02"),   # 对照：可能中性
]


def main():
    scorer = FinbertSentimentScorer(cache_path=CACHE_PATH)
    for sym, d in PAIRS:
        sig = generate_trade_signal(
            sym, d,
            history=None,          # adapter=None，Kronos 方向取 0
            adapter=None,
            scorer=scorer,
        )
        print(f"{sym} {d}:")
        print(f"  方向={sig.direction} combined={sig.combined_score} "
              f"kronos={sig.kronos_direction} sentiment={sig.sentiment_score} "
              f"conf={sig.confidence} n_texts={sig.n_texts}")


if __name__ == "__main__":
    main()
