"""临时诊断脚本:验证 FinBERT 对 A 股因子理由的情感区分度。
在 Kronos 根目录运行:  python check_sentiment_dist.py
"""
import sqlite3
import statistics
from collections import Counter

from app.sentiment import FinbertSentimentScorer

CACHE_PATH = "app/sentiment/sentiment_cache.pkl"
DB_PATH = "data/sequoia_v2.db"


def main():
    s = FinbertSentimentScorer(cache_path=CACHE_PATH)
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT symbol,date FROM factor_dragon_tiger "
        "WHERE reason IS NOT NULL ORDER BY RANDOM() LIMIT 200"
    ).fetchall()
    con.close()

    res = s.score_many([(r[0], str(r[1])) for r in rows], persist=True)
    scores = [r.score for r in res if r.n_texts > 0]

    print("n=", len(scores), "mean=", round(statistics.mean(scores), 3),
          "min=", min(scores), "max=", max(scores))
    print(Counter("pos" if x > 0.2 else "neg" if x < -0.2 else "neu" for x in scores))
    for r in sorted(res, key=lambda x: x.score)[:3]:
        print("NEG", r.symbol, r.date, r.score, r.texts[0][:40])
    for r in sorted(res, key=lambda x: x.score)[-3:]:
        print("POS", r.symbol, r.date, r.score, r.texts[0][:40])


if __name__ == "__main__":
    main()
