"""从网络(新浪/akshare)补全 sequoia ``stock_daily`` 到本地 SQLite，供 Kronos loader 使用。

复用 Sequoia-X 项目的 ``DataEngine.backfill_sina()``（新浪源、后复权、自带
outstanding_share），其写入的 ``stock_daily`` 表列
(symbol, date, open, high, low, close, volume, turnover, outstanding_share)
正好是 Kronos ``finetune/sequoia_db_loader.py`` 读取的列。

仅下载与现有文本嵌入完全对齐的 195 只标的（来自
``data/sequoia_fusion_cpu/text_embeddings.pkl`` 的 data 键），
保证后续融合训练时价格与文本键一致。

用法:
    python finetune/download_sequoia_db.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

# Sequoia-X 项目路径（数据补全逻辑所在地），加入 sys.path 以便 import sequoia_x
SEQUOIA_X_ROOT = "/Users/xuyan/Desktop/Sequoia-X"
if SEQUOIA_X_ROOT not in sys.path:
    sys.path.insert(0, SEQUOIA_X_ROOT)

from sequoia_x.core.config import Settings  # noqa: E402
from sequoia_x.data.engine import DataEngine  # noqa: E402

KRONOS = Path("/Users/xuyan/Desktop/Kronos")
TARGET_DB = str(KRONOS / "data" / "sequoia_v2_downloaded.db")
START_DATE = "2024-01-01"
SYMBOLS_PKL = KRONOS / "data" / "sequoia_fusion_cpu" / "text_embeddings.pkl"
WORKERS = 4


def load_symbols() -> list[str]:
    te = pickle.load(open(SYMBOLS_PKL, "rb"))
    return sorted(te["data"].keys())


def main() -> None:
    symbols = load_symbols()
    print(f"[download] {len(symbols)} symbols -> {TARGET_DB} (start={START_DATE})", flush=True)
    settings = Settings(
        db_path=TARGET_DB,
        start_date=START_DATE,
        feishu_webhook_url="https://example.com/placeholder",
    )
    engine = DataEngine(settings)
    engine.backfill_sina(symbols=symbols, workers=WORKERS, reset=False)

    # 小结
    import sqlite3

    with sqlite3.connect(TARGET_DB) as conn:
        n_rows = conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
        n_syms = conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_daily").fetchone()[0]
        dmin, dmax = conn.execute(
            "SELECT MIN(date), MAX(date) FROM stock_daily"
        ).fetchone()
    print(f"[download] done: rows={n_rows} symbols={n_syms} range={dmin}..{dmax}", flush=True)


if __name__ == "__main__":
    main()
