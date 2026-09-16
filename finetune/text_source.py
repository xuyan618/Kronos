"""
text_source.py —— 构建「价格—文本」按 symbol+date 对齐的文本语料。

支持两类来源（可同时开启，即「两者结合」）：

1. 因子文本（factor）：直接来自 sequoia_v2.db 中已有的因子表
   - factor_dragon_tiger.reason  龙虎榜上榜原因
   - factor_theme.reason / tags  题材概念理由与标签
   这些字段本身就是按 symbol+date 对齐的现成文本，无需额外新闻语料。

2. 外部新闻（news）：可插拔。若数据库中存在 `news` 表
   （建议列：symbol, date, title, content），会自动读取并合并；
   不存在则静默跳过。这样后续接入新闻语料无需改动代码。

未来泄漏控制
------------
因子表带 `announce_date`（披露日）。若披露日晚于因子所属交易日，
说明该信息在当日并不可知，会被过滤掉，避免训练时引入未来信息。
"""

import os
import sqlite3

import pandas as pd

DB_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sequoia_v2.db"
)

# （表, 文本列列表）
FACTOR_TEXT_SOURCES = [
    ("factor_dragon_tiger", ["reason"]),
    ("factor_theme", ["reason", "tags"]),
]

NEWS_TABLE = "news"
NEWS_TEXT_COLS = ["title", "content"]


def _table_exists(con, name):
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _cols_of(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _read_source(con, table, text_cols, date_col="date"):
    """读取一张因子/新闻表，返回 DataFrame[symbol, date, text]。"""
    if not _table_exists(con, table):
        return None
    cols = _cols_of(con, table)
    if "symbol" not in cols or date_col not in cols:
        return None
    use_cols = ["symbol", date_col] + [c for c in text_cols if c in cols]
    if len(use_cols) <= 2:  # 没有任何文本列
        return None

    # 有 announce_date 时一并取出，用于未来泄漏过滤
    has_announce = "announce_date" in cols
    sel = ", ".join(use_cols + (["announce_date"] if has_announce else []))
    df = pd.read_sql(f"SELECT {sel} FROM {table} WHERE {date_col} IS NOT NULL", con)
    if df.empty:
        return None

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "symbol"])

    # 合并文本列
    parts = []
    for c in text_cols:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
    if not parts:
        return None
    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p
    df["text"] = text.str.strip()
    df = df[df["text"].str.len() > 0]

    # 未来泄漏过滤：披露日晚于所属交易日的行丢弃
    if has_announce:
        ann = pd.to_datetime(df["announce_date"], errors="coerce")
        keep = ann.isna() | (ann <= df[date_col])
        df = df[keep]

    return df[["symbol", date_col, "text"]].rename(columns={date_col: "date"})


def build_text_by_symbol(db_path=DB_PATH_DEFAULT, symbols=None, sources=("factor", "news")):
    """
    构建 {symbol: {date_str(YYYY-MM-DD): text}}。

    Args:
        db_path: sqlite 路径
        symbols: 可选的标的集合，只保留这些标的（建议传入价格数据的标的，提速）
        sources: ('factor', 'news') 的任意组合
    """
    assert os.path.exists(db_path), f"DB not found: {db_path}"
    con = sqlite3.connect(db_path)

    frames = []
    if "factor" in sources:
        for table, cols in FACTOR_TEXT_SOURCES:
            try:
                df = _read_source(con, table, cols)
            except Exception as e:  # 单表异常不影响整体
                print(f"[text_source] skip {table}: {e}")
                df = None
            if df is not None and not df.empty:
                print(f"[text_source] {table}: {len(df)} rows")
                frames.append(df)
    if "news" in sources:
        try:
            df = _read_source(con, NEWS_TABLE, NEWS_TEXT_COLS)
        except Exception as e:
            print(f"[text_source] skip {NEWS_TABLE}: {e}")
            df = None
        if df is not None and not df.empty:
            print(f"[text_source] {NEWS_TABLE}: {len(df)} rows")
            frames.append(df)
        else:
            print(f"[text_source] no usable `{NEWS_TABLE}` table (外部新闻未接入，跳过)")
    con.close()

    if not frames:
        print("[text_source] WARNING: no text source available")
        return {}

    all_df = pd.concat(frames, ignore_index=True)

    if symbols is not None:
        sym_set = set(symbols)
        all_df = all_df[all_df["symbol"].isin(sym_set)]
    if all_df.empty:
        print("[text_source] WARNING: no text after symbol filtering")
        return {}

    # 同一 (symbol, date) 的多条文本按空格拼接
    all_df["date_key"] = all_df["date"].dt.strftime("%Y-%m-%d")
    grouped = all_df.groupby(["symbol", "date_key"], sort=False)["text"].apply(
        lambda s: " ".join(s.tolist())
    )

    out = {}
    for (sym, dkey), txt in grouped.items():
        out.setdefault(sym, {})[dkey] = txt

    n_dates = sum(len(v) for v in out.values())
    print(f"[text_source] built text for {len(out)} symbols, {n_dates} (symbol, date) pairs")
    return out
