"""
backfill_news_724.py —— 用东财全球资讯 7×24 快讯补全 news 表的历史新闻。

为什么能补全
------------
东财 7×24 快讯接口(np-weblist)按 realSort(epoch 微秒)降序分页,每条快讯
带 stockList 字段(关联个股代码)。通过 sortEnd 逐页回翻,可一路翻到 2024 年;
再把「命中训练标的」的快讯写进 news(symbol, date, title, ...),即可让新闻文本
覆盖 2024-2025——免费源里唯一能回溯的历史新闻渠道(东财个股新闻搜索只能取到
最近两个月,已实测)。

落库
----
直接写入与 fetch_news.py 完全相同的 news 表(symbol, date, title, content,
source, url, created_at),UNIQUE(symbol, date, title) 去重。重跑
build_text_embeddings 时会自动把这批新闻与因子文本合并,无需改任何代码。

注意 / 风控
----------
- 这是东财接口,有风控。务必串行 + 限速(默认 1.2s)。被临时封禁时脚本可断点
  续传(检查点文件 .news724_checkpoint 记录上次 sortEnd)。
- 只写入 stockList 命中训练标的的快讯,避免无限膨胀。
- 与 factor 回补共享东财配额,若两者同时跑建议加大 --sleep 或错开。

用法
----
  # 补全 2024-01-02 ~ 2026-03-11(已有 2026-03-12~09-12 由 fetch_news 提供)
  python finetune/backfill_news_724.py --start 2024-01-02 --end 2026-03-11
  # 全量(含已覆盖区间,INSERT OR IGNORE 自动去重)
  python finetune/backfill_news_724.py --start 2024-01-02 --end 2026-09-14 --sleep 1.5
  # 清掉断点重来
  rm -f finetune/.news724_checkpoint
"""
import argparse
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DB = os.path.join(ROOT, "data", "sequoia_v2.db")
DEFAULT_DATASET = os.path.join(ROOT, "data", "sequoia_fusion_cpu")
CHECKPOINT = os.path.join(HERE, ".news724_checkpoint")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
URL = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"


def norm_code(raw):
    """'0.159940' / '90.BK0464' -> '159940';板块/指数(BKxxxx)或非6位返回 None。"""
    if not raw or "." not in raw:
        return None
    code = raw.split(".", 1)[1]
    if len(code) != 6 or not code.isdigit() or code.startswith("BK"):
        return None
    return code


def load_universe(dataset):
    """训练标的(6位代码集合),缺省回退全市场。"""
    pkl = os.path.join(dataset, "train_data.pkl")
    if os.path.exists(pkl):
        try:
            import pickle
            with open(pkl, "rb") as f:
                syms = set(pickle.load(f).keys())
            print(f"[news724] 训练标的 {len(syms)} 只(来自 train_data.pkl)")
            return syms
        except Exception as exc:  # noqa: BLE001
            print(f"[news724] 读取 train_data.pkl 失败: {exc}")
    con = sqlite3.connect(DEFAULT_DB)
    try:
        syms = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM stock_daily")}
    finally:
        con.close()
    print(f"[news724] 全市场标的 {len(syms)} 只")
    return syms


def fetch_page(sort_end, page_size=50, timeout=10):
    params = {
        "client": "web", "biz": "web_724", "fastColumn": "102",
        "sortEnd": str(sort_end), "pageSize": str(page_size),
        "req_trace": str(uuid.uuid4()),
    }
    headers = {"User-Agent": UA, "Referer": "https://kuaixun.eastmoney.com/"}
    r = requests.get(URL, params=params, headers=headers, timeout=timeout)
    d = r.json()
    return d.get("data", {}).get("fastNewsList", []) or []


def parse_showtime(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="用东财7x24快讯回补历史新闻(news表)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--start", default="2024-01-02", help="起始日期 YYYY-MM-DD")
    ap.add_argument("--end", default="2026-09-14", help="结束日期 YYYY-MM-DD")
    ap.add_argument("--sleep", type=float, default=1.2, help="每次请求间隔(秒,限速防封)")
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--source", default="东财7x24")
    ap.add_argument("--max-items", type=int, default=0, help="最多处理多少条快讯(0=不限)")
    ap.add_argument("--no-resume", action="store_true", help="忽略断点,从头开始")
    args = ap.parse_args()

    universe = load_universe(args.dataset)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    # 起始 sortEnd: end 之后一天的 00:00(取该时刻之前 = end 当天及更早)
    sort_end = int((end_dt + timedelta(days=1)).timestamp() * 1_000_000)

    # 断点续传
    if not args.no_resume and os.path.exists(CHECKPOINT):
        try:
            v = int(open(CHECKPOINT).read().strip())
            sort_end = v
            print(f"[news724] 续传自 sortEnd={sort_end}")
        except Exception:
            pass

    con = sqlite3.connect(args.db)
    con.execute(
        """CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, date TEXT NOT NULL, title TEXT NOT NULL,
            content TEXT, source TEXT, url TEXT, created_at TEXT,
            UNIQUE(symbol, date, title))"""
    )
    con.commit()

    total_written = 0
    items_seen = 0
    t0 = time.time()
    stop = False
    empty_streak = 0

    while not stop:
        page = fetch_page(sort_end, args.page_size)
        if not page:
            empty_streak += 1
            if empty_streak >= 3:
                print("[news724] 连续空页,结束")
                break
            time.sleep(args.sleep)
            continue
        empty_streak = 0

        last_sort = None
        for it in page:
            st = parse_showtime(it.get("showTime", ""))
            if st is None:
                continue
            last_sort = it.get("realSort")
            if st < start_dt:           # 已越过起点(降序,之后都更老)
                stop = True
                break
            items_seen += 1
            date_str = st.strftime("%Y-%m-%d")
            title = (it.get("title") or "").strip()
            if not title:
                continue
            summary = (it.get("summary") or "").strip()
            codes = {
                c for s in (it.get("stockList") or [])
                if (c := norm_code(s)) and c in universe
            }
            if not codes:
                continue
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows = [(c, date_str, title, summary, args.source, "", now) for c in codes]
            try:
                cur = con.executemany(
                    "INSERT OR IGNORE INTO news "
                    "(symbol,date,title,content,source,url,created_at) "
                    "VALUES (?,?,?,?,?,?,?)", rows)
                total_written += cur.rowcount or 0
            except sqlite3.Error as exc:
                print(f"[news724] DB err: {exc}")

        if last_sort:
            sort_end = int(last_sort)
            open(CHECKPOINT, "w").write(str(sort_end))
        con.commit()

        if args.max_items and items_seen >= args.max_items:
            print("[news724] 达到 max-items,停止")
            break
        time.sleep(args.sleep)

    con.close()
    print(f"[news724] 完成: 看到 {items_seen} 条快讯, news 表新增 {total_written} 行, "
          f"耗时 {time.time() - t0:.0f}s")
    print(f"[news724] 断点: {CHECKPOINT} (重跑自动续传; 清掉则从头)")


if __name__ == "__main__":
    sys.exit(main())
