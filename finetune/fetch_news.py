"""
fetch_news.py —— 抓取个股新闻素材并入库（供前期融合使用）。

抓到的新闻写入 sequoia_v2.db 的 `news` 表：

    news(symbol, date, title, content, source, url, created_at)
    UNIQUE(symbol, date, title)   -- 去重，重复运行会增量累积

该表与 finetune/text_source.py 的「外部新闻」接口对接：
建好这张表后，build_text_embeddings.py 会自动把新闻与因子文本合并，
无需改动任何代码。

数据源
------
eastmoney（默认）：直接调用东方财富搜索接口，支持分页，
                   只依赖标准库（urllib），无第三方依赖。
akshare          ：akshare 的 stock_news_em，每只标的仅返回最新 10 条，
                   作备用通道（--provider akshare）。

注意
----
免费财经接口通常只提供**近期**新闻，难以回溯到 2024 年的历史。
若要积累训练期文本，建议**定期（如每日）运行本脚本**，靠 UNIQUE 约束增量累积。

用法示例
--------
    # 抓取默认数据集里前 20 只标的的新闻（每只 3 页 × 50 条）
    python finetune/fetch_news.py --limit 20

    # 指定标的、抓更多页、限速
    python finetune/fetch_news.py --symbols 000001,000002 --pages 5 --page-size 50 --sleep 0.5

    # 只看不写
    python finetune/fetch_news.py --limit 3 --dry-run
"""

import argparse
import html
import json
import os
import pickle
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DB = os.path.join(ROOT, "data", "sequoia_v2.db")
DEFAULT_DATASET = os.path.join(ROOT, "data", "sequoia_fusion_cpu")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS news (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT NOT NULL,
    date       TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT,
    source     TEXT,
    url        TEXT,
    created_at TEXT,
    UNIQUE(symbol, date, title)
);
"""

CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_news_symbol_date ON news(symbol, date);"

INSERT_SQL = """
INSERT OR IGNORE INTO news (symbol, date, title, content, source, url, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_TAG_RE = re.compile(r"</?em>")


def clean_text(s):
    """去除 <em> 高亮标签与多余空白，并还原 HTML 实体。"""
    if not s:
        return ""
    s = _TAG_RE.sub("", str(s))
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------- provider: eastmoney（默认） -------------------------
def fetch_eastmoney(symbol, pages=3, page_size=50, sleep=0.3, max_retry=2, verbose=True):
    """分页抓取东方财富个股新闻，返回 [{date,title,content,source,url}, ...]。"""
    out = []
    for page in range(1, pages + 1):
        inner = {
            "uid": "",
            "keyword": symbol,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": page,
                    "pageSize": page_size,
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }
        params = {
            "cb": "jQuery_fusion",
            "param": json.dumps(inner, ensure_ascii=False),
            "_": str(int(time.time() * 1000)),
        }
        url = "https://search-api-web.eastmoney.com/search/jsonp?" + urllib.parse.urlencode(params)

        data = None
        for attempt in range(max_retry + 1):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": UA, "Referer": "https://so.eastmoney.com/"}
                )
                raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
                s, e = raw.find("("), raw.rfind(")")
                data = json.loads(raw[s + 1:e])
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retry:
                    if verbose:
                        print(f"    [warn] {symbol} page{page} 失败: {type(exc).__name__}: {exc}")
                    data = None
                else:
                    time.sleep(sleep * 2)

        if not data:
            continue
        arts = (data.get("result") or {}).get("cmsArticleWebOld") or []
        if not arts:
            break

        for a in arts:
            date_raw = str(a.get("date") or "")[:10]
            title = clean_text(a.get("title"))
            if not date_raw or not title:
                continue
            out.append({
                "date": date_raw,
                "title": title,
                "content": clean_text(a.get("content")),
                "source": a.get("mediaName") or "",
                "url": a.get("url") or "",
            })

        if len(arts) < page_size:  # 没有更多页
            break
        time.sleep(sleep)
    return out


# ------------------------- provider: akshare（备用） -------------------------
def fetch_akshare(symbol, pages=1, page_size=10, sleep=0.3, max_retry=2, verbose=True):
    """akshare stock_news_em：每只标的仅最新 10 条，无分页。"""
    try:
        import akshare as ak
    except ImportError:
        if verbose:
            print("    [warn] akshare 未安装，跳过（pip install akshare）")
        return []
    try:
        df = ak.stock_news_em(symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"    [warn] {symbol} akshare 失败: {type(exc).__name__}: {exc}")
        return []
    out = []
    for _, row in df.iterrows():
        date_raw = str(row.get("发布时间") or "")[:10]
        title = clean_text(row.get("新闻标题"))
        if not date_raw or not title:
            continue
        out.append({
            "date": date_raw,
            "title": title,
            "content": clean_text(row.get("新闻内容")),
            "source": str(row.get("文章来源") or ""),
            "url": str(row.get("新闻链接") or ""),
        })
    return out


PROVIDERS = {"eastmoney": fetch_eastmoney, "akshare": fetch_akshare}


# ------------------------- 标的来源 -------------------------
def load_symbols(args):
    if args.symbols:
        return [s.strip() for s in args.symbols.split(",") if s.strip()]

    # 全市场：直接用行情表里去重后的标的（与价格数据完全对齐）
    if args.all_symbols:
        con = sqlite3.connect(args.db)
        try:
            syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM stock_daily")]
        finally:
            con.close()
        print(f"[news] 全市场模式：stock_daily 中 {len(syms)} 只标的")
        return syms

    # 默认：取训练数据集里的标的（抓我们真正会训练的那些）
    pkl = os.path.join(args.dataset, "train_data.pkl")
    if os.path.exists(pkl):
        try:
            with open(pkl, "rb") as f:
                syms = sorted(pickle.load(f).keys())
            print(f"[news] 从 {pkl} 读取到 {len(syms)} 只标的")
            return syms
        except Exception as exc:  # noqa: BLE001
            print(f"[news] 读取数据集失败({exc})，回退到 stock_basic")

    # 回退：数据库 stock_basic
    con = sqlite3.connect(args.db)
    try:
        syms = [r[0] for r in con.execute("SELECT code FROM stock_basic")]
    finally:
        con.close()
    print(f"[news] 从 stock_basic 读取到 {len(syms)} 只标的")
    return syms


def order_symbols_hot_first(symbols, db):
    """热点事件优先：按因子表（题材/龙虎榜）出现次数降序。"""
    hot = {}
    con = sqlite3.connect(db)
    try:
        for table in ("factor_theme", "factor_dragon_tiger"):
            try:
                for sym, cnt in con.execute(
                    f"SELECT symbol, COUNT(*) c FROM {table} GROUP BY symbol"
                ):
                    hot[sym] = hot.get(sym, 0) + cnt
            except Exception:  # noqa: BLE001
                continue
    finally:
        con.close()
    n_hot = sum(1 for s in symbols if s in hot)
    print(f"[news] 热点优先：{n_hot} 只标的出现在因子表中")
    return sorted(symbols, key=lambda s: (-hot.get(s, 0), s))


def filter_existing(symbols, db):
    """断点续抓：跳过已经抓到新闻的标的。"""
    con = sqlite3.connect(db)
    try:
        have = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM news")}
    finally:
        con.close()
    left = [s for s in symbols if s not in have]
    print(f"[news] 跳过已抓取的 {len(symbols) - len(left)} 只，剩余 {len(left)} 只")
    return left


# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser(description="抓取个股新闻并入库（前期融合用）")
    ap.add_argument("--db", default=DEFAULT_DB, help="sqlite 路径")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="价格 pickle 目录（用于取标的）")
    ap.add_argument("--symbols", default=None, help="逗号分隔的标的列表，优先于自动读取")
    ap.add_argument("--limit", type=int, default=20, help="最多抓取多少只标的（0=全部）")
    ap.add_argument("--all-symbols", action="store_true",
                    help="标的池用全市场（stock_daily 去重），而非训练数据集里的标的")
    ap.add_argument("--hot-first", action="store_true",
                    help="热点事件优先：按因子表(龙虎榜/题材)出现次数降序排列")
    ap.add_argument("--skip-existing", action="store_true",
                    help="跳过已有新闻的标的（断点续抓）")
    ap.add_argument("--deadline", default=None,
                    help="截止时刻 HH:MM（本地时间），到点优雅停止")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="最长运行秒数，到点优雅停止")
    ap.add_argument("--provider", default="eastmoney", choices=sorted(PROVIDERS))
    ap.add_argument("--pages", type=int, default=3, help="每只标的抓几页")
    ap.add_argument("--page-size", type=int, default=50, help="每页条数")
    ap.add_argument("--sleep", type=float, default=0.3, help="每次请求间隔秒（限速）")
    ap.add_argument("--max-retry", type=int, default=2, help="单页失败重试次数")
    ap.add_argument("--dry-run", action="store_true", help="只抓取不写库")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"[news] DB 不存在: {args.db}")
        return 1

    symbols = load_symbols(args)
    if args.skip_existing:
        symbols = filter_existing(symbols, args.db)
    if args.hot_first:
        symbols = order_symbols_hot_first(symbols, args.db)
    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        print("[news] 没有可抓取的标的")
        return 1

    # 截止时间
    deadline = None
    if args.deadline:
        hh, mm = args.deadline.split(":")
        deadline = datetime.now().replace(hour=int(hh), minute=int(mm),
                                          second=0, microsecond=0)
        if deadline <= datetime.now():
            deadline += timedelta(days=1)
        print(f"[news] 截止时间: {deadline.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"[news] 待抓取 {len(symbols)} 只标的, provider={args.provider}, "
          f"pages={args.pages}, page_size={args.page_size}, dry_run={args.dry_run}")

    con = sqlite3.connect(args.db)
    if not args.dry_run:
        con.execute(CREATE_SQL)
        con.execute(CREATE_INDEX_SQL)
        con.commit()

    fetch_fn = PROVIDERS[args.provider]
    total_new, total_fetched, ok_syms, fail_syms = 0, 0, 0, 0
    stopped_early = False
    t0 = time.time()

    for i, sym in enumerate(symbols, 1):
        # 到点优雅停止（保证已抓数据已 commit）
        if deadline and datetime.now() >= deadline:
            print(f"[news] 到达截止时间 {args.deadline}，停止（已完成 {i - 1}/{len(symbols)}）")
            stopped_early = True
            break
        if args.max_seconds and (time.time() - t0) >= args.max_seconds:
            print(f"[news] 达到最长运行 {args.max_seconds}s，停止（已完成 {i - 1}/{len(symbols)}）")
            stopped_early = True
            break

        try:
            items = fetch_fn(sym, pages=args.pages, page_size=args.page_size,
                             sleep=args.sleep, max_retry=args.max_retry)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(symbols)}] {sym} 异常: {type(exc).__name__}: {exc}")
            fail_syms += 1
            continue

        total_fetched += len(items)
        new_cnt = 0
        if items and not args.dry_run:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows = [(sym, it["date"], it["title"], it["content"], it["source"], it["url"], now)
                    for it in items]
            cur = con.executemany(INSERT_SQL, rows)
            new_cnt = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            con.commit()
        total_new += new_cnt
        if items:
            ok_syms += 1
        else:
            fail_syms += 1

        dates = sorted({it["date"] for it in items})
        span = f"{dates[0]}~{dates[-1]}" if dates else "-"
        print(f"  [{i}/{len(symbols)}] {sym}: 抓到 {len(items)} 条, 新增 {new_cnt} 条, 日期 {span}")
        time.sleep(args.sleep)

    con.close()
    print(f"\n[news] 完成: 标的 {len(symbols)} 只（成功 {ok_syms} / 无数据 {fail_syms}）, "
          f"抓到 {total_fetched} 条, 入库新增 {total_new} 条, 耗时 {time.time() - t0:.1f}s")
    if args.dry_run:
        print("[news] dry-run 模式，未写入数据库")
    return 0


if __name__ == "__main__":
    sys.exit(main())
