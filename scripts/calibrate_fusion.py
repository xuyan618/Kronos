#!/usr/bin/env python3
"""批量校准 FUSION_DIRECTION_SCALE（融合模型单标的标定刻度）。

数据来源：sequoia_v2.db 的 stock_daily 表（open/high/low/close/volume/turnover/outstanding_share）。
价格与文本走同一 DB，保证取数一致（脚本会设置 SEQUOIA_V2_DB 指向同一库）。

生产 pipeline（USE_FUSION_MODEL=1）把融合模型 predict_return 输出（≈次日收益率）标定到 [-1,1]：
    direction = clip(pred_return / FUSION_DIRECTION_SCALE, -1, 1)
再经 fuse_signals 以 |direction|>0.2 触发多/空。本脚本对一组标的算出预测收益分布，
并给出候选 scale 及对应的「满仓比例 / 触发交易比例」，方便按实盘挑刻度。

用法：
  # 逐标的：每个标的取 DB 中最新交易日各预测一次（快，得到横截面分布）
  python scripts/calibrate_fusion.py --symbols AAPL,MSFT,NVDA

  # 区间采样：每个标的在 [START,END] 内逐交易日（每 --step 个取一个）预测（分布更稳）
  python scripts/calibrate_fusion.py --symbols AAPL,MSFT,NVDA --range 2026-03-01 2026-09-10 --step 3

  # 指定某日对所有标的统一预测
  python scripts/calibrate_fusion.py --symbols AAPL,MSFT,NVDA --as-of 2026-09-10

环境变量：SEQUOIA_V2_DB（DB 路径）、SEQUOIA_FUSION_MODEL_DIR、TEXT_FINBERT_MODEL、HF_ENDPOINT。
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# 确保仓库根在 sys.path（无论从哪运行）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.kronos.fusion_adapter import last_predicted_return

FUSION_L = 90  # 上下文长度（与训练/回测一致）


def resolve_db(cli_db: str | None) -> str:
    candidates = []
    if cli_db:
        candidates.append(cli_db)
    env = os.environ.get("SEQUOIA_V2_DB")
    if env:
        candidates.append(env)
    # 优先共享库 BigAData，其次仓库内 data/sequoia_v2.db
    candidates.append("/Users/xuyan/Desktop/BigAData/sequoia_v2.db")
    candidates.append(str(ROOT / "data" / "sequoia_v2.db"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise SystemExit(
        "找不到 sequoia_v2.db。请用 --db 指定路径，或设置环境变量 SEQUOIA_V2_DB。"
    )


def load_history(db_path: str, symbol: str, start: str | None, end: str | None) -> pd.DataFrame | None:
    con = sqlite3.connect(db_path)
    sql = (
        "SELECT date, open, high, low, close, volume, turnover, outstanding_share "
        "FROM stock_daily WHERE symbol = ?"
    )
    params: list = [symbol]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date ASC"
    try:
        df = pd.read_sql(sql, con, params=params)
    finally:
        con.close()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["date"])
    # 列名对齐 last_predicted_return 期望：volume(内部转 vol) / amt / outstanding_share
    df = df.rename(columns={"turnover": "amt"})
    out = df[["timestamp", "open", "high", "low", "close",
              "volume", "amt", "outstanding_share"]].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="校准 FUSION_DIRECTION_SCALE（数据来自 sequoia_v2.db）")
    ap.add_argument("--symbols", required=True, help="逗号分隔标的，如 AAPL,MSFT,NVDA")
    ap.add_argument("--db", help="sequoia_v2.db 路径（默认按 SEQUOIA_V2_DB / data/sequoia_v2.db 查找）")
    ap.add_argument("--as-of", help="统一预测日 YYYY-MM-DD（对所有标的用同一天）")
    ap.add_argument("--range", nargs=2, metavar=("START", "END"),
                    help="区间采样模式：YYYY-MM-DD YYYY-MM-DD，逐交易日预测")
    ap.add_argument("--step", type=int, default=1, help="区间模式下的采样步长（每 N 个交易日取一个）")
    ap.add_argument("--min-history", type=int, default=91,
                    help="最少历史行数（默认 91 = L90 + 1），不足则跳过")
    args = ap.parse_args()

    db = resolve_db(args.db)
    # 统一文本查找走同一 DB
    os.environ.setdefault("SEQUOIA_V2_DB", db)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    rows = []
    for sym in symbols:
        # 始终加载完整历史：last_predicted_return 需要 as_of 前 FUSION_L 行作上下文，
        # 不过滤日期才能给区间/指定日模式提供足够窗口。
        hist = load_history(db, sym, start=None, end=None)
        if hist is None:
            print(f"[skip] {sym}: DB 中无数据")
            continue
        if len(hist) < args.min_history:
            print(f"[skip] {sym}: 历史仅 {len(hist)} 行 (<{args.min_history})")
            continue

        if args.as_of:
            d = pd.Timestamp(args.as_of)
            if d not in hist["timestamp"].values:
                print(f"[skip] {sym}: as-of {args.as_of} 不在该标的历史中")
                continue
            dates = [d]
        elif args.range:
            s0, s1 = pd.Timestamp(args.range[0]), pd.Timestamp(args.range[1])
            cand = [i for i in range(FUSION_L, len(hist))
                    if s0 <= hist["timestamp"].iloc[i] <= s1]
            dates = [hist["timestamp"].iloc[cand[j]] for j in range(0, len(cand), max(1, args.step))]
        else:
            # 默认：最新交易日
            dates = [hist["timestamp"].iloc[-1]]

        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            try:
                direction, pred_ret = last_predicted_return(sym, hist, d_str)
            except Exception as e:  # 单点失败不应中断整批
                print(f"[skip] {sym} @ {d_str}: {e}")
                continue
            rows.append((sym, d_str, pred_ret, direction))

    if not rows:
        raise SystemExit("无有效预测")

    df = pd.DataFrame(rows, columns=["symbol", "as_of", "pred_return", "direction"])
    pr = df["pred_return"]

    print(f"\n=== 数据源: {db} ===")
    print(f"=== 样本: {len(pr)} 条（{df['symbol'].nunique()} 标的）===")

    if args.range or len(df) > 60:
        # 区间/大样本：只打印分布，不逐行
        print("\n=== 预测收益分布（pred_return = 次日预测收益率，含区间采样）===")
    else:
        print("\n=== 逐标的预测（pred_return = 次日预测收益率）===")
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(df.to_string(index=False))
        print("\n=== 预测收益分布 ===")

    print(f"样本数       : {len(pr)}")
    print(f"均值         : {pr.mean():+.4f}")
    print(f"标准差       : {pr.std():.4f}")
    print(f"最小 / 最大  : {pr.min():+.4f} / {pr.max():+.4f}")
    print(f"正收益占比   : {(pr > 0).mean() * 100:.1f}%")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"  P{int(q * 100):02d}       : {pr.quantile(q):+.4f}")

    print("\n=== 候选 FUSION_DIRECTION_SCALE ===")
    print("direction = clip(pred_return / scale, -1, 1)；|direction|>0.2 触发多/空。")
    std = pr.std()
    if std > 0:
        for k in (1.0, 1.5, 2.0):
            s = k * std
            full = (pr.abs() >= s).mean() * 100          # 被截断到 ±1（满仓）的比例
            trigger = (pr.abs() > 0.2 * s).mean() * 100   # 会触发交易的比例
            print(f"  scale = {s:.4f}  (±{k:.1f}σ): 满仓比例 {full:5.1f}% | 触发交易 {trigger:5.1f}%")
    p95 = pr.abs().quantile(0.95)
    if p95 > 0:
        s = p95
        full = (pr.abs() >= s).mean() * 100
        trigger = (pr.abs() > 0.2 * s).mean() * 100
        print(f"  scale = {s:.4f}  (P95|pred|): 满仓比例 {full:5.1f}% | 触发交易 {trigger:5.1f}%  (更保守)")

    print("\n使用方法：设置环境变量 FUSION_DIRECTION_SCALE=<scale> 后跑 pipeline（USE_FUSION_MODEL=1）。")


if __name__ == "__main__":
    main()
