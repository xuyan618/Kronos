"""
sequoia_db_loader.py —— 把 data/sequoia_v2.db 的 stock_daily 表
转换成 finetune/dataset.py (QlibDataset) 期望的 pickle 格式。

针对「在原有权重上引入新维度」的目标，本加载器在原始 6 维
(open/high/low/close/vol/amt) 基础上额外保留 outstanding_share（流通股本，
日频、全量覆盖），因此 feature_list = 7 维。

列映射（db -> Kronos feature_list）:
    open/high/low/close -> 原样
    volume              -> vol
    turnover            -> amt  (成交额)
    outstanding_share   -> outstanding_share (新增维度)

运行:  python finetune/sequoia_db_loader.py
"""

import os
import sqlite3
import pickle
import pandas as pd

if os.environ.get("SEQUOIA_FUSION") == "1":
    from config_sequoia_fusion import get_config as _get_cfg
elif os.environ.get("SEQUOIA_CPU") == "1":
    from config_sequoia_cpu import get_config as _get_cfg
elif os.environ.get("SEQUOIA") == "1":
    from config_sequoia import get_config as _get_cfg
else:
    from config import Config
    _get_cfg = Config
_cfg = _get_cfg()

# 默认指向 BigAData 的 v2 库；可用环境变量 SEQUOIA_DB 覆盖（如工作区内 data/sequoia_v2.db）。
DB_PATH = os.environ.get(
    "SEQUOIA_DB",
    "/Users/xuyan/Desktop/BigAData/sequoia_v2.db",
)

# Sequoia 数据约 2024-01 .. 2026-09，做非重叠切分
# 可用环境变量覆盖（前期融合需要把区间指到「有文本」的时段；不设置时行为与原来一致）
TRAIN_RANGE = (
    os.environ.get("TRAIN_RANGE_START", "2024-01-01"),
    os.environ.get("TRAIN_RANGE_END", "2025-06-30"),
)
VAL_RANGE = (
    os.environ.get("VAL_RANGE_START", "2025-07-01"),
    os.environ.get("VAL_RANGE_END", "2026-02-28"),
)
TEST_RANGE = (
    os.environ.get("TEST_RANGE_START", "2026-03-01"),
    os.environ.get("TEST_RANGE_END", "2026-09-11"),
)

# 最小序列长度：默认 = lookback + predict + 1（保证单段能切出足够窗口）
BASE_MIN = _cfg.lookback_window + _cfg.predict_window + 1
# 全局可覆盖；并按 train/val/test 分别设下限。严格时间切分下，验证/测试段
# 窗口通常较短，默认放开（0 = 不丢样本，全部保留用于评测）。
MIN_LEN = int(os.environ.get("MIN_LEN_OVERRIDE", BASE_MIN))
TRAIN_MIN_LEN = int(os.environ.get("TRAIN_MIN_LEN", MIN_LEN))
VAL_MIN_LEN = int(os.environ.get("VAL_MIN_LEN", 0))
TEST_MIN_LEN = int(os.environ.get("TEST_MIN_LEN", 0))


def load_raw(db_path):
    con = sqlite3.connect(db_path)
    sql = (
        "SELECT symbol, date, open, high, low, close, volume, turnover, outstanding_share "
        "FROM stock_daily"
    )
    df = pd.read_sql(sql, con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"volume": "vol", "turnover": "amt"})
    return df[
        ["symbol", "date", "open", "high", "low", "close", "vol", "amt", "outstanding_share"]
    ]


def build_per_symbol(df):
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        g.index.name = "datetime"
        g = g[["open", "high", "low", "close", "vol", "amt", "outstanding_share"]].dropna()
        if len(g) >= TRAIN_MIN_LEN:
            out[sym] = g
    # CPU 冒烟测试：仅取前 max_symbols 只标的
    cap = getattr(_cfg, "max_symbols", None)
    if cap:
        out = dict(list(out.items())[: int(cap)])
    return out


def split(data):
    tr0, tr1 = pd.to_datetime(TRAIN_RANGE[0]), pd.to_datetime(TRAIN_RANGE[1])
    vr0, vr1 = pd.to_datetime(VAL_RANGE[0]), pd.to_datetime(VAL_RANGE[1])
    te0, te1 = pd.to_datetime(TEST_RANGE[0]), pd.to_datetime(TEST_RANGE[1])
    train, val, test = {}, {}, {}
    for sym, g in data.items():
        tr = g[(g.index >= tr0) & (g.index <= tr1)]
        vr = g[(g.index >= vr0) & (g.index <= vr1)]
        te = g[(g.index >= te0) & (g.index <= te1)]
        if len(tr) >= TRAIN_MIN_LEN:
            train[sym] = tr
        if len(vr) >= VAL_MIN_LEN:
            val[sym] = vr
        if len(te) >= TEST_MIN_LEN:
            test[sym] = te
    return train, val, test


def main():
    assert os.path.exists(DB_PATH), f"DB not found: {DB_PATH}"
    print(f"[loader] reading {DB_PATH}")
    raw = load_raw(DB_PATH)
    print(f"[loader] raw rows={len(raw)} symbols={raw['symbol'].nunique()}")

    data = build_per_symbol(raw)
    print(f"[loader] usable symbols (>= {TRAIN_MIN_LEN} rows) = {len(data)}")

    # 可选白名单：仅保留「有文本」的标的（融合训练聚焦文本覆盖集）
    wl_path = os.environ.get("SEQUOIA_SYMBOL_WHITELIST")
    if wl_path and os.path.exists(wl_path):
        import json
        with open(wl_path) as _f:
            wl = set(json.load(_f))
        data = {s: g for s, g in data.items() if s in wl}
        print(f"[loader] whitelist applied: {len(data)} symbols retained")

    train, val, test = split(data)
    print(f"[loader] train={len(train)} val={len(val)} test={len(test)}")

    os.makedirs(_cfg.dataset_path, exist_ok=True)
    with open(os.path.join(_cfg.dataset_path, "train_data.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(_cfg.dataset_path, "val_data.pkl"), "wb") as f:
        pickle.dump(val, f)
    with open(os.path.join(_cfg.dataset_path, "test_data.pkl"), "wb") as f:
        pickle.dump(test, f)
    print(f"[loader] saved pickles to {_cfg.dataset_path}")


if __name__ == "__main__":
    main()
