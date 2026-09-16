"""中文金融情感打分（FinBERT）— 供决策层融合使用。

模型：`yiyanghkust/finbert-tone-chinese`
    - 基于 bert-base-chinese，在中文金融语料上微调的 3 分类（积极/中性/消极）。
    - 体积约 400MB（fp32）/ ~110MB（int8），纯 CPU 可跑，经 hf-mirror 可下。

输出：每条文本的 `score ∈ [-1, 1]`（消极=-1，中性=0，积极=+1），
由模型概率直接得到 `score = P(积极) - P(消极)`，连续且可聚合。

设计要点：
    - 模型权重**懒加载**，import 本模块不触发 transformers / torch 下载。
    - 自动走 `HF_ENDPOINT`（默认 hf-mirror.com），与训练环境一致。
    - 文本按 (symbol, date) 从 `sequoia_v2.db` 读取（复用与训练一致的表结构），
      并按 "未来泄漏过滤"（announce_date 晚于交易日则丢弃）。
    - 支持内存缓存 + 可选 pickle 持久化，避免对 8.8 万条 (symbol,date) 重复推理。
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ----------------------------------------------------------------------------
# 默认配置
# ----------------------------------------------------------------------------
DEFAULT_MODEL = "yiyanghkust/finbert-tone-chinese"
DEFAULT_MAX_LENGTH = 128
HF_MIRROR = "https://hf-mirror.com"

# 与 finetune/text_source.py 保持一致的表/列定义
DB_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "sequoia_v2.db",
)
FACTOR_TEXT_SOURCES = [
    ("factor_dragon_tiger", ["reason"]),
    ("factor_theme", ["reason", "tags"]),
]
NEWS_TABLE = "news"
NEWS_TEXT_COLS = ["title", "content"]


@dataclass(frozen=True)
class SentimentResult:
    """单 (symbol, date) 的聚合情感结果。"""

    symbol: str
    date: str
    score: float = 0.0          # 聚合情感分 ∈ [-1, 1]
    label: str = "neutral"      # 主导标签
    n_texts: int = 0            # 参与打分的文本条数
    probs: tuple[float, ...] = ()  # 各类别平均概率（顺序同模型 id2label）
    texts: tuple[str, ...] = field(default=())


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _cols_of(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _read_source(
    con: sqlite3.Connection,
    table: str,
    text_cols: Sequence[str],
    date_col: str = "date",
    symbol: str | None = None,
    date: str | None = None,
) -> "list[tuple[str, str]]":
    """读取一张表，返回 [(text, date_str), ...]（已按 symbol/date 过滤 + 未来泄漏过滤）。

    symbol/date 为空时返回整表（供批量构建使用）。
    """
    if not _table_exists(con, table):
        return []
    cols = _cols_of(con, table)
    if "symbol" not in cols or date_col not in cols:
        return []
    use_cols = ["symbol", date_col] + [c for c in text_cols if c in cols]
    if len(use_cols) <= 2:
        return []

    conds: list[str] = []
    params: list[str] = []
    if symbol is not None:
        conds.append("symbol = ?")
        params.append(str(symbol))
    if date is not None:
        conds.append(f"date({date_col}) = date(?)")
        params.append(str(date))
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    has_announce = "announce_date" in cols
    sel = ", ".join(use_cols + (["announce_date"] if has_announce else []))
    df = None
    try:
        df = __import__("pandas").read_sql(
            f"SELECT {sel} FROM {table} {where}", con, params=params
        )
    except Exception:
        return []
    if df is None or df.empty:
        return []

    df[date_col] = __import__("pandas").to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "symbol"])

    # 合并文本列
    parts = [df[c].fillna("").astype(str) for c in text_cols if c in df.columns]
    if not parts:
        return []
    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p
    df["text"] = text.str.strip()
    df = df[df["text"].str.len() > 0]

    if has_announce:
        ann = __import__("pandas").to_datetime(df["announce_date"], errors="coerce")
        keep = ann.isna() | (ann <= df[date_col])
        df = df[keep]

    out = []
    for _, row in df.iterrows():
        out.append((str(row["text"]), row[date_col].strftime("%Y-%m-%d")))
    return out


class FinbertSentimentScorer:
    """中文金融 FinBERT 情感打分器（按 (symbol, date) 缓存）。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_length: int = DEFAULT_MAX_LENGTH,
        cache_path: str | None = None,
        use_int8: bool = False,
        hf_mirror: str = HF_MIRROR,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.cache_path = cache_path
        self.use_int8 = use_int8
        self.hf_mirror = hf_mirror

        self._model = None
        self._tokenizer = None
        self._pos_idx = 2
        self._neu_idx = 1
        self._neg_idx = 0
        self._labels: tuple[str, ...] = ("negative", "neutral", "positive")

        # (symbol, date) -> SentimentResult
        self._cache: dict[tuple[str, str], SentimentResult] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                print(f"[sentiment] 已加载缓存: {cache_path} ({len(self._cache)} 条)")
            except Exception as e:
                print(f"[sentiment] 缓存读取失败({e})，从头开始")

    # ------------------------------------------------------------------ 模型
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("HF_ENDPOINT", self.hf_mirror)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        print(f"[sentiment] 加载 FinBERT: {self.model_name} (device={self.device})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        model.to(self.device)

        if self.use_int8 and self.device == "cpu":
            print("[sentiment] 应用 int8 动态量化(CPU 省内存)...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        # 解析类别映射（兼容中文/英文标签）
        id2label = getattr(model.config, "id2label", None)
        if id2label:
            items = [id2label[i] for i in sorted(int(k) for k in id2label.keys())]
            self._labels = tuple(str(x) for x in items)
        self._pos_idx, self._neu_idx, self._neg_idx = self._resolve_indices()

        self._model = model
        print(f"[sentiment] 类别顺序: {self._labels} "
              f"(pos={self._pos_idx}, neu={self._neu_idx}, neg={self._neg_idx})")

    def _resolve_indices(self):
        pos = neu = neg = None
        for i, lab in enumerate(self._labels):
            low = str(lab).lower()
            if any(k in low for k in ("pos", "积极", "看涨", "看多", "bull")):
                pos = i
            elif any(k in low for k in ("neg", "消极", "看跌", "看空", "bear")):
                neg = i
            elif any(k in low for k in ("neu", "中性", "中")):
                neu = i
        # 兜底：按常见顺序 [neg, neu, pos]
        return (pos if pos is not None else 2,
                neu if neu is not None else 1,
                neg if neg is not None else 0)

    # --------------------------------------------------------------- 文本打分
    def score_texts(self, texts: Sequence[str]):
        """对一批文本打分，返回 list[(score, label, probs)]。无需模型时抛错。"""
        self._ensure_model()
        import torch

        clean = [str(t).strip() for t in texts if t and str(t).strip()]
        if not clean:
            return [(0.0, "neutral", tuple(1 / 3 for _ in self._labels))]
        enc = self._tokenizer(
            clean, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().float().numpy()

        results = []
        for p in probs:
            score = float(p[self._pos_idx]) - float(p[self._neg_idx])
            idx = int(p.argmax())
            results.append((score, self._labels[idx], tuple(float(x) for x in p)))
        return results

    # --------------------------------------------------------- 单条 (symbol,date)
    def read_texts(
        self,
        symbol: str,
        date: str,
        db_path: str = DB_PATH_DEFAULT,
        sources: Sequence[str] = ("factor", "news"),
    ) -> list[str]:
        """仅读取 (symbol, date) 的文本（不加载模型），方便调试/测试。"""
        assert os.path.exists(db_path), f"DB not found: {db_path}"
        con = sqlite3.connect(db_path)
        rows: list[str] = []
        try:
            if "factor" in sources:
                for table, cols in FACTOR_TEXT_SOURCES:
                    for txt, _ in _read_source(con, table, cols, symbol=symbol, date=date):
                        if txt:
                            rows.append(txt)
            if "news" in sources:
                for txt, _ in _read_source(con, NEWS_TABLE, NEWS_TEXT_COLS, symbol=symbol, date=date):
                    if txt:
                        rows.append(txt)
        finally:
            con.close()
        return rows

    def score_symbol_date(
        self,
        symbol: str,
        date: str,
        db_path: str = DB_PATH_DEFAULT,
        sources: Sequence[str] = ("factor", "news"),
        use_cache: bool = True,
    ) -> SentimentResult:
        key = (symbol, str(date))
        if use_cache and key in self._cache:
            return self._cache[key]

        texts = self.read_texts(symbol, date, db_path, sources)
        if not texts:
            res = SentimentResult(symbol=symbol, date=str(date), n_texts=0)
            self._cache[key] = res
            return res

        scored = self.score_texts(texts)
        scores = [s for s, _, _ in scored]
        n = len(scores)
        avg_score = sum(scores) / n if n else 0.0
        # 主导标签 = 平均概率最大类
        avg_probs = [sum(p[i] for _, _, p in scored) / n for i in range(len(self._labels))]
        dom = int(max(range(len(avg_probs)), key=lambda i: avg_probs[i]))
        res = SentimentResult(
            symbol=symbol, date=str(date),
            score=round(avg_score, 4),
            label=self._labels[dom],
            n_texts=n,
            probs=tuple(round(x, 4) for x in avg_probs),
            texts=tuple(texts),
        )
        self._cache[key] = res
        return res

    # ----------------------------------------------------- 批量 (symbol,date) 列表
    def score_many(
        self,
        pairs: Iterable[tuple[str, str]],
        db_path: str = DB_PATH_DEFAULT,
        sources: Sequence[str] = ("factor", "news"),
        persist: bool = True,
    ) -> list[SentimentResult]:
        """批量打分；persist=True 时每批落盘缓存。"""
        results: list[SentimentResult] = []
        for sym, d in pairs:
            results.append(self.score_symbol_date(sym, d, db_path, sources))
            if persist and len(results) % 500 == 0:
                self.save_cache()
        if persist:
            self.save_cache()
        return results

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)
        print(f"[sentiment] 缓存已保存: {self.cache_path} ({len(self._cache)} 条)")


def get_scorer(model_name: str = DEFAULT_MODEL, **kwargs) -> FinbertSentimentScorer:
    """便捷构造器。"""
    return FinbertSentimentScorer(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # 演示：对数据库里第一条有文本的样本打分（首次会下载模型，约 400MB）。
    import sys

    db = DB_PATH_DEFAULT
    scorer = FinbertSentimentScorer(cache_path="app/sentiment/sentiment_cache.pkl")
    # 取一个样例 (symbol, date)
    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT symbol, date FROM factor_dragon_tiger WHERE reason IS NOT NULL LIMIT 1"
    ).fetchone()
    con.close()
    if not row:
        print("DB 中无可用文本，退出。")
        sys.exit(0)
    sym, d = row[0], str(row[1])
    print(f"样例: symbol={sym} date={d}")
    print("文本:", scorer.read_texts(sym, d)[:3])
    res = scorer.score_symbol_date(sym, d)
    print("情感结果:", res)
