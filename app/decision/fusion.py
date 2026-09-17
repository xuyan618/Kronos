"""决策层融合：Kronos 价格预测方向 × FinBERT 文本情感方向。

设计：
    - Kronos 方向：由 KronosForecastAdapter 的 OHLCV 预测推导
      （预测期末收盘价相对最后已知收盘价的归一化收益率，裁剪到 [-1, 1]）。
    - 情感方向：由 app.sentiment 的 FinBERT 打分得到，score ∈ [-1, 1]。
    - 最终方向 = 两者加权求和后裁剪到 [-1, 1]；
      combined > +threshold → long，< -threshold → short，否则 hold。
    - confidence = |combined|，可作为后续 RiskResult / Evidence 的输入。

该模块**不修改**现有 `make_decision`（那是风险/期望收益层），
只在更上层提供一个"产生交易方向"的融合信号，供调用方注入决策。
"""

from __future__ import annotations

import os

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.sentiment import FinbertSentimentScorer, SentimentResult


@dataclass(frozen=True)
class FusedSignal:
    combined_score: float          # 加权融合结果 ∈ [-1, 1]
    kronos_direction: float        # Kronos 推导方向 ∈ [-1, 1]
    sentiment_score: float         # FinBERT 情感分 ∈ [-1, 1]
    direction: str                 # "long" / "short" / "hold"
    confidence: float              # = |combined_score|
    n_texts: int                   # 参与打分的文本条数


def kronos_direction_from_forecast(
    forecast: Any,
    last_close: float,
    clip: float = 1.0,
) -> float:
    """由 Kronos 预测推导方向：期末预测收盘价相对最后已知收盘价的归一化收益率。"""
    try:
        closes = [float(p.close) for p in forecast.forecasts]
    except AttributeError:
        # 兼容：直接传入收盘价序列
        closes = [float(x) for x in forecast]
    if not closes:
        return 0.0
    fwd = closes[-1]
    if last_close in (0.0, None) or last_close != last_close:
        return 0.0
    ret = (fwd - last_close) / last_close
    direction = max(-clip, min(clip, float(ret)))
    # 符号校正开关（默认关闭）。
    # 回测(backtest_predictor_fusion.py)对「早期融合模型 sequoia_predictor_fusion」的预测 z 分
    # 证实方向与次日收益反向，取负后 RankIC 翻正。但本函数使用的是原始 KronosForecastAdapter
    # 的预测收益，尚未单独验证方向，故默认不翻转。若后续对原始 Kronos 回测确认同样反向，或改为
    # 路由融合模型输出，再置 KRONOS_DIRECTION_NEGATE=1。
    if os.environ.get("KRONOS_DIRECTION_NEGATE", "0") == "1":
        direction = -direction
    return direction


def fuse_signals(
    kronos_direction: float,
    sentiment_score: float,
    kronos_weight: float = 0.6,
    sentiment_weight: float = 0.4,
    threshold: float = 0.2,
    n_texts: int = 0,
) -> FusedSignal:
    """加权融合 Kronos 方向与情感分，产出最终方向与置信度。

    注意：情感在约 90% 的 (symbol,date) 上为中性（score=0）。若始终预留
    ``sentiment_weight`` 给情感，Kronos 主信号会被永久稀释。因此只在情感
    确实可用（有文本且 |score|>0）时才做加权融合；否则不稀释 Kronos。
    """
    sentiment_present = n_texts > 0 and abs(sentiment_score) > 1e-9
    if sentiment_present:
        combined = kronos_weight * kronos_direction + sentiment_weight * sentiment_score
    else:
        # 情感缺失或中性：保留 Kronos 主信号完整强度
        combined = kronos_direction
    combined = max(-1.0, min(1.0, combined))
    if combined > threshold:
        direction = "long"
    elif combined < -threshold:
        direction = "short"
    else:
        direction = "hold"
    return FusedSignal(
        combined_score=round(combined, 4),
        kronos_direction=round(float(kronos_direction), 4),
        sentiment_score=round(float(sentiment_score), 4),
        direction=direction,
        confidence=round(abs(combined), 4),
        n_texts=n_texts,
    )


def generate_trade_signal(
    symbol: str,
    date: str,
    history: Any,
    *,
    adapter: Any | None = None,
    scorer: FinbertSentimentScorer | None = None,
    db_path: str | None = None,
    horizon: int = 5,
    sources: Sequence[str] = ("factor", "news"),
    weights: Mapping[str, float] | None = None,
    kronos_direction: float | None = None,
) -> FusedSignal:
    """端到端：给定标的/日期/历史价，产出 Kronos×情感 融合交易信号。

    - adapter: KronosForecastAdapter 实例（为空则 Kronos 方向取 0）。
    - scorer: FinbertSentimentScorer 实例（为空则情感分取 0）。
    - history: 含 ["close"] 及 KronosForecastAdapter 所需列的 DataFrame。
    - kronos_direction: 已算好的 Kronos 方向（避免重复推理）；未给且 adapter
      非空时，由 adapter 重新预测推导。
    """
    # 1) Kronos 方向
    if kronos_direction is not None:
        kronos_dir = float(kronos_direction)
    else:
        kronos_dir = 0.0
        if adapter is not None:
            fc = adapter.forecast(history, horizon)
            last_close = float(history["close"].iloc[-1])
            kronos_dir = kronos_direction_from_forecast(fc, last_close)

    # 2) 情感方向
    sentiment = 0.0
    n_texts = 0
    if scorer is not None:
        # db_path 为 None 时不传，让 scorer 用其默认数据库路径
        if db_path is not None:
            res: SentimentResult = scorer.score_symbol_date(symbol, date, db_path, sources)
        else:
            res: SentimentResult = scorer.score_symbol_date(symbol, date, sources=sources)
        sentiment = res.score
        n_texts = res.n_texts

    w = dict(weights or {})
    sig = fuse_signals(kronos_dir, sentiment, n_texts=n_texts, **w)
    return sig


# 便捷别名
fuse = fuse_signals
