"""中文金融情感分析（FinBERT），供决策层融合使用。"""

from app.sentiment.finbert_sentiment import (
    DEFAULT_MODEL,
    FinbertSentimentScorer,
    SentimentResult,
    get_scorer,
)

__all__ = [
    "DEFAULT_MODEL",
    "FinbertSentimentScorer",
    "SentimentResult",
    "get_scorer",
]
