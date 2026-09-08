"""Causal, configurable technical-structure analysis for OHLCV data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import Mapping

import pandas as pd


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_ALIASES = {
    "open": "open", "Open": "open",
    "high": "high", "High": "high",
    "low": "low", "Low": "low",
    "close": "close", "Close": "close",
    "volume": "volume", "Volume": "volume",
}


@dataclass(frozen=True)
class TechnicalConfig:
    """Parameters for technical calculations; all windows are in candles."""

    fast_ma: int = 20
    slow_ma: int = 50
    atr_window: int = 14
    volatility_window: int = 20
    momentum_window: int = 10
    swing_window: int = 3
    level_window: int = 50
    volume_window: int = 20
    breakout_buffer_atr: float = 0.1
    min_history: int = 20
    levels: int = 3

    def __post_init__(self) -> None:
        for name in (
            "fast_ma", "slow_ma", "atr_window", "volatility_window",
            "momentum_window", "swing_window", "level_window",
            "volume_window", "min_history", "levels",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.fast_ma >= self.slow_ma:
            raise ValueError("fast_ma must be less than slow_ma")
        if not isfinite(self.breakout_buffer_atr) or self.breakout_buffer_atr < 0:
            raise ValueError("breakout_buffer_atr must be finite and non-negative")


@dataclass(frozen=True)
class TechnicalReport:
    """Serializable specialist output with evidence and limitations."""

    status: str
    as_of: str | None
    timeframe: str
    observations: int
    latest_timestamp: str | None
    indicators: Mapping[str, object]
    levels: Mapping[str, object]
    signals: Mapping[str, object]
    evidence: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


def _timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _prepare(frame: pd.DataFrame, as_of: object | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("OHLCV input must be a pandas DataFrame")
    renamed = frame.rename(columns={key: value for key, value in _ALIASES.items()})
    missing = [column for column in OHLCV_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"OHLCV input is missing columns: {', '.join(missing)}")
    result = renamed.loc[:, list(OHLCV_COLUMNS)].copy()
    timestamps = pd.to_datetime(frame.index, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("OHLCV index must contain valid timestamps")
    result.index = timestamps
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if as_of is not None:
        result = result.loc[result.index <= _timestamp(as_of)]
    for column in OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC prices must be numeric and non-null")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be at least open, close, and low")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be at most open, close, and high")
    return result


def _unknown(as_of: str | None, timeframe: str, count: int, reason: str) -> TechnicalReport:
    return TechnicalReport(
        status="UNKNOWN", as_of=as_of, timeframe=timeframe, observations=count,
        latest_timestamp=None, indicators={}, levels={}, signals={},
        reasons=(reason,), limitations=("Insufficient historical candles.",),
    )


def _value(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def calculate_technical(
    frame: pd.DataFrame,
    *,
    as_of: object | None = None,
    timeframe: str = "unspecified",
    config: TechnicalConfig | None = None,
) -> TechnicalReport:
    """Calculate one timeframe using candles available no later than ``as_of``.

    Rolling windows are right-aligned and swing points compare only against
    earlier candles, so neither indicator nor level can inspect a future bar.
    """
    config = config or TechnicalConfig()
    visible = _prepare(frame, as_of)
    as_of_text = None if as_of is None else _timestamp(as_of).isoformat().replace("+00:00", "Z")
    if len(visible) < max(config.min_history, config.slow_ma, config.atr_window + 1):
        report = _unknown(as_of_text, timeframe, len(visible), "not enough candles for configured windows")
        if len(visible):
            report = replace(
                report,
                latest_timestamp=visible.index[-1].isoformat().replace("+00:00", "Z"),
            )
        return report

    close, high, low, volume = visible["close"], visible["high"], visible["low"], visible["volume"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(config.atr_window, min_periods=config.atr_window).mean()
    fast = close.rolling(config.fast_ma, min_periods=config.fast_ma).mean()
    slow = close.rolling(config.slow_ma, min_periods=config.slow_ma).mean()
    returns = close.pct_change()
    volatility = returns.rolling(config.volatility_window, min_periods=config.volatility_window).std()
    momentum = close / close.shift(config.momentum_window) - 1
    avg_volume = volume.rolling(config.volume_window, min_periods=config.volume_window).mean()
    relative_volume = volume / avg_volume.replace(0, pd.NA)
    latest = visible.iloc[-1]
    last_atr = _value(atr.iloc[-1])
    last_fast, last_slow = _value(fast.iloc[-1]), _value(slow.iloc[-1])
    last_momentum = _value(momentum.iloc[-1])
    last_rel_volume = _value(relative_volume.iloc[-1])

    if last_fast is None or last_slow is None:
        trend, trend_reason = "UNKNOWN", "moving-average windows are not fully populated"
    elif latest.close > last_fast > last_slow:
        trend, trend_reason = "UP", "close is above rising-order fast and slow moving averages"
    elif latest.close < last_fast < last_slow:
        trend, trend_reason = "DOWN", "close is below falling-order fast and slow moving averages"
    else:
        trend, trend_reason = "RANGE", "moving averages do not confirm directional alignment"

    # These are historical extrema only. A swing is confirmed by comparison
    # with candles already closed before it, never by a centered/future window.
    prior_high = high.shift(1).rolling(config.swing_window, min_periods=config.swing_window).max()
    prior_low = low.shift(1).rolling(config.swing_window, min_periods=config.swing_window).min()
    swing_highs = visible.index[high >= prior_high].tolist()
    swing_lows = visible.index[low <= prior_low].tolist()
    resistance = high.shift(1).rolling(config.level_window, min_periods=1).max().iloc[-1]
    support = low.shift(1).rolling(config.level_window, min_periods=1).min().iloc[-1]
    resistance = _value(resistance)
    support = _value(support)
    buffer = (last_atr or 0.0) * config.breakout_buffer_atr
    breakout = "UNKNOWN"
    if resistance is not None and latest.close > resistance + buffer:
        breakout = "UP"
    elif support is not None and latest.close < support - buffer:
        breakout = "DOWN"
    elif support is not None and resistance is not None:
        breakout = "NONE"
    pullback = "UNKNOWN"
    if trend == "UP" and last_fast is not None and latest.low <= last_fast <= latest.close:
        pullback = "UP"
    elif trend == "DOWN" and last_fast is not None and latest.high >= last_fast >= latest.close:
        pullback = "DOWN"
    elif trend != "UNKNOWN":
        pullback = "NONE"

    structure = "UNKNOWN"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        higher_high = high.loc[swing_highs[-1]] > high.loc[swing_highs[-2]]
        higher_low = low.loc[swing_lows[-1]] > low.loc[swing_lows[-2]]
        lower_high = high.loc[swing_highs[-1]] < high.loc[swing_highs[-2]]
        lower_low = low.loc[swing_lows[-1]] < low.loc[swing_lows[-2]]
        structure = "BULLISH" if higher_high and higher_low else "BEARISH" if lower_high and lower_low else "MIXED"

    indicators = {
        "close": _value(latest.close), "atr": last_atr,
        "atr_pct": None if last_atr is None else last_atr / float(latest.close),
        "volatility": _value(volatility.iloc[-1]),
        "fast_ma": last_fast, "slow_ma": last_slow, "momentum": last_momentum,
        "volume": _value(latest.volume), "average_volume": _value(avg_volume.iloc[-1]),
        "relative_volume": last_rel_volume,
    }
    levels = {
        "support": support, "resistance": resistance,
        "swing_highs": [stamp.isoformat().replace("+00:00", "Z") for stamp in swing_highs[-config.levels:]],
        "swing_lows": [stamp.isoformat().replace("+00:00", "Z") for stamp in swing_lows[-config.levels:]],
    }
    signals = {
        "trend": trend, "market_structure": structure, "breakout": breakout,
        "pullback": pullback,
        "volume_state": "UNKNOWN" if last_rel_volume is None else (
            "ABOVE_AVERAGE" if last_rel_volume > 1 else "BELOW_AVERAGE"
        ),
    }
    reasons = (trend_reason, f"{len(swing_highs)} historical swing highs and {len(swing_lows)} swing lows")
    limitations = ("Levels are rolling historical extrema, not predictive price targets.",)
    return TechnicalReport(
        status="OK", as_of=as_of_text, timeframe=timeframe, observations=len(visible),
        latest_timestamp=visible.index[-1].isoformat().replace("+00:00", "Z"),
        indicators=indicators, levels=levels, signals=signals,
        evidence=(
            f"latest candle {visible.index[-1].isoformat().replace('+00:00', 'Z')}",
            f"ATR({config.atr_window}), MA({config.fast_ma}/{config.slow_ma}), "
            f"volume average({config.volume_window})",
        ),
        reasons=reasons, limitations=limitations,
    )


def calculate_multi_timeframe(
    frames: Mapping[str, pd.DataFrame],
    *,
    as_of: object | None = None,
    config: TechnicalConfig | None = None,
) -> dict[str, TechnicalReport]:
    """Run the same causal configuration independently for each timeframe."""
    return {
        timeframe: calculate_technical(frame, as_of=as_of, timeframe=timeframe, config=config)
        for timeframe, frame in frames.items()
    }


analyze_technical = calculate_technical
analyze_multi_timeframe = calculate_multi_timeframe
