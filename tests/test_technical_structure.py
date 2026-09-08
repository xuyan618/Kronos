import numpy as np
import pandas as pd

from app.technical import TechnicalConfig, calculate_multi_timeframe, calculate_technical


def candles(count=80):
    index = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
    close = pd.Series(np.arange(count, dtype=float) + 100, index=index)
    return pd.DataFrame({
        "Open": close - 0.5, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": np.full(count, 100.0),
    }, index=index)


def test_atr_and_trend_are_deterministic_on_synthetic_data():
    result = calculate_technical(candles(), config=TechnicalConfig(atr_window=14))
    assert result.status == "OK"
    assert result.signals["trend"] == "UP"
    assert result.indicators["atr"] == 2.0
    assert result.indicators["fast_ma"] < result.indicators["close"]


def test_levels_and_volume_are_historical_and_explicit():
    frame = candles()
    frame.iloc[-1, frame.columns.get_loc("Close")] = 190
    frame.iloc[-1, frame.columns.get_loc("High")] = 191
    frame.iloc[-1, frame.columns.get_loc("Volume")] = 300
    result = calculate_technical(frame, config=TechnicalConfig(level_window=20))
    assert result.levels["resistance"] < 190
    assert result.signals["breakout"] == "UP"
    assert result.indicators["relative_volume"] > 1


def test_as_of_excludes_later_candles_and_changes_report_timestamp():
    frame = candles()
    early = calculate_technical(frame, as_of="2024-02-15T00:00:00Z")
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("Close")] = 10000
    same_early = calculate_technical(changed, as_of="2024-02-15T00:00:00Z")
    assert early.to_dict() == same_early.to_dict()
    assert early.latest_timestamp == "2024-02-15T00:00:00Z"


def test_insufficient_history_is_unknown_not_zero():
    result = calculate_technical(candles(5), config=TechnicalConfig(min_history=10))
    assert result.status == "UNKNOWN"
    assert result.indicators == {}
    assert result.reasons


def test_multi_timeframe_reports_each_timeframe():
    results = calculate_multi_timeframe({"1d": candles(), "1w": candles(10)})
    assert results["1d"].timeframe == "1d"
    assert results["1w"].status == "UNKNOWN"
