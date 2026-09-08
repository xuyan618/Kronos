from unittest.mock import Mock

import pandas as pd
import pytest

from app.kronos import ForecastConfig, KronosForecastAdapter


def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="D"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100.0, 110.0, 120.0],
        }
    )


def adapter(predictor: Mock) -> KronosForecastAdapter:
    return KronosForecastAdapter(
        model=object(),
        tokenizer=object(),
        config=ForecastConfig(timeframe="1D"),
        predictor_factory=lambda *args, **kwargs: predictor,
    )


def test_forecast_has_stable_schema_and_unavailable_uncertainty_fields():
    predictor = Mock()
    predictor.predict.return_value = pd.DataFrame(
        {
            "open": [13.0, 14.0],
            "high": [14.0, 15.0],
            "low": [12.0, 13.0],
            "close": [13.5, 14.5],
            "volume": [130.0, 140.0],
            "amount": [1755.0, 2030.0],
        }
    )

    result = adapter(predictor).forecast(history(), 2)

    assert result.to_dict() == {
        "timeframe": "1D",
        "context_length": 3,
        "forecast_horizon": 2,
        "forecasts": [
            {"timestamp": "2025-01-04T00:00:00", "open": 13.0, "high": 14.0, "low": 12.0, "close": 13.5, "volume": 130.0, "amount": 1755.0},
            {"timestamp": "2025-01-05T00:00:00", "open": 14.0, "high": 15.0, "low": 13.0, "close": 14.5, "volume": 140.0, "amount": 2030.0},
        ],
        "confidence": None,
        "forecast_range": None,
        "unavailable": ["confidence", "forecast_range"],
    }
    predictor.predict.assert_called_once()


@pytest.mark.parametrize("column", ["timestamp", "open", "high", "low", "close", "volume"])
def test_missing_required_history_column_is_rejected(column):
    frame = history().drop(columns=[column])
    with pytest.raises(ValueError, match="missing required columns"):
        adapter(Mock()).forecast(frame, 1)


def test_invalid_market_values_are_rejected():
    frame = history()
    frame.loc[0, "high"] = 1.0
    with pytest.raises(ValueError, match="high"):
        adapter(Mock()).forecast(frame, 1)
