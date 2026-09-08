import json

import pandas as pd

from app.data.config import DataConfig
from app.kronos import ForecastPoint, ForecastResult
from app.pipeline import run_pipeline


def prices():
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [100, 110, 120],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )


class FakeAdapter:
    def __init__(self, config=None):
        self.config = config

    def forecast(self, history, horizon):
        assert list(history.columns) == ["timestamp", "close", "high", "low", "open", "volume"]
        return ForecastResult(
            timeframe="1D",
            context_length=len(history),
            forecast_horizon=horizon,
            forecasts=tuple(
                ForecastPoint("2025-01-04T00:00:00", 13, 14, 12, 13.5, 130, 1755)
                for _ in range(horizon)
            ),
        )


def test_pipeline_persists_data_and_reports_mock_forecast_and_decision(tmp_path):
    report = run_pipeline(
        data_config=DataConfig(("ABC",), tmp_path),
        symbol="ABC",
        forecast_horizon=1,
        downloader=lambda ticker, start, end: prices(),
        adapter_factory=FakeAdapter,
        risk_inputs={
            "entry": 10.0, "stop": 9.0, "target": 12.0, "direction": "long",
            "equity": 100_000.0, "risk_budget": 1_000.0,
            "max_portfolio_risk": 2_000.0, "contract_multiplier": 1.0,
            "tick_size": 0.01, "win_probability": 0.8, "sample_count": 100,
        },
    )

    assert report["data"]["yahoo"]["status"] == "success"
    assert report["forecast"]["status"] == "success"
    assert report["confidence"] is None
    assert report["risk"]["status"] == "success"
    assert report["decision"]["status"] == "success"
    json.dumps(report)


def test_pipeline_never_fabricates_forecast_when_model_is_unavailable(tmp_path):
    report = run_pipeline(
        data_config=DataConfig(("ABC",), tmp_path),
        symbol="ABC",
        forecast_horizon=1,
        downloader=lambda ticker, start, end: prices(),
        adapter_factory=lambda config: (_ for _ in ()).throw(
            ModuleNotFoundError("missing model")
        ),
    )

    assert report["forecast"]["status"] == "failed"
    assert "result" not in report["forecast"]
    assert report["confidence"] is None
