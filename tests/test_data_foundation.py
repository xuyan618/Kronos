from datetime import date

import pandas as pd
import pytest

from app.data.config import DataConfig
from app.data.engine import collect_market_data
from app.data.providers import ProviderStatus
from app.data.schema import (
    OBSERVATION_COLUMNS,
    Observation,
    ObservationValidationError,
    filter_as_of,
    normalize_observations,
)
from app.data.yahoo import collect_yahoo


def row(**overrides):
    result = dict(
        series_id="fred:TEST",
        observation_date=date(2024, 1, 2),
        available_at="2024-01-03T12:00:00Z",
        revision_id="r1",
        value=1.5,
        source="fred",
        unit="percent",
        frequency="daily",
        asset_name="Test",
        category="macro",
        field="value",
        availability_method="release",
    )
    result.update(overrides)
    return result


def test_normalization_has_contract_columns_and_utc_timestamps():
    frame = normalize_observations([row(available_at="2024-01-03 12:00:00")])
    assert list(frame.columns) == list(OBSERVATION_COLUMNS)
    assert frame.loc[0, "available_at"] == "2024-01-03T12:00:00Z"
    assert frame.loc[0, "observation_date"] == date(2024, 1, 2)


def test_validation_rejects_missing_metadata_and_future_date():
    with pytest.raises(ObservationValidationError):
        Observation(**row(source=""))
    with pytest.raises(ObservationValidationError):
        Observation(**row(observation_date=date.today().replace(year=date.today().year + 1)))


def test_as_of_returns_latest_visible_release():
    frame = normalize_observations([
        row(revision_id="r1", value=1.0, available_at="2024-01-03T12:00:00Z"),
        row(revision_id="r2", value=2.0, available_at="2024-01-05T12:00:00Z"),
    ])
    visible = filter_as_of(frame, "2024-01-04T00:00:00Z")
    assert len(visible) == 1
    assert visible.iloc[0]["value"] == 1.0


def test_yahoo_normalization_and_engine_persists(tmp_path):
    prices = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [100]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    frame = collect_yahoo(("ABC",), available_at=pd.Timestamp("2024-01-03", tz="UTC"),
                          downloader=lambda ticker, start, end: prices)
    assert len(frame) == 5
    assert set(frame["source"]) == {"yahoo"}
    result = collect_market_data(DataConfig(tickers=("ABC",), root=tmp_path),
                                 downloader=lambda ticker, start, end: prices)
    assert result.status is ProviderStatus.SUCCESS
    assert (tmp_path / "processed" / "yahoo_observations.csv").exists()


def test_provider_failure_is_explicit(tmp_path):
    result = collect_market_data(DataConfig(tickers=("ABC",), root=tmp_path),
                                 downloader=lambda ticker, start, end: (_ for _ in ()).throw(
                                     RuntimeError("provider unavailable")))
    assert result.status is ProviderStatus.FAILED
    assert "provider unavailable" in result.error


def test_configuration_is_centralized_and_creates_paths(tmp_path):
    config = DataConfig(tickers=("SPY",), root=tmp_path)
    config.ensure_paths()
    assert config.raw_path.is_dir()
    assert config.processed_path.is_dir()
    assert config.cache_path.is_dir()
