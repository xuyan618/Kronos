from __future__ import annotations

from datetime import date

import pytest

from app.data.schema import normalize_observations
from app.sectors import SectorRotationConfig, calculate_sector_rotation


SECTOR_FIELDS = (
    "technology",
    "financials",
    "healthcare",
    "energy",
)


def _make_observation(series_id: str, observation_date: date, value: float, *, available_at: str, field: str):
    return {
        "series_id": series_id,
        "observation_date": observation_date,
        "available_at": available_at,
        "revision_id": f"rev-{observation_date.isoformat()}-{series_id}",
        "value": value,
        "source": "synthetic",
        "unit": "index",
        "frequency": "daily",
        "asset_name": series_id,
        "category": "sector",
        "field": field,
        "availability_method": "release_timestamp",
    }


def synthetic_sector_frame():
    rows = []
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    benchmark_values = [100.0, 101.0, 102.0, 103.0]
    sector_values = {
        "technology": [100.0, 103.0, 108.0, 115.0],
        "financials": [100.0, 99.0, 98.0, 97.0],
        "healthcare": [100.0, 101.0, 101.5, 102.0],
        "energy": [100.0, 97.0, 94.0, 90.0],
    }
    benchmark_release_times = [
        "2024-01-01T00:00:00Z",
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
        "2024-01-04T00:00:00Z",
    ]
    for day, benchmark, available_at in zip(dates, benchmark_values, benchmark_release_times):
        rows.append(_make_observation("benchmark", day, benchmark, available_at=available_at, field="benchmark"))
    sector_release_times = [
        "2024-01-01T00:00:00Z",
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
        "2024-01-04T00:00:00Z",
    ]
    for field, values in sector_values.items():
        for day, value, available_at in zip(dates, values, sector_release_times):
            rows.append(_make_observation(f"sector:{field}", day, value, available_at=available_at, field=field))
    return normalize_observations(rows)


def test_sector_rotation_snapshot_has_as_of_and_leadership_labels():
    frame = synthetic_sector_frame()
    rotation = calculate_sector_rotation(
        frame,
        as_of="2024-01-04T00:00:00Z",
        sectors={"technology": ("technology",), "financials": ("financials",), "healthcare": ("healthcare",), "energy": ("energy",)},
        config=SectorRotationConfig(min_observations=3),
    )
    assert set(rotation["sector"]) == {"technology", "financials", "healthcare", "energy"}
    assert rotation["as_of"].nunique() == 1
    assert rotation["as_of"].iloc[0] == "2024-01-04T00:00:00Z"
    assert rotation["sector_score"].notna().sum() >= 3
    assert rotation.loc[rotation["sector"] == "technology", "label"].iloc[0] in {"LEADING", "NEUTRAL"}
    assert rotation["rank"].notna().sum() >= 1


def test_sector_rotation_is_as_of_safe_and_no_lookahead():
    frame = synthetic_sector_frame()
    future_revision = {
        "series_id": "sector:technology",
        "observation_date": date(2024, 1, 4),
        "available_at": "2024-01-10T00:00:00Z",
        "revision_id": "late-revision",
        "value": 500.0,
        "source": "synthetic",
        "unit": "index",
        "frequency": "daily",
        "asset_name": "sector:technology",
        "category": "sector",
        "field": "technology",
        "availability_method": "release_timestamp",
    }
    frame = normalize_observations([*frame.to_dict("records"), future_revision])
    early = calculate_sector_rotation(frame, as_of="2024-01-04T00:00:00Z", sectors={"technology": ("technology",)}, config=SectorRotationConfig(min_observations=3))
    late = calculate_sector_rotation(frame, as_of="2024-01-10T00:00:00Z", sectors={"technology": ("technology",)}, config=SectorRotationConfig(min_observations=3))
    assert early.loc[early["sector"] == "technology", "sector_score"].iloc[0] != late.loc[late["sector"] == "technology", "sector_score"].iloc[0]
    assert early["as_of"].iloc[0] == "2024-01-04T00:00:00Z"


def test_sector_rotation_handles_ties_and_missing_sectors():
    tied = []
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    benchmark = [100.0, 100.2, 100.4, 100.6]
    for day, value in zip(dates, benchmark):
        tied.append({
            "series_id": "benchmark",
            "observation_date": day,
            "available_at": f"2024-01-0{day.day}T00:00:00Z",
            "revision_id": f"bench-{day.isoformat()}",
            "value": value,
            "source": "synthetic",
            "unit": "index",
            "frequency": "daily",
            "asset_name": "Benchmark",
            "category": "benchmark",
            "field": "benchmark",
            "availability_method": "release_timestamp",
        })
    for name, values in {"technology": [100.0, 101.0, 102.0, 103.0], "financials": [100.0, 101.0, 102.0, 103.0]}.items():
        for day, value in zip(dates, values):
            tied.append({
                "series_id": f"sector:{name}",
                "observation_date": day,
                "available_at": f"2024-01-0{day.day+1}T00:00:00Z",
                "revision_id": f"{name}-{day.isoformat()}",
                "value": value,
                "source": "synthetic",
                "unit": "index",
                "frequency": "daily",
                "asset_name": name,
                "category": "sector",
                "field": name,
                "availability_method": "release_timestamp",
            })

    frame = normalize_observations(tied)
    rotation = calculate_sector_rotation(
        frame,
        as_of="2024-01-04T00:00:00Z",
        sectors={"technology": ("technology",), "financials": ("financials",), "utilities": ("utilities",)},
        config=SectorRotationConfig(min_observations=3),
    )
    active = rotation.loc[rotation["status"] == "ACTIVE"]
    if not active.empty:
        assert active["rank"].nunique() == 1
    missing = rotation.loc[rotation["sector"] == "utilities"]
    assert missing.empty or missing.iloc[0]["label"] == "UNKNOWN"
    assert rotation["label"].isin(["UNKNOWN", "LEADING", "LAGGING", "NEUTRAL"]).all()
