from datetime import date, timedelta

from app.cross_asset import (
    CrossAssetConfig,
    RelationshipDefinition,
    analyze_cross_asset,
    calculate_cross_asset_history,
)
from app.data.schema import normalize_observations


def observations(*, late_release=False, missing_right=False):
    rows = []
    start = date(2024, 1, 1)
    for index in range(6):
        day = start + timedelta(days=index)
        available = f"2024-01-{index + 1:02d}T18:00:00Z"
        rows.append({
            "series_id": "synthetic:stocks", "observation_date": day,
            "available_at": available, "revision_id": f"s{index}", "value": 100 + index * 2,
            "source": "synthetic", "unit": "price", "frequency": "D",
            "asset_name": "Stocks", "category": "market", "field": "stocks",
            "availability_method": "vendor_timestamp",
        })
        if not missing_right or index != 3:
            rows.append({
                "series_id": "synthetic:bonds", "observation_date": day,
                "available_at": available, "revision_id": f"b{index}", "value": 100 + index * .5,
                "source": "synthetic", "unit": "price", "frequency": "D",
                "asset_name": "Bonds", "category": "market", "field": "bonds",
                "availability_method": "vendor_timestamp",
            })
    if late_release:
        rows.append({
            "series_id": "synthetic:stocks", "observation_date": date(2024, 1, 5),
            "available_at": "2024-02-01T00:00:00Z", "revision_id": "late", "value": 1,
            "source": "synthetic", "unit": "price", "frequency": "D",
            "asset_name": "Stocks", "category": "market", "field": "stocks",
            "availability_method": "vendor_timestamp",
        })
    return normalize_observations(rows)


def relationship():
    return (RelationshipDefinition(
        "stocks_vs_bonds", ("stocks",), ("bonds",), "Stocks versus bonds.",
        lookback=4, threshold=.001, minimum_observations=3,
    ),)


def test_relationship_aligns_dates_and_explains_signal():
    report = analyze_cross_asset(
        observations(), as_of="2024-01-07T00:00:00Z", relationships=relationship(),
    )
    signal = report.signals[0]
    assert signal.state == "POSITIVE"
    assert signal.score == 2
    assert signal.observations == 5
    assert "relative return" in signal.explanation


def test_release_after_as_of_cannot_change_earlier_result():
    frame = observations(late_release=True)
    early = analyze_cross_asset(
        frame, as_of="2024-01-06T00:00:00Z", relationships=relationship(),
    ).signals[0]
    later = analyze_cross_asset(
        frame, as_of="2024-02-02T00:00:00Z", relationships=relationship(),
    ).signals[0]
    assert early.input_dates[-1] == "2024-01-05"
    assert early.metric != later.metric


def test_missing_alignment_is_unknown_not_zero():
    report = analyze_cross_asset(
        observations(missing_right=True), as_of="2024-01-07T00:00:00Z",
        config=CrossAssetConfig(minimum_observations=5), relationships=relationship(),
    )
    signal = report.signals[0]
    assert signal.state == "UNKNOWN"
    assert signal.score is None
    assert "required" in signal.limitations[0]


def test_history_is_chronological_and_serializable():
    history = calculate_cross_asset_history(
        observations(), as_of="2024-01-07T00:00:00Z", relationships=relationship(),
    )
    assert list(history["observation_date"]) == sorted(history["observation_date"])
    assert isinstance(history.iloc[-1]["signals"], list)
