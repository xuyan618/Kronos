from datetime import date

from app.data.schema import normalize_observations
from app.macro import calculate_macro_factors, calculate_macro_history


FIELDS = (
    "growth", "inflation", "liquidity", "rates", "yield_curve",
    "financial_conditions", "credit", "usd", "volatility", "risk_appetite",
)


def synthetic_frame():
    rows = []
    for index, observation_date in enumerate(
        (date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)), start=1
    ):
        for field in FIELDS:
            rows.append({
                "series_id": f"synthetic:{field}", "observation_date": observation_date,
                "available_at": f"2024-03-{index:02d}T00:00:00Z",
                "revision_id": f"r{index}", "value": float(index), "source": "synthetic",
                "unit": "value", "frequency": "monthly", "asset_name": "Synthetic",
                "category": "macro", "field": field, "availability_method": "release",
            })
    return normalize_observations(rows)


def test_factor_output_is_bounded_and_regime_is_explicit():
    result = calculate_macro_factors(synthetic_frame(), as_of="2024-03-15T00:00:00Z")
    row = result.iloc[0]
    assert -2 <= row["growth_score"] <= 2
    assert 0 <= row["macro_score"] <= 1
    assert row["primary_regime"] in {"EXPANSION", "STAGFLATION", "CONTRACTION", "TRANSITION", "UNKNOWN"}
    assert row["known_factors"] == len(FIELDS)


def test_missing_series_is_unknown_not_zero():
    frame = synthetic_frame().loc[lambda value: value["field"] == "growth"]
    row = calculate_macro_factors(frame, as_of="2024-03-15T00:00:00Z").iloc[0]
    assert row["inflation_state"] == "UNKNOWN"
    assert row["inflation_score"] is None
    assert row["primary_regime"] == "UNKNOWN"


def test_later_release_cannot_change_earlier_as_of_result():
    frame = synthetic_frame()
    frame = normalize_observations([
        *frame.to_dict("records"),
        {
            "series_id": "synthetic:growth", "observation_date": date(2024, 3, 1),
            "available_at": "2024-04-01T00:00:00Z", "revision_id": "late",
            "value": 99.0, "source": "synthetic", "unit": "value",
            "frequency": "monthly", "asset_name": "Synthetic", "category": "macro",
            "field": "growth", "availability_method": "release",
        },
    ])
    early = calculate_macro_factors(
        frame, as_of="2024-03-15T00:00:00Z", target_date=date(2024, 3, 1)
    ).iloc[0]
    later = calculate_macro_factors(
        frame, as_of="2024-04-15T00:00:00Z", target_date=date(2024, 3, 1)
    ).iloc[0]
    assert early["growth_score"] != later["growth_score"]
    assert early["as_of"] != later["as_of"]


def test_history_is_chronological_and_as_of_aware():
    history = calculate_macro_history(synthetic_frame(), as_of="2024-03-15T00:00:00Z")
    assert list(history["observation_date"]) == sorted(history["observation_date"])
    assert len(history) == 3
