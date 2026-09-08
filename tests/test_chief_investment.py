import json
from datetime import datetime, timezone

import pytest

from app.chief_investment import SpecialistReport, reconcile, serialize_report


AS_OF = datetime(2025, 1, 2, tzinfo=timezone.utc)


def report(name, direction, signal, *, evidence="KNOWN", **kwargs):
    return SpecialistReport(
        name, direction, signal, AS_OF, AS_OF, f"app/{name}", evidence=evidence, **kwargs
    )


def test_unanimous_bullish_is_buy():
    result = reconcile(
        [report(name, "BULLISH", 0.8) for name in ("macro", "technical", "kronos")],
        as_of=AS_OF,
        critical_specialists=(),
    )
    assert result.action == "BUY"
    assert result.score == pytest.approx(0.8)
    assert 0 <= result.confidence <= 1


def test_unanimous_bearish_is_sell():
    result = reconcile(
        [report(name, "BEARISH", -0.5) for name in ("macro", "technical")],
        as_of=AS_OF,
        critical_specialists=(),
    )
    assert result.action == "SELL"


def test_contradiction_requests_analysis_and_vetoes():
    result = reconcile(
        [report("macro", "BULLISH", 1), report("technical", "BEARISH", -1)],
        as_of=AS_OF,
        critical_specialists=(),
    )
    assert result.action == "NO_TRADE"
    assert "unresolved_contradiction" in result.vetoes
    assert any(item.action == "REQUEST_ANALYSIS" for item in result.additional_analysis)


def test_unknown_is_not_neutral_and_requests_evidence():
    result = reconcile(
        [report("macro", "NEUTRAL", 0), report("technical", "UNKNOWN", None, evidence="UNKNOWN")],
        as_of=AS_OF,
        critical_specialists=(),
    )
    assert result.action == "NO_TRADE"  # no directional signal from neutral evidence
    assert any(item.specialist == "technical" for item in result.additional_analysis)


def test_stale_report_is_hard_veto():
    stale = SpecialistReport("macro", "BULLISH", 1, "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "app/macro")
    result = reconcile([stale], as_of=AS_OF, max_age=3600, critical_specialists=())
    assert result.action == "NO_TRADE"
    assert "stale_report" in result.vetoes


def test_invalid_risk_is_hard_veto():
    result = reconcile(
        [report("macro", "BULLISH", 1), report("risk_decision", "BULLISH", 1, risk_valid=False, expected_value=10)],
        as_of=AS_OF,
    )
    assert result.action == "NO_TRADE"
    assert "invalid_risk" in result.vetoes


def test_confidence_and_json_are_deterministic():
    reports = [report("technical", "BULLISH", 0.5), report("macro", "BULLISH", 0.5)]
    first = reconcile(reports, as_of=AS_OF, critical_specialists=())
    second = reconcile(reversed(reports), as_of=AS_OF, critical_specialists=())
    assert first.to_dict() == second.to_dict()
    assert json.loads(serialize_report(first))["confidence_is_win_probability"] is False
